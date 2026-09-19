#!/usr/bin/env python3
"""Concept-head OOD experiment (YOLO, Faster R-CNN, or RT-DETR; VOC or BDD).

Four stages, run in order:

  A  GT INDEX   decode YOLO `.txt` labels in ID-train tars -> data/id/gt_{voc,bdd}.json
  B  EXTRACT    one detector forward per image; cache float8 ROI features
                (YOLO 896x7x7, FRCNN 256x7x7)
                -> data/{detector}/{dataset}/roi/{id_train,id_val,near_ood,far_ood}.pt
  C  TRAIN+EVAL concept heads from data/{detector}/{dataset}/training_data.pt,
                SPK4 + native kNN, classwise Isolation Forest
                -> data/{detector}/{dataset}/concept_head_ood/seed_{seed}/
  D  VARIANTS   MDS / BAM / KNN / iForest on SPK features -> spk_variants/
                (runs after C by default; alone: --stage D)

Default is one concept-head seed (42). Pass `--seeds 42 43 44` for a 3-seed
run (stage A/B once; stage C retrains and rescores per seed, then mean±std).
Headline FPR95 / AUROC tables use no ID-val outlier removal. Optional spk_knn
mode drops each class's highest 5% ID-val kNN distances fit on standardized SPK4
(see --id-val-outlier-modes).

Checkpoints live under model/{detector}/ so one Colab session can hold YOLO, FRCNN, and RT-DETR.
Image tars under data/id and data/ood are shared across detectors.

All three detectors keep at most 30 boxes per image (`max_det=30`).

Examples
--------
    DETECTOR=yolo DATASET=voc bash mount_data.sh
    python run.py --root /content/spk --detector yolo --dataset voc
    python run.py --root /content/spk --detector yolo --dataset voc --stage D
    python run.py --root /content/spk --detector yolo --dataset voc --seeds 42 43 44

    DETECTOR=yolo DATASET=bdd bash mount_data.sh
    python run.py --root /content/spk --detector yolo --dataset bdd --max-images 200 --epochs 6
    python run.py --root /content/spk --detector yolo --dataset bdd --splits near_ood far_ood --force --extract-only
    # After mount_data.sh copied roi/native_knn/concept_head_ood/seed_*, this skips
    # extract and head training; it only builds gt_{dataset}.json if missing, then eval.
    python run.py --root /content/spk --detector yolo --dataset bdd

    DETECTOR=frcnn DATASET=voc bash mount_data.sh
    python run.py --root /content/spk --detector frcnn --dataset voc
    python run.py --root /content/spk --detector frcnn --dataset bdd
    python run.py --root /content/spk --detector rtdetr --dataset voc
    python run.py --root /content/spk --detector rtdetr --dataset bdd
    # checkpoint: /content/spk/model/frcnn/voc_vanilla.pth
    # outputs:    /content/spk/data/frcnn/voc/concept_head_ood/
    # backups:    /content/drive/MyDrive/experiments/{detector}-{dataset}/
    #             roi/  native_knn/  concept_head_ood/  gt_{dataset}.json  spk_variants/
    # assets:     /content/drive/MyDrive/assets/shared/datasets/id/{dataset}/gt_{dataset}.json
    #             (uploaded once when missing, for mount_data.sh --eval-only)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import pickle
import shutil
import tarfile
import time
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import cv2
import numpy as np
import ood_baseline
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision.ops import nms, roi_align
from tqdm.auto import tqdm

# `__file__` is absent when the script is piped into a remote kernel (Colab CLI),
# so fall back to the cwd and let --root override in every case.
ROOT = Path(globals().get("__file__", "run.py")).resolve().parent

VOC20 = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

VOC14 = frozenset({
    "bird", "bottle", "car", "cat", "cow", "dog", "horse", "person", "sheep",
    "bicycle", "boat", "bus", "chair", "train",
})

BDD10 = [
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
    "traffic_light", "traffic_sign",
]

# Detectron2 VOC-20 order used by m-hood FX `voc_vanilla.pth` (not alphabetical VOC).
FRCNN_VOC20 = (
    "person", "bird", "cat", "cow", "dog", "horse", "sheep", "airplane",
    "bicycle", "boat", "bus", "car", "motorcycle", "train", "bottle",
    "chair", "dining table", "potted plant", "couch", "tv",
)

FRCNN_BDD10 = (
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
    "traffic_light", "traffic_sign",
)


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    checkpoint: str
    splits: tuple[str, ...]
    id_splits: frozenset[str]
    split_tar_prefix: dict[str, str]
    split_output_names: dict[str, str]
    split_to_knn: dict[str, str]
    split_to_prior: dict[str, tuple[str, str]]
    native_split_specs: dict[str, dict[str, str]]
    label_classes: tuple[str, ...]
    eval_classes: frozenset[str]
    gt_train_glob: str
    gt_fixed_size: tuple[int, int] | None
    roi_cache_type: str
    strip_train_prefix: bool = False
    # Detector `names` -> eval/GT names (YOLO-BDD pedestrian; FRCNN airplane/tv).
    label_aliases: dict[str, str] = field(default_factory=dict)
    detector: str = "yolo"
    feature_channels: int = 896
    max_det: int = 30
    detector_class_names: tuple[str, ...] | None = None
    frcnn_config: str = "model/frcnn/frcnn_fx/FX_vanilla_voc.yaml"
    roi_feature_definition: str = "YOLO neck multiscale ROIAlign 896x7x7; single-pass unified forward"
    native_feature_definition: str = "YOLO neck P3/P4/P5 global mean+std pooling"


def make_dataset_profile(
    name: str,
    train_prefix: str,
    val_prefix: str,
    label_classes: tuple[str, ...],
    eval_classes: frozenset[str],
    *,
    fixed_size: tuple[int, int] | None = None,
    strip_train_prefix: bool = False,
    label_aliases: dict[str, str] | None = None,
) -> DatasetProfile:
    """VOC 和 BDD 使用相同的目录约定，仅在调用处列出不同配置。"""
    train, val = f"{name}_train", f"{name}_val"
    return DatasetProfile(
        name=name,
        checkpoint=f"{name}_vanilla.pt",
        splits=(train, val, "near_ood", "far_ood"),
        id_splits=frozenset({train, val}),
        split_tar_prefix={
            train: train_prefix, val: val_prefix,
            "near_ood": f"near_ood_{name}", "far_ood": "far_ood",
        },
        split_output_names={
            train: "id_train.pt", val: "id_val.pt",
            "near_ood": "near_ood.pt", "far_ood": "far_ood.pt",
        },
        split_to_knn={
            train: f"{name}_id_train", val: f"{name}_id_val",
            "near_ood": f"near_ood_{name}", "far_ood": f"far_ood_{name}",
        },
        split_to_prior={
            train: ("train_detector_rows.csv", "id_train_tp"),
            val: ("val_detector_rows.csv", "id_val_tp"),
            "near_ood": ("near_detector_rows.csv", "near_ood_fp"),
            "far_ood": ("far_detector_rows.csv", "far_ood_fp"),
        },
        native_split_specs={
            f"{name}_id_train": {"data_source": "id_train_tp", "protocol": f"{name}_fp8_train_pool"},
            f"{name}_id_val": {"data_source": "id_val_tp", "protocol": f"{name}_id"},
            f"near_ood_{name}": {"data_source": "near_ood_fp", "protocol": "near_far_ood"},
            f"far_ood_{name}": {"data_source": "far_ood_fp", "protocol": "near_far_ood"},
        },
        label_classes=label_classes,
        eval_classes=eval_classes,
        gt_train_glob=f"{train_prefix}-*.tar",
        gt_fixed_size=fixed_size,
        roi_cache_type=f"{name}_fp8_roi_features",
        strip_train_prefix=strip_train_prefix,
        label_aliases=label_aliases or {},
    )


VOC_PROFILE = make_dataset_profile(
    "voc", "voc_yolo_train", "voc_yolo_val", tuple(VOC20), VOC14,
)
BDD_PROFILE = make_dataset_profile(
    "bdd", "bdd_train_10k", "bdd_val", tuple(BDD10), frozenset(BDD10),
    fixed_size=(1280, 720),
    strip_train_prefix=True,
    label_aliases={
        "pedestrian": "person",
        "traffic light": "traffic_light",
        "traffic sign": "traffic_sign",
    },
)

FRCNN_VOC_PROFILE = replace(
    VOC_PROFILE,
    detector="frcnn",
    checkpoint="voc_vanilla.pth",
    feature_channels=256,
    max_det=30,
    roi_cache_type="frcnn_voc_fp8_roi_features",
    detector_class_names=FRCNN_VOC20,
    label_aliases={
        "airplane": "aeroplane",
        "motorcycle": "motorbike",
        "dining table": "diningtable",
        "potted plant": "pottedplant",
        "couch": "sofa",
        "tv": "tvmonitor",
    },
    roi_feature_definition="Faster R-CNN FPN p2-p5 ROIAlign 256x7x7; max_det=30",
    native_feature_definition="Faster R-CNN FPN p2-p5 global mean+std pooling -> 2048-d vector",
)

FRCNN_BDD_PROFILE = replace(
    BDD_PROFILE,
    detector="frcnn",
    checkpoint="bdd_vanilla.pth",
    feature_channels=256,
    roi_cache_type="frcnn_bdd_fp8_roi_features",
    detector_class_names=FRCNN_BDD10,
    frcnn_config="model/frcnn/frcnn_fx/FX_vanilla_bdd.yaml",
    roi_feature_definition="Faster R-CNN FPN p2-p5 ROIAlign 256x7x7; max_det=30",
    native_feature_definition="Faster R-CNN FPN p2-p5 global mean+std pooling -> 2048-d vector",
)

RTDETR_VOC_PROFILE = replace(
    VOC_PROFILE,
    detector="rtdetr",
    checkpoint="voc_vanilla.pt",
    roi_cache_type="rtdetr_voc_fp8_roi_features",
    feature_channels=768,
    roi_feature_definition="RT-DETR hybrid-encoder multi-scale ROIAlign; NMS-free top-30",
    native_feature_definition="RT-DETR encoder feature-map global mean+std pooling",
)

RTDETR_BDD_PROFILE = replace(
    BDD_PROFILE,
    detector="rtdetr",
    checkpoint="bdd_vanilla.pt",
    roi_cache_type="rtdetr_bdd_fp8_roi_features",
    feature_channels=768,
    roi_feature_definition="RT-DETR hybrid-encoder multi-scale ROIAlign; NMS-free top-30",
    native_feature_definition="RT-DETR encoder feature-map global mean+std pooling",
)

PROFILES: dict[tuple[str, str], DatasetProfile] = {
    ("yolo", "voc"): VOC_PROFILE,
    ("yolo", "bdd"): BDD_PROFILE,
    ("frcnn", "voc"): FRCNN_VOC_PROFILE,
    ("frcnn", "bdd"): FRCNN_BDD_PROFILE,
    ("rtdetr", "voc"): RTDETR_VOC_PROFILE,
    ("rtdetr", "bdd"): RTDETR_BDD_PROFILE,
}
PROFILE = VOC_PROFILE


def set_profile(detector: str, dataset: str) -> None:
    """Switch detector + VOC/BDD tar prefixes, classes, and checkpoint."""
    global PROFILE, SPLITS, SPLIT_TAR_PREFIX, SPLIT_OUTPUT_NAMES, SPLIT_TO_KNN
    global SPLIT_TO_PRIOR, NATIVE_SPLIT_SPECS
    try:
        PROFILE = PROFILES[(detector, dataset)]
    except KeyError:
        known = ", ".join(f"{d}/{s}" for d, s in sorted(PROFILES))
        raise SystemExit(f"unknown --detector {detector!r} --dataset {dataset!r} (choose from {known})") from None
    SPLITS = PROFILE.splits
    SPLIT_TAR_PREFIX = PROFILE.split_tar_prefix
    SPLIT_OUTPUT_NAMES = PROFILE.split_output_names
    SPLIT_TO_KNN = PROFILE.split_to_knn
    SPLIT_TO_PRIOR = PROFILE.split_to_prior
    NATIVE_SPLIT_SPECS = PROFILE.native_split_specs
    refresh_paths()


def set_root(root: Path) -> None:
    """Re-anchor every asset and output path at `root` (used by --root)."""
    global ROOT
    ROOT = root.expanduser().resolve()
    refresh_paths()


def refresh_paths() -> None:
    global CHECKPOINT, DATASET_DIR, OOD_DIR, ROI_DIR, GT_INDEX
    global TRAINING_DATA, NATIVE_ROOT, PRIOR_DIR, FRCNN_CFG, ARCH_DIR, SPK_VARIANTS_DIR
    CHECKPOINT = ROOT / "model" / PROFILE.detector / PROFILE.checkpoint
    DATASET_DIR = ROOT / "data/id"
    OOD_DIR = ROOT / "data/ood"
    ARCH_DIR = ROOT / "data" / PROFILE.detector / PROFILE.name
    ROI_DIR = ARCH_DIR / "roi"
    GT_INDEX = ROOT / "data/id" / f"gt_{PROFILE.name}.json"
    TRAINING_DATA = ARCH_DIR / "training_data.pt"
    NATIVE_ROOT = ARCH_DIR / "native_knn"
    SPK_VARIANTS_DIR = ARCH_DIR / "spk_variants"
    PRIOR_DIR = ARCH_DIR / "detection_prior_rows"
    FRCNN_CFG = ROOT / PROFILE.frcnn_config


refresh_paths()

def resolve_training_data(path: Path) -> Path:
    """Drive/rsync sometimes materializes `training_data.pt` as a directory."""
    if path.is_file():
        return path
    if path.is_dir():
        nested = [p for p in path.rglob("*.pt") if p.is_file()]
        named = [p for p in nested if p.name == "training_data.pt"]
        chosen = named or nested
        if chosen:
            print(f"  {path} is a directory; using {chosen[0]}", flush=True)
            return chosen[0]
        raise IsADirectoryError(
            f"{path} is a directory with no .pt inside; copy the file onto the VM"
        )
    raise FileNotFoundError(path)


def stage_c_can_reuse_activations(args: argparse.Namespace) -> bool:
    """True when stage C can eval from activations.csv without training_data.pt."""
    if getattr(args, "retrain", False) or getattr(args, "rescore", False):
        return False
    seeds = list(getattr(args, "seeds", [getattr(args, "seed", SEED)]))
    out_root = getattr(args, "out_root", args.out)
    flat = out_root / "activations.csv"
    for seed in seeds:
        path = seed_dir(out_root, seed) / "activations.csv"
        if path.is_file() or (len(seeds) == 1 and flat.is_file()):
            continue
        return False
    return True


# Canonical extraction settings (see README); not exposed as flags on purpose.
# max_det is always 30 (YOLO NMS cap and FRCNN TEST.DETECTIONS_PER_IMAGE).
CONF = 0.25
MAX_DET = 30
IOU = 0.7
IMGSZ = 640
ROI_SIZE = 7
EXTRACT_BATCH = 16
NATIVE_CHUNK_SIZE = 512
NATIVE_DTYPE = torch.float16
HEAD_BATCH = 64
FP8_E4M3_MAX = 448.0
FEATURE_CHANNELS = 896
VRAM_FRACTION = 0.70

SPLITS = PROFILE.splits
SPLIT_TAR_PREFIX = PROFILE.split_tar_prefix
SPLIT_OUTPUT_NAMES = PROFILE.split_output_names
SPLIT_TO_KNN = PROFILE.split_to_knn
SPLIT_TO_PRIOR = PROFILE.split_to_prior
NATIVE_SPLIT_SPECS = PROFILE.native_split_specs

PRIOR_FIELDS = ["class", "data_source", "image_path", "file_name", "bbox_xyxy", "detector_confidence"]

# The 4 numbers that define `spk local` (see SPK_METHODS.md). Order matters only
# for readability; the Isolation Forest treats them as an unordered feature set.
SPK4_COLS = ["known_max", "unknown", "proxy_max", "relative_area"]
KNN_COL = "native_knn"
SPK_FULL_COLS = SPK4_COLS + [KNN_COL]
NATIVE_K = 5
ID_VAL_OUTLIER_FRACTION = 0.05
ID_VAL_OUTLIER_MODES = ("none", "iforest", "spk_knn")
ID_VAL_OUTLIER_SETTING = {
    "none": "without_outlier_removal",
    "iforest": "with_outlier_removal",
    "spk_knn": "with_spk_knn_outlier_removal",
}
SPK_KNN_OUTLIER_COLS = SPK4_COLS
SPK_OUTLIER_KNN_K = 5

# ROI caches consumed by stage C, and the split name we give each one.
ROI_SPLITS = {
    "id_train": "id_train.pt",
    "id_val": "id_val.pt",
    "near_ood": "near_ood.pt",
    "far_ood": "far_ood.pt",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _cuda_index(device: str | torch.device) -> int | None:
    if not torch.cuda.is_available():
        return None
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return None
    return resolved.index if resolved.index is not None else torch.cuda.current_device()


def vram_budget_bytes(device: str | torch.device, fraction: float) -> int | None:
    index = _cuda_index(device)
    if index is None:
        return None
    total = torch.cuda.get_device_properties(index).total_memory
    return int(total * fraction)


def choose_batch_size(
    trial,
    device: str | torch.device,
    fraction: float,
    *,
    min_batch: int = 1,
    max_batch: int = 256,
    default: int | None = None,
    label: str = "batch",
) -> int:
    """Largest batch whose peak allocation stays under `fraction` of VRAM."""
    budget = vram_budget_bytes(device, fraction)
    if budget is None:
        fallback = default if default is not None else min_batch
        print(f"  {label}: CPU / no CUDA, using {fallback}", flush=True)
        return fallback
    index = _cuda_index(device)
    assert index is not None
    total = torch.cuda.get_device_properties(index).total_memory
    torch.cuda.empty_cache()

    def _peak(batch: int) -> int | None:
        torch.cuda.reset_peak_memory_stats(index)
        torch.cuda.empty_cache()
        try:
            trial(batch)
            torch.cuda.synchronize(index)
            return int(torch.cuda.max_memory_allocated(index))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None

    lo, hi = min_batch, min_batch
    best = min_batch
    peak = _peak(min_batch)
    if peak is None:
        print(f"  {label}: even batch={min_batch} OOM; using {min_batch}", flush=True)
        return min_batch
    if peak > budget:
        print(
            f"  {label}: batch={min_batch} already uses {peak / 1e9:.1f} GB "
            f"(budget {budget / 1e9:.1f} GB of {total / 1e9:.1f} GB); keeping {min_batch}",
            flush=True,
        )
        return min_batch

    # 先逐次翻倍，找到超过显存预算的上界，再用二分搜索缩小范围。
    while hi < max_batch:
        hi = min(hi * 2, max_batch)
        peak = _peak(hi)
        if peak is None or peak > budget:
            break
        best = lo = hi

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        peak = _peak(mid)
        if peak is not None and peak <= budget:
            best = lo = mid
        else:
            hi = mid

    peak = _peak(best) or 0
    print(
        f"  {label}: {best}  ({peak / 1e9:.1f} GB peak, "
        f"budget {budget / 1e9:.1f} GB = {fraction:.0%} of {total / 1e9:.1f} GB)",
        flush=True,
    )
    return best


def tune_extract_batch(engine, args: argparse.Namespace) -> int:
    if args.extract_batch > 0:
        print(f"  extract batch: {args.extract_batch} (manual)", flush=True)
        return args.extract_batch
    if PROFILE.detector == "frcnn":
        # Detectron2 ImageList pads the whole batch to the largest H/W. Far-OOD
        # images can be 4k, so GPU batch>1 OOMs even when an 800x1200 trial looks fine.
        # infer_batch_bgr therefore runs one image at a time; this number is only
        # how many tars we decode before that loop.
        print("  extract batch: 8 (FRCNN: decode 8, infer 1 at a time)", flush=True)
        return 8
    if PROFILE.detector == "rtdetr":
        dummy = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)

        def trial(batch: int) -> None:
            engine.infer_batch_bgr([dummy] * batch)

        chosen = choose_batch_size(
            trial, args.device, args.vram_frac,
            min_batch=1, max_batch=32, default=EXTRACT_BATCH, label="extract batch",
        )
        torch.cuda.empty_cache()
        return max(chosen, 1)

    dummy = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
    worst_boxes = torch.tensor(
        [[8.0, 8.0, 64.0, 64.0]] * PROFILE.max_det, dtype=torch.float32,
    )

    def trial(batch: int) -> None:
        tensors, infos = [], []
        for _ in range(batch):
            tensor, info = engine._letterbox_rgb(dummy)
            tensors.append(tensor)
            infos.append(info)
        batch_tensor = torch.cat(tensors, dim=0)
        neck_levels, preds = engine._forward_batch(batch_tensor)
        nms_preds = engine._nms_detect_preds(preds)
        for batch_idx in range(batch):
            feats = [level[batch_idx] for level in neck_levels]
            engine._pool_native_from_neck(feats, infos[batch_idx])
            boxes = worst_boxes
            engine._roi_align_from_neck(feats, engine._project_boxes(boxes.to(engine.device), infos[batch_idx]))

    chosen = choose_batch_size(
        trial, args.device, args.vram_frac,
        min_batch=1, max_batch=256, default=EXTRACT_BATCH, label="extract batch",
    )
    torch.cuda.empty_cache()
    return max(chosen, 1)


def _ensure_extract_batch(engine, args: argparse.Namespace) -> None:
    if int(getattr(args, "extract_batch_resolved", 0) or 0) > 0:
        return
    args.extract_batch_resolved = tune_extract_batch(engine, args)


def tune_head_batch(in_channels: int, n_concepts: int, args: argparse.Namespace) -> int:
    if args.head_batch > 0:
        return args.head_batch
    cache_key = (in_channels, n_concepts)
    cached = getattr(args, "_head_batch_cache", None)
    if cached is None:
        args._head_batch_cache = {}
        cached = args._head_batch_cache
    if cache_key in cached:
        return cached[cache_key]
    if _cuda_index(args.device) is None:
        cached[cache_key] = HEAD_BATCH
        return HEAD_BATCH

    device = torch.device(args.device)

    def trial(batch: int) -> None:
        head = ConceptHead(in_channels, n_concepts).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=5e-4)
        xb = torch.randn(batch, in_channels, ROI_SIZE, ROI_SIZE, device=device)
        yb = torch.zeros(batch, n_concepts, ROI_SIZE, ROI_SIZE, device=device)
        yb[:, 0] = 1.0
        groups = [slice(0, max(n_concepts - 1, 1)), slice(max(n_concepts - 1, 1), n_concepts)]
        loss = total_loss(head(xb), yb, groups)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        del head, optimizer, xb, yb, loss

    chosen = choose_batch_size(
        trial, args.device, args.vram_frac,
        min_batch=8, max_batch=4096, default=HEAD_BATCH, label=f"head batch ({n_concepts} concepts)",
    )
    torch.cuda.empty_cache()
    cached[cache_key] = chosen
    return chosen


def canonical_file_name(name: str) -> str:
    text = Path(str(name).strip()).name
    if PROFILE.strip_train_prefix and text.startswith("train_") and text.count("_") >= 2:
        text = text.rsplit("_", 1)[-1]
    if not text.endswith(".jpg"):
        text = f"{text}.jpg"
    return text


# ===========================================================================
# Stage A: ground-truth box index
# ===========================================================================

def _boxes_from_yolo_labels(
    text: str, class_names: tuple[str, ...], width: int, height: int
) -> list[dict]:
    boxes = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        class_id = int(float(fields[0]))
        if not 0 <= class_id < len(class_names):
            continue
        xc, yc, box_width, box_height = map(float, fields[1:5])
        boxes.append({
            "class": class_names[class_id],
            "bbox_xyxy": [
                (xc - box_width / 2) * width, (yc - box_height / 2) * height,
                (xc + box_width / 2) * width, (yc + box_height / 2) * height,
            ],
        })
    return boxes


def build_gt_index(source_dir: Path, output: Path) -> None:
    """Extract absolute ID-train boxes out of the YOLO-label tar shards."""
    paths = sorted(source_dir.glob(PROFILE.gt_train_glob))
    if not paths:
        raise FileNotFoundError(f"No {PROFILE.gt_train_glob} under {source_dir}")

    index: dict[str, list[dict]] = {}
    class_names = PROFILE.label_classes
    fixed_size = PROFILE.gt_fixed_size
    for path in paths:
        with tarfile.open(path) as archive:
            if fixed_size is not None:
                width, height = fixed_size
                members = [m for m in archive.getmembers() if m.isfile() and Path(m.name).suffix.lower() == ".txt"]
                for label_member in tqdm(members, desc=f"gt {path.name}"):
                    label_handle = archive.extractfile(label_member)
                    if label_handle is None:
                        continue
                    stem = Path(label_member.name).stem
                    index[stem] = _boxes_from_yolo_labels(
                        label_handle.read().decode(), class_names, width, height
                    )
                continue

            members = {m.name: m for m in archive.getmembers() if m.isfile()}
            images = {Path(n).stem: m for n, m in members.items() if Path(n).suffix.lower() in IMAGE_SUFFIXES}
            labels = {Path(n).stem: m for n, m in members.items() if Path(n).suffix.lower() == ".txt"}
            for stem, label_member in tqdm(sorted(labels.items()), desc=f"gt {path.name}"):
                image_member = images.get(stem)
                if image_member is None:
                    raise ValueError(f"{path}: image missing for {stem}")
                image_handle = archive.extractfile(image_member)
                label_handle = archive.extractfile(label_member)
                if image_handle is None or label_handle is None:
                    raise ValueError(f"{path}: unreadable members for {stem}")
                with Image.open(io.BytesIO(image_handle.read())) as image:
                    width, height = image.size
                index[stem] = _boxes_from_yolo_labels(
                    label_handle.read().decode(), class_names, width, height
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index, separators=(",", ":")) + "\n")
    print(f"  {len(paths)} tars, {len(index)} images -> {output}")


def _sample_id_train_stems(source_dir: Path, limit: int = 16) -> list[str]:
    paths = sorted(source_dir.glob(PROFILE.gt_train_glob))
    if not paths:
        return []
    stems: list[str] = []
    with tarfile.open(paths[0]) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            suffix = Path(member.name).suffix.lower()
            if suffix not in IMAGE_SUFFIXES and suffix != ".txt":
                continue
            stems.append(Path(member.name).stem)
            if len(stems) >= limit:
                break
    return stems


def gt_index_is_current(path: Path, source_dir: Path) -> bool:
    """True if `path` indexes this dataset's ID-train tars (not a leftover VOC file)."""
    if not path.is_file():
        return False
    try:
        index = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(index, dict) or not index:
        return False
    stems = _sample_id_train_stems(source_dir)
    if not stems:
        return False
    hits = sum(stem in index for stem in stems)
    return hits >= max(1, len(stems) // 2)


# ===========================================================================
# Stage B: detector forward + ROI caches
# ===========================================================================

@dataclass(frozen=True)
class ImageItem:
    file_name: str
    tar_member: str
    tar_path: Path


class VocSplitImageSource:
    """Lazy image source reading every split from tar shards."""

    def __init__(self, split: str, max_images: int = 0) -> None:
        self.split = split
        self.max_images = max_images
        self._tar: tarfile.TarFile | None = None
        self._members_by_tar: dict[Path, list[tarfile.TarInfo]] = {}

        search_dir = DATASET_DIR if split in PROFILE.id_splits else OOD_DIR
        self.tar_paths = sorted(search_dir.glob(f"{SPLIT_TAR_PREFIX[split]}-*.tar"))
        if not self.tar_paths:
            raise FileNotFoundError(
                f"No {SPLIT_TAR_PREFIX[split]}-*.tar shards for split {split} in {search_dir}"
            )

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None

    def __enter__(self) -> VocSplitImageSource:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _get_members(self, tar_path: Path) -> list[tarfile.TarInfo]:
        if tar_path not in self._members_by_tar:
            with tarfile.open(tar_path, "r") as tf:
                members = [
                    m for m in tf.getmembers()
                    if m.isfile() and Path(m.name).suffix.lower() in IMAGE_SUFFIXES
                ]
                self._members_by_tar[tar_path] = sorted(members, key=lambda m: m.name)
        return self._members_by_tar[tar_path]

    def count_images(self) -> int:
        total = sum(len(self._get_members(tar_path)) for tar_path in self.tar_paths)
        return min(total, self.max_images) if self.max_images > 0 else total

    def iter_items(self) -> Iterator[ImageItem]:
        count = 0
        for tar_path in self.tar_paths:
            for member in self._get_members(tar_path):
                if self.max_images > 0 and count >= self.max_images:
                    return
                yield ImageItem(
                    file_name=Path(member.name).name,
                    tar_member=member.name,
                    tar_path=tar_path,
                )
                count += 1

    def load_bgr(self, item: ImageItem) -> np.ndarray | None:
        if self._tar is None or self._tar.name != str(item.tar_path):
            if self._tar is not None:
                self._tar.close()
            self._tar = tarfile.open(item.tar_path, "r")

        payload = self._tar.extractfile(item.tar_member)
        if payload is None:
            return None
        arr = np.frombuffer(payload.read(), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def image_uri(self, item: ImageItem) -> str:
        return f"tar://{item.tar_path.name}::{item.tar_member}"


@dataclass
class UnifiedImageResult:
    detections: list[dict[str, Any]]
    native_embedding: torch.Tensor
    roi_features: torch.Tensor


class YOLOUnifiedForward:
    """One model forward per batch; fan out to detections, native embedding, and ROI features."""

    def __init__(self, checkpoint_path: str | Path, device: str | torch.device) -> None:
        from ultralytics import YOLO

        wrapper = YOLO(str(checkpoint_path))
        self.yolo = wrapper
        self.model = wrapper.model.to(device).eval()
        self.names = wrapper.names
        self.device = torch.device(device)
        self.strides = [8, 16, 32]
        self.detect_input_indices = (16, 19, 22)
        self.detect_layer_index = 23

        for param in self.model.parameters():
            param.requires_grad = False

    def _letterbox_rgb(self, image_rgb: np.ndarray) -> tuple[torch.Tensor, dict[str, Any]]:
        h, w = image_rgb.shape[:2]
        ratio = min(IMGSZ / h, IMGSZ / w)
        nw, nh = int(round(w * ratio)), int(round(h * ratio))
        resized = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
        pad_l = int(round((IMGSZ - nw) / 2 - 0.1))
        pad_t = int(round((IMGSZ - nh) / 2 - 0.1))
        canvas[pad_t : pad_t + nh, pad_l : pad_l + nw] = resized
        tensor = torch.from_numpy(canvas).permute(2, 0, 1).float().div(255.0)
        return tensor.unsqueeze(0), {
            "orig_hw": (h, w),
            "ratio": ratio,
            "pad_l": pad_l,
            "pad_t": pad_t,
            "nw": nw,
            "nh": nh,
        }

    def _project_boxes(self, boxes_xyxy: torch.Tensor, info: dict[str, Any]) -> torch.Tensor:
        boxes = boxes_xyxy.clone().to(torch.float32)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * info["ratio"] + info["pad_l"]
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * info["ratio"] + info["pad_t"]

        limit = float(IMGSZ - 1)
        boxes[:, 0] = boxes[:, 0].clamp_(0.0, limit)
        boxes[:, 1] = boxes[:, 1].clamp_(0.0, limit)
        boxes[:, 2] = boxes[:, 2].clamp_(min=1.0, max=float(IMGSZ))
        boxes[:, 3] = boxes[:, 3].clamp_(min=1.0, max=float(IMGSZ))
        boxes[:, 2] = torch.maximum(boxes[:, 2], boxes[:, 0] + 1.0)
        boxes[:, 3] = torch.maximum(boxes[:, 3], boxes[:, 1] + 1.0)
        return boxes

    @staticmethod
    def _xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
        xyxy = xywh.clone()
        xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
        return xyxy

    def _nms_detect_preds(self, preds: torch.Tensor) -> list[torch.Tensor]:
        """Turn Detect-head (B, 4+nc, N) xywh+scores into per-image (K, 6) xyxy rows."""
        pred = preds.float()
        if pred.ndim != 3:
            raise ValueError(f"unexpected pred shape {tuple(pred.shape)}")
        if pred.shape[1] > pred.shape[2]:
            pred = pred.transpose(1, 2)
        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:]
        outputs: list[torch.Tensor] = []
        class_offset = 7680.0  # keep class-aware NMS from suppressing across classes
        for image_idx in range(pred.shape[0]):
            scores, labels = cls_scores[image_idx].max(dim=0)
            keep = scores > CONF
            if not keep.any():
                outputs.append(pred.new_zeros((0, 6)))
                continue
            xyxy = self._xywh_to_xyxy(boxes_xywh[image_idx].T[keep])
            scores = scores[keep]
            labels = labels[keep].to(dtype=xyxy.dtype)
            keep_idx = nms(xyxy + labels.unsqueeze(1) * class_offset, scores, IOU)
            keep_idx = keep_idx[: PROFILE.max_det]
            outputs.append(
                torch.cat([xyxy[keep_idx], scores[keep_idx, None], labels[keep_idx, None]], dim=1)
            )
        return outputs

    @torch.inference_mode()
    def _forward_batch(self, batch_tensor: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        outputs: list[torch.Tensor | None] = []
        current: torch.Tensor | list[torch.Tensor] = batch_tensor.to(self.device)
        feats: list[torch.Tensor] = []
        use_amp = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
            for layer in self.model.model:
                if layer.i == self.detect_layer_index:
                    break
                if layer.f != -1:
                    current = (
                        outputs[layer.f]
                        if isinstance(layer.f, int)
                        else [current if j == -1 else outputs[j] for j in layer.f]
                    )
                current = layer(current)
                outputs.append(current if layer.i in self.model.save else None)
                if layer.i in self.detect_input_indices:
                    feats.append(current.detach())
            preds, _ = self.model.model[self.detect_layer_index]([f.clone() for f in feats])
        return feats, preds

    def _pool_native_from_neck(
        self, feats_per_image: list[torch.Tensor], info: dict[str, Any]
    ) -> torch.Tensor:
        pooled: list[torch.Tensor] = []
        for feat, stride in zip(feats_per_image, self.strides):
            h_feat, w_feat = feat.shape[1], feat.shape[2]  # Shape is [C, H, W]
            y_min = max(0, min(int(round(info["pad_t"] / stride)), h_feat))
            y_max = max(0, min(y_min + int(round(info["nh"] / stride)), h_feat))
            x_min = max(0, min(int(round(info["pad_l"] / stride)), w_feat))
            x_max = max(0, min(x_min + int(round(info["nw"] / stride)), w_feat))
            if y_min >= y_max:
                y_min, y_max = 0, h_feat
            if x_min >= x_max:
                x_min, x_max = 0, w_feat
            valid_feat = feat[:, y_min:y_max, x_min:x_max]
            pooled.append(valid_feat.mean(dim=(1, 2)))
            pooled.append(valid_feat.std(dim=(1, 2), unbiased=False))
        return torch.cat(pooled, dim=0)

    def _roi_align_from_neck(
        self, feats_per_image: list[torch.Tensor], boxes_model: torch.Tensor
    ) -> torch.Tensor:
        total_channels = sum(f.shape[0] for f in feats_per_image)
        if boxes_model.numel() == 0:
            return torch.empty((0, total_channels, ROI_SIZE, ROI_SIZE), dtype=torch.float32)
        batch_idx = torch.zeros((boxes_model.shape[0], 1), dtype=torch.float32, device=self.device)
        rois = torch.cat([batch_idx, boxes_model.to(self.device)], dim=1).to(dtype=feats_per_image[0].dtype)
        pooled = [
            roi_align(
                feat.unsqueeze(0),
                rois,
                output_size=(ROI_SIZE, ROI_SIZE),
                spatial_scale=1.0 / float(stride),
                aligned=True,
            )
            for feat, stride in zip(feats_per_image, self.strides)
        ]
        return torch.cat(pooled, dim=1).cpu()

    def _parse_pred_row(self, pred: torch.Tensor) -> list[dict[str, Any]]:
        # After NMS: [N, 6] = (x1, y1, x2, y2, score, class_id) in letterbox pixels.
        if pred is None or pred.numel() == 0:
            return []
        detections: list[dict[str, Any]] = []
        for row in pred:
            raw_label = str(self.names[int(row[5].item())]).strip().lower()
            pred_class = PROFILE.label_aliases.get(raw_label, raw_label)
            if pred_class in PROFILE.eval_classes:
                detections.append(
                    {
                        "pred_class": pred_class,
                        "detector_class": pred_class,
                        "detector_label": raw_label,
                        "detector_confidence": float(row[4].item()),
                        "bbox_xyxy": row[:4].detach().cpu().tolist(),
                    }
                )
        return detections

    def _scale_detections_to_orig(
        self, detections: list[dict[str, Any]], info: dict[str, Any]
    ) -> list[dict[str, Any]]:
        scaled: list[dict[str, Any]] = []
        for det in detections:
            x1_m, y1_m, x2_m, y2_m = det["bbox_xyxy"]
            scaled.append(
                {
                    **det,
                    "bbox_xyxy": [
                        float((x1_m - info["pad_l"]) / info["ratio"]),
                        float((y1_m - info["pad_t"]) / info["ratio"]),
                        float((x2_m - info["pad_l"]) / info["ratio"]),
                        float((y2_m - info["pad_t"]) / info["ratio"]),
                    ],
                }
            )
        return scaled

    @torch.inference_mode()
    def infer_batch_bgr(self, images_bgr: list[np.ndarray]) -> list[UnifiedImageResult]:
        if not images_bgr:
            return []

        tensors: list[torch.Tensor] = []
        infos: list[dict[str, Any]] = []
        for image_bgr in images_bgr:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            tensor, info = self._letterbox_rgb(image_rgb)
            tensors.append(tensor)
            infos.append(info)

        batch_tensor = torch.cat(tensors, dim=0)
        neck_levels, preds = self._forward_batch(batch_tensor)
        # Detect-head output is (B, 4+nc, 8400) xywh+scores, not post-NMS (N, 6).
        nms_preds = self._nms_detect_preds(preds)

        results: list[UnifiedImageResult] = []
        for batch_idx in range(batch_tensor.shape[0]):
            feats_per_image = [level[batch_idx] for level in neck_levels]
            native = self._pool_native_from_neck(feats_per_image, infos[batch_idx]).detach().cpu()

            detections_norm = self._parse_pred_row(nms_preds[batch_idx])
            detections = self._scale_detections_to_orig(detections_norm, infos[batch_idx])

            if detections:
                boxes = torch.tensor([d["bbox_xyxy"] for d in detections], dtype=torch.float32)
                roi = self._roi_align_from_neck(feats_per_image, self._project_boxes(boxes, infos[batch_idx]))
            else:
                total_channels = sum(f.shape[0] for f in feats_per_image)
                roi = torch.empty((0, total_channels, ROI_SIZE, ROI_SIZE), dtype=torch.float32)

            results.append(UnifiedImageResult(detections, native, roi))
        return results


def _last_decoder_layer(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        return tensor[-1]
    if tensor.ndim == 3:
        return tensor
    raise ValueError(f"unexpected decoder tensor shape {tuple(tensor.shape)}")


class RTDETRUnifiedForward:
    """Ultralytics RT-DETR: NMS-free detections + encoder ROIAlign + mean+std native."""

    def __init__(self, checkpoint_path: str | Path, device: str | torch.device) -> None:
        from ultralytics import RTDETR

        wrapper = RTDETR(str(checkpoint_path))
        self.names = wrapper.names
        self.model = wrapper.model.to(device).eval()
        self.device = torch.device(device)
        self.max_det = int(PROFILE.max_det)
        for param in self.model.parameters():
            param.requires_grad = False
        self._feat_buf: list[torch.Tensor] | None = None
        # Ultralytics RT-DETR has no HybridEncoder module: the YAML expands the
        # hybrid encoder into AIFI + FPN/PAN RepC3 blocks, then RTDETRDecoder.
        # Capture the decoder's multi-scale inputs (P3/P4/P5, typically 256-d).
        decoder = next(
            (m for m in self.model.modules() if m.__class__.__name__ == "RTDETRDecoder"),
            None,
        )
        if decoder is None:
            raise RuntimeError("RT-DETR RTDETRDecoder not found; cannot ROIAlign encoder features")
        decoder.register_forward_pre_hook(self._decoder_pre_hook)

    def _decoder_pre_hook(self, _module, inp) -> None:
        feats = inp[0] if inp else None
        if isinstance(feats, (list, tuple)):
            maps = [t.detach() for t in feats if torch.is_tensor(t) and t.ndim == 4]
        elif torch.is_tensor(feats) and feats.ndim == 4:
            maps = [feats.detach()]
        else:
            maps = []
        self._feat_buf = maps if maps else None

    @staticmethod
    def _scale_fill_rgb(image_rgb: np.ndarray) -> tuple[torch.Tensor, dict[str, Any]]:
        h, w = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0)
        return tensor.unsqueeze(0), {"orig_hw": (h, w)}

    @staticmethod
    def _xywh_norm_to_xyxy_orig(xywh: torch.Tensor, orig_hw: tuple[int, int]) -> torch.Tensor:
        boxes = xywh.reshape(-1, 4).float()
        orig_h, orig_w = orig_hw
        cx, cy, bw, bh = boxes.unbind(dim=1)
        return torch.stack([
            (cx - bw / 2) * orig_w,
            (cy - bh / 2) * orig_h,
            (cx + bw / 2) * orig_w,
            (cy + bh / 2) * orig_h,
        ], dim=1)

    def _project_boxes(self, boxes_xyxy: torch.Tensor, orig_hw: tuple[int, int]) -> torch.Tensor:
        orig_h, orig_w = orig_hw
        boxes = boxes_xyxy.clone().to(torch.float32)
        boxes[:, [0, 2]] *= float(IMGSZ) / max(orig_w, 1)
        boxes[:, [1, 3]] *= float(IMGSZ) / max(orig_h, 1)
        limit = float(IMGSZ - 1)
        boxes[:, 0] = boxes[:, 0].clamp_(0.0, limit)
        boxes[:, 1] = boxes[:, 1].clamp_(0.0, limit)
        boxes[:, 2] = boxes[:, 2].clamp_(min=1.0, max=float(IMGSZ))
        boxes[:, 3] = boxes[:, 3].clamp_(min=1.0, max=float(IMGSZ))
        boxes[:, 2] = torch.maximum(boxes[:, 2], boxes[:, 0] + 1.0)
        boxes[:, 3] = torch.maximum(boxes[:, 3], boxes[:, 1] + 1.0)
        return boxes

    def _parse_decoder(self, out: Any) -> tuple[torch.Tensor, torch.Tensor]:
        n_classes = len(self.names)
        if isinstance(out, (list, tuple)) and len(out) >= 2 and isinstance(out[1], (list, tuple)):
            extra = out[1]
            boxes = _last_decoder_layer(extra[0]).float()
            logits = _last_decoder_layer(extra[1]).float()
            return boxes, logits
        preds = out[0] if isinstance(out, (list, tuple)) else out
        if preds.ndim != 3 or preds.shape[-1] < 4 + n_classes:
            raise ValueError(f"unexpected RT-DETR output shape {tuple(getattr(preds, 'shape', ()))}")
        boxes = preds[..., :4].float()
        scores = preds[..., 4:4 + n_classes].float().clamp(1e-6, 1.0 - 1e-6)
        logits = torch.log(scores / (1.0 - scores))
        return boxes, logits

    def _decode_queries(
        self, boxes_xywh: torch.Tensor, logits: torch.Tensor, orig_hw: tuple[int, int]
    ) -> list[dict[str, Any]]:
        scores = logits.sigmoid()
        conf, labels = scores.max(dim=-1)
        keep = conf > CONF
        if not keep.any():
            return []
        conf, labels = conf[keep], labels[keep]
        boxes_kept = boxes_xywh[keep]
        order = torch.argsort(conf, descending=True)[: self.max_det]
        detections: list[dict[str, Any]] = []
        for idx in order.tolist():
            label = int(labels[idx].item())
            raw_label = str(self.names[label]).strip().lower()
            pred_class = PROFILE.label_aliases.get(raw_label, raw_label)
            if pred_class not in PROFILE.eval_classes:
                continue
            xyxy = self._xywh_norm_to_xyxy_orig(boxes_kept[idx], orig_hw)[0]
            detections.append({
                "pred_class": pred_class,
                "detector_class": pred_class,
                "detector_label": raw_label,
                "detector_confidence": float(conf[idx].item()),
                "bbox_xyxy": xyxy.detach().cpu().tolist(),
            })
        return detections

    def _pool_native(self, feats_per_image: list[torch.Tensor]) -> torch.Tensor:
        pooled: list[torch.Tensor] = []
        for feat in feats_per_image:
            pooled.append(feat.mean(dim=(1, 2)))
            pooled.append(feat.std(dim=(1, 2), unbiased=False))
        return torch.cat(pooled, dim=0)

    def _roi_align(
        self, feats_per_image: list[torch.Tensor], boxes_model: torch.Tensor
    ) -> torch.Tensor:
        total_channels = sum(f.shape[0] for f in feats_per_image)
        if boxes_model.numel() == 0:
            return torch.empty((0, total_channels, ROI_SIZE, ROI_SIZE), dtype=torch.float32)
        batch_idx = torch.zeros((boxes_model.shape[0], 1), dtype=torch.float32, device=self.device)
        rois = torch.cat([batch_idx, boxes_model.to(self.device)], dim=1).to(dtype=feats_per_image[0].dtype)
        pooled = []
        for feat in feats_per_image:
            stride = float(IMGSZ) / float(feat.shape[-2])
            pooled.append(
                roi_align(
                    feat.unsqueeze(0), rois, output_size=(ROI_SIZE, ROI_SIZE),
                    spatial_scale=1.0 / stride, aligned=True,
                )
            )
        return torch.cat(pooled, dim=1).cpu()

    @torch.inference_mode()
    def infer_batch_bgr(self, images_bgr: list[np.ndarray]) -> list[UnifiedImageResult]:
        if not images_bgr:
            return []
        tensors, infos = [], []
        for image_bgr in images_bgr:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            tensor, info = self._scale_fill_rgb(rgb)
            tensors.append(tensor)
            infos.append(info)
        batch_tensor = torch.cat(tensors, dim=0)
        use_amp = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
            out = self.model(batch_tensor.to(self.device))
        if not self._feat_buf:
            raise RuntimeError("RT-DETR encoder hook produced no feature maps")
        boxes, logits = self._parse_decoder(out)
        results: list[UnifiedImageResult] = []
        for batch_idx, info in enumerate(infos):
            feats = [level[batch_idx] for level in self._feat_buf]
            native = self._pool_native(feats).detach().cpu()
            detections = self._decode_queries(boxes[batch_idx], logits[batch_idx], info["orig_hw"])
            if detections:
                det_boxes = torch.tensor([d["bbox_xyxy"] for d in detections], dtype=torch.float32)
                roi = self._roi_align(feats, self._project_boxes(det_boxes, info["orig_hw"]))
            else:
                total = sum(f.shape[0] for f in feats)
                roi = torch.empty((0, total, ROI_SIZE, ROI_SIZE), dtype=torch.float32)
            results.append(UnifiedImageResult(detections, native, roi))
        return results


def _fx_split_predictions(predictions, proposals):
    """Pass through already-split per-image tuples from Detectron2 box/score heads."""
    sizes = [len(p) for p in proposals]
    if isinstance(predictions, (list, tuple)):
        if len(predictions) == len(proposals):
            return list(predictions)
        if len(predictions) == 1:
            predictions = predictions[0]
        else:
            raise ValueError(
                f"cannot split prediction collection of length {len(predictions)} "
                f"across {len(proposals)} images"
            )
    return list(predictions.split(sizes, dim=0))


class FRCNNUnifiedForward:
    """Detectron2 FX Faster R-CNN: detections + 256x7x7 ROI + FPN mean+std native."""

    def __init__(self, checkpoint_path: str | Path, device: str | torch.device) -> None:
        import os
        import sys

        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import get_cfg
        from detectron2.modeling import build_model
        from detectron2.structures import Boxes

        self.Boxes = Boxes
        self.device = torch.device(device)
        self.max_det = int(PROFILE.max_det)
        self.roi_channels = int(PROFILE.feature_channels)
        self.class_names = tuple(PROFILE.detector_class_names or PROFILE.label_classes)

        cfg_path = Path(FRCNN_CFG)
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"{cfg_path} (upload frcnn_fx/ next to run.py, or let mount_data.sh copy it)"
            )
        fx_dir = cfg_path.parent
        if str(fx_dir) not in sys.path:
            sys.path.insert(0, str(fx_dir))
        import utils.background as fx_bg  # noqa: E402
        import utils.fxrcnn as fx_mod  # noqa: F401,E402
        # Detectron2 already returns per-image tuples; FX must not split them again.
        fx_bg.split_predictions = _fx_split_predictions
        fx_mod.split_predictions = _fx_split_predictions

        old_cwd = os.getcwd()
        os.chdir(str(fx_dir))
        cfg = get_cfg()
        cfg.merge_from_file(str(cfg_path))
        cfg.MODEL.WEIGHTS = str(Path(checkpoint_path).resolve())
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.class_names)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONF
        cfg.TEST.DETECTIONS_PER_IMAGE = self.max_det
        cfg.MODEL.DEVICE = str(self.device)
        cfg.freeze()
        self.model = build_model(cfg)
        self.model.eval()
        DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)
        self.model.to(self.device)
        os.chdir(old_cwd)
        for param in self.model.parameters():
            param.requires_grad = False

    def _reshape_pooled(self, roi_features: torch.Tensor) -> torch.Tensor:
        if roi_features.dim() == 2:
            side = int(roi_features.shape[1] ** 0.5)
            roi_features = roi_features.view(roi_features.shape[0], -1, side, side)
        return roi_features

    def _parse_instances(self, instances) -> list[dict[str, Any]]:
        if instances is None or len(instances) == 0:
            return []
        keep = instances.scores >= CONF
        if not bool(keep.any()):
            return []
        boxes = instances.pred_boxes[keep].tensor.detach().cpu()
        scores = instances.scores[keep].detach().cpu()
        classes = instances.pred_classes[keep].detach().cpu()
        detections: list[dict[str, Any]] = []
        for i in range(boxes.shape[0]):
            idx = int(classes[i].item())
            if not 0 <= idx < len(self.class_names):
                continue
            raw_label = str(self.class_names[idx]).strip().lower()
            pred_class = PROFILE.label_aliases.get(raw_label, raw_label)
            if pred_class not in PROFILE.eval_classes:
                continue
            detections.append({
                "pred_class": pred_class,
                "detector_class": pred_class,
                "detector_label": raw_label,
                "detector_confidence": float(scores[i].item()),
                "bbox_xyxy": boxes[i].tolist(),
            })
        detections.sort(key=lambda d: d["detector_confidence"], reverse=True)
        return detections[: self.max_det]

    def _pool_native(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        names = list(self.model.roi_heads.box_in_features)
        pooled: list[torch.Tensor] = []
        for name in names:
            feat = features[name]
            pooled.append(feat.mean(dim=(2, 3)).squeeze(0))
            pooled.append(feat.std(dim=(2, 3), unbiased=False).squeeze(0))
        return torch.cat(pooled, dim=0).detach().cpu().float()

    def _roi_from_features(
        self, features: dict[str, torch.Tensor], boxes_model: torch.Tensor,
    ) -> torch.Tensor:
        if boxes_model.numel() == 0:
            return torch.empty((0, self.roi_channels, ROI_SIZE, ROI_SIZE), dtype=torch.float32)
        feature_list = [features[f] for f in self.model.roi_heads.box_in_features]
        roi = self.model.roi_heads.box_pooler(feature_list, [self.Boxes(boxes_model.to(self.device))])
        return self._reshape_pooled(roi).detach().cpu().float()

    def _infer_one_bgr(self, image_bgr: np.ndarray) -> UnifiedImageResult:
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        tensor = torch.as_tensor(image.transpose(2, 0, 1), device=self.device)
        batched = [{"image": tensor, "height": orig_h, "width": orig_w}]
        images = self.model.preprocess_image(batched)
        features = self.model.backbone(images.tensor)
        output = self.model.inference(batched)[0]
        model_h, model_w = images.image_sizes[0]
        detections = self._parse_instances(output["instances"])
        native = self._pool_native(features)
        if detections:
            boxes = torch.tensor([d["bbox_xyxy"] for d in detections], dtype=torch.float32)
            boxes_model = boxes.clone()
            boxes_model[:, [0, 2]] *= float(model_w) / max(orig_w, 1)
            boxes_model[:, [1, 3]] *= float(model_h) / max(orig_h, 1)
            roi = self._roi_from_features(features, boxes_model)
        else:
            roi = torch.empty((0, self.roi_channels, ROI_SIZE, ROI_SIZE), dtype=torch.float32)
        return UnifiedImageResult(detections, native, roi)

    @torch.inference_mode()
    def infer_batch_bgr(self, images_bgr: list[np.ndarray]) -> list[UnifiedImageResult]:
        # One image at a time: FPN pads the batch to max H/W, so a single large
        # Far-OOD photo would otherwise inflate every other image in the batch.
        return [self._infer_one_bgr(image_bgr) for image_bgr in images_bgr]


def build_engine(device: str | torch.device):
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"{CHECKPOINT} (detector weights required to build ROI/native caches; "
            f"reuse mounted data/{PROFILE.detector}/{PROFILE.name}/{{roi,native_knn}}/ "
            "from mount_data.sh, or pass --stage C/D when those caches already exist)"
        )
    if PROFILE.detector == "frcnn" and not FRCNN_CFG.is_file():
        raise FileNotFoundError(FRCNN_CFG)
    if PROFILE.detector == "frcnn":
        return FRCNNUnifiedForward(CHECKPOINT, device)
    if PROFILE.detector == "rtdetr":
        return RTDETRUnifiedForward(CHECKPOINT, device)
    return YOLOUnifiedForward(CHECKPOINT, device)


def quantize_fp8_per_roi(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-ROI float8 quantization of detector neck features."""
    x = features.float().cpu()
    scales = (x.abs().amax(dim=(1, 2, 3), keepdim=False) / FP8_E4M3_MAX).clamp(min=1e-8)
    scaled = (x / scales.view(-1, 1, 1, 1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return scaled.to(torch.float8_e4m3fn), scales.to(torch.float32)


# Native-embedding chunk I/O below is only used with --with-native.
def chunk_path(split_dir: Path, chunk_idx: int) -> Path:
    return split_dir / "chunks" / f"chunk_{chunk_idx:05d}.pt"


def write_native_chunk(
    native_dir: Path,
    chunk_idx: int,
    embeddings: list[torch.Tensor],
    image_rows: list[dict[str, Any]],
    detection_rows: list[dict[str, Any]],
    knn_split: str,
) -> None:
    chunk_emb = torch.stack(embeddings, dim=0).contiguous()
    spec = NATIVE_SPLIT_SPECS[knn_split]
    torch.save(
        {
            "split": knn_split,
            "data_source": spec["data_source"],
            "protocol": spec["protocol"],
            "checkpoint": str(CHECKPOINT.resolve()),
            "image_pooling": "mean_std",
            "dtype": str(NATIVE_DTYPE).replace("torch.", ""),
            "feature_definition": PROFILE.native_feature_definition,
            "feature_names": ["p3", "p4", "p5"],
            "embedding_shape": [len(embeddings), int(chunk_emb.shape[1])],
            "image_embeddings": chunk_emb,
            "image_rows": image_rows,
            "detection_rows": detection_rows,
            "unified_forward": True,
        },
        chunk_path(native_dir, chunk_idx),
    )


def write_native_indexes(native_dir: Path, image_rows: list[dict], detection_rows: list[dict]) -> None:
    for name, rows in [("images.jsonl", image_rows), ("detections.jsonl", detection_rows)]:
        with (native_dir / name).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    fields = [
        "split", "data_source", "protocol", "image_row", "file_name", "image_path",
        "num_detections", "embedding_chunk", "embedding_row_in_chunk",
    ]
    with (native_dir / "images.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in image_rows:
            writer.writerow({k: row.get(k) for k in fields})


def iter_image_batches(
    source: VocSplitImageSource, total: int, desc: str, batch_size: int,
) -> Iterator[list[ImageItem]]:
    batch: list[ImageItem] = []
    for item in tqdm(source.iter_items(), total=total, desc=desc):
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def native_split_dir(split: str) -> Path:
    return NATIVE_ROOT / SPLIT_OUTPUT_NAMES[split].replace(".pt", "")


def native_ready(native_dir: Path) -> bool:
    chunk_dir = native_dir / "chunks"
    return chunk_dir.is_dir() and any(chunk_dir.glob("chunk_*.pt"))


def _unpickle_torch_header(path: Path) -> dict[str, Any] | None:
    """Load the pickle dict from a torch.save zip, dropping tensor payloads."""

    class _Drop:
        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            pass

        def __call__(self, *args, **kwargs):
            return _Drop()

    class _Unpickler(pickle.Unpickler):
        def persistent_load(self, pid):
            return _Drop()

        def find_class(self, module, name):
            if module.startswith("torch") or module.startswith("numpy"):
                return _Drop
            return super().find_class(module, name)

    try:
        with zipfile.ZipFile(path) as zf:
            name = next((n for n in zf.namelist() if n.endswith("data.pkl")), None)
            if name is None:
                return None
            with zf.open(name) as fh:
                obj = _Unpickler(fh).load()
        return obj if isinstance(obj, dict) else None
    except (zipfile.BadZipFile, pickle.UnpicklingError, OSError, EOFError, StopIteration):
        return None


def roi_cache_counts(path: Path) -> dict[str, Any]:
    """Read num_images / num_detections without materializing ROI tensors."""
    info: dict[str, Any] = {"output": str(path.resolve()), "status": "existing"}
    if not path.is_file():
        info["status"] = "missing"
        return info
    header = _unpickle_torch_header(path)
    if header is not None:
        if header.get("num_detections") is not None:
            info["num_detections"] = int(header["num_detections"])
        if header.get("num_images") is not None:
            info["num_images"] = int(header["num_images"])
        if "num_detections" in info and "num_images" in info:
            return info
        meta = header.get("metadata")
        if "num_detections" not in info and isinstance(meta, list):
            info["num_detections"] = len(meta)
        if "num_images" in info and "num_detections" in info:
            return info
    payload = torch.load(path, map_location="cpu", weights_only=False)
    info["num_detections"] = int(payload["num_detections"])
    info["num_images"] = int(payload["num_images"])
    return info


def write_extraction_summary(summaries: list[dict[str, Any]], elapsed_sec: float) -> Path:
    """Write extraction_summary.json, keeping counts for splits not in this run."""
    path = ROI_DIR / "extraction_summary.json"
    by_split: dict[str, dict[str, Any]] = {}
    if path.is_file():
        try:
            previous = json.loads(path.read_text())
            for item in previous.get("splits", []):
                name = item.get("split")
                if name:
                    by_split[name] = item
        except (json.JSONDecodeError, OSError):
            pass
    for item in summaries:
        by_split[item["split"]] = item
    ordered: list[dict[str, Any]] = []
    for split in SPLITS:
        cache = ROI_DIR / SPLIT_OUTPUT_NAMES[split]
        if split in by_split:
            item = by_split[split]
            roi = dict(item.get("roi") or {})
            if roi.get("num_detections") is None or item.get("num_images") is None:
                stats = roi_cache_counts(cache)
                roi = {**stats, **roi}
                item = {
                    **item,
                    "num_images": item.get("num_images"),
                    "roi": roi,
                }
                if item["num_images"] is None:
                    item["num_images"] = stats.get("num_images")
            ordered.append(item)
        else:
            stats = roi_cache_counts(cache)
            ordered.append({
                "split": split,
                "num_images": stats.get("num_images"),
                "roi": {**stats, "status": stats.get("status", "existing")},
            })
    for split, item in by_split.items():
        if split not in SPLITS:
            ordered.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "detector": PROFILE.detector,
                "dataset": PROFILE.name,
                "checkpoint": str(CHECKPOINT.resolve()),
                "max_det": PROFILE.max_det,
                "splits": ordered,
                "elapsed_sec": elapsed_sec,
            },
            indent=2,
        )
        + "\n"
    )
    return path


def split_needs_extract(split: str, args: argparse.Namespace) -> tuple[bool, bool, bool]:
    """Return (need_roi, need_native, need_prior) for this split."""
    roi_path = ROI_DIR / SPLIT_OUTPUT_NAMES[split]
    skip_roi_write = bool(getattr(args, "_skip_roi_write", False))
    need_roi = (not skip_roi_write) and (args.force or not roi_path.is_file())
    need_native = args.with_native and not native_ready(native_split_dir(split))
    need_prior = bool(args.with_prior)
    return need_roi, need_native, need_prior


def save_roi_cache(
    roi_path: Path, split: str, roi_fp8_chunks: list, roi_scale_chunks: list,
    roi_rows: list[dict], num_images: int, elapsed: float,
) -> dict:
    """将 ROI 特征和元数据按原有格式保存，返回提取摘要中的 ROI 条目。"""
    if roi_fp8_chunks:
        features_fp8 = torch.cat(roi_fp8_chunks, dim=0)
        scales = torch.cat(roi_scale_chunks, dim=0)
    else:
        features_fp8 = torch.empty(
            (0, PROFILE.feature_channels, ROI_SIZE, ROI_SIZE), dtype=torch.float8_e4m3fn
        )
        scales = torch.empty((0,), dtype=torch.float32)
    payload = {
        "cache_type": PROFILE.roi_cache_type,
        "dataset": PROFILE.name,
        "detector": PROFILE.detector,
        "split": split,
        "ood_protocol": split if split in PROFILE.id_splits else "near_far_ood",
        "features_fp8": features_fp8,
        "scales": scales,
        "metadata": roi_rows,
        "num_detections": int(features_fp8.shape[0]),
        "num_images": int(num_images),
        "feature_channels": int(features_fp8.shape[1]) if features_fp8.numel() else PROFILE.feature_channels,
        "roi_size": ROI_SIZE,
        "fp8_format": "e4m3fn_per_roi",
        "feature_definition": PROFILE.roi_feature_definition,
        "detector_path": str(CHECKPOINT.resolve()),
        "inference": {
            "conf": CONF, "max_det": PROFILE.max_det, "iou": IOU,
            "imgsz": IMGSZ, "unified_forward": True,
        },
        "elapsed_sec": elapsed,
    }
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, roi_path)
    return {"output": str(roi_path.resolve()), "num_detections": int(payload["num_detections"])}


def extract_split(split: str, engine, args: argparse.Namespace) -> dict[str, Any]:
    """Detector forward over one split; write the ROI cache (plus optional extras)."""
    roi_path = ROI_DIR / SPLIT_OUTPUT_NAMES[split]
    knn_split = SPLIT_TO_KNN[split]
    prior_name, prior_source = SPLIT_TO_PRIOR[split]
    prior_path = PRIOR_DIR / prior_name
    native_dir = native_split_dir(split)
    need_roi, need_native, need_prior = split_needs_extract(split, args)
    if not need_roi and not need_native and not need_prior:
        stats = roi_cache_counts(roi_path)
        print(
            f"  {split}: caches exist, skipping "
            f"(images={stats.get('num_images')} dets={stats.get('num_detections')}; "
            f"use --force to redo ROI)",
            flush=True,
        )
        return {
            "split": split,
            "num_images": stats.get("num_images"),
            "roi": {**stats, "status": "skipped_existing"},
        }

    t0 = time.perf_counter()
    roi_rows: list[dict[str, Any]] = []
    roi_fp8_chunks: list[torch.Tensor] = []
    roi_scale_chunks: list[torch.Tensor] = []
    prior_rows = 0

    native_image_rows: list[dict[str, Any]] = []
    native_detection_rows: list[dict[str, Any]] = []
    chunk_embeddings: list[torch.Tensor] = []
    chunk_image_rows: list[dict[str, Any]] = []
    chunk_detection_rows: list[dict[str, Any]] = []
    chunk_idx = 0
    image_row = 0

    prior_file = None
    prior_writer = None
    if args.with_prior:
        PRIOR_DIR.mkdir(parents=True, exist_ok=True)
        prior_file = prior_path.open("w", newline="", encoding="utf-8")
        prior_writer = csv.DictWriter(prior_file, fieldnames=PRIOR_FIELDS)
        prior_writer.writeheader()
    if need_native:
        (native_dir / "chunks").mkdir(parents=True, exist_ok=True)
        print(f"  {split}: extracting image-level native embeddings"
              f"{'' if need_roi else ' (keeping existing ROI cache)'}", flush=True)

    with VocSplitImageSource(split, max_images=args.max_images) as source:
        num_images = source.count_images()
        extract_batch = int(getattr(args, "extract_batch_resolved", EXTRACT_BATCH))
        for batch_items in iter_image_batches(
            source, num_images, f"processing {split}", extract_batch,
        ):
            images_bgr = []
            for item in batch_items:
                bgr = source.load_bgr(item)
                if bgr is None:
                    raise FileNotFoundError(f"Failed to load {item.file_name}")
                images_bgr.append(bgr)

            for item, result, bgr in zip(batch_items, engine.infer_batch_bgr(images_bgr), images_bgr):
                height, width = bgr.shape[:2]
                image_path = source.image_uri(item)
                file_name = canonical_file_name(item.file_name)

                if prior_writer is not None:
                    for det in result.detections:
                        prior_writer.writerow({
                            "class": det["pred_class"],
                            "data_source": prior_source,
                            "image_path": image_path,
                            "file_name": item.file_name,
                            "bbox_xyxy": json.dumps([float(x) for x in det["bbox_xyxy"]]),
                            "detector_confidence": det["detector_confidence"],
                        })
                        prior_rows += 1

                if need_roi and result.detections:
                    for det in result.detections:
                        roi_rows.append({
                            "image_path": image_path,
                            "file_name": item.file_name,
                            "width": width,
                            "height": height,
                            "bbox_xyxy": det["bbox_xyxy"],
                            "detector_confidence": det["detector_confidence"],
                            "detector_label": det["detector_label"],
                            "pred_class": det["pred_class"],
                            "data_source": "detector_fp",
                        })
                    fp8, scales = quantize_fp8_per_roi(result.roi_features.float())
                    roi_fp8_chunks.append(fp8)
                    roi_scale_chunks.append(scales)

                if need_native:
                    expected_chunk = image_row // NATIVE_CHUNK_SIZE
                    if expected_chunk != chunk_idx:
                        write_native_chunk(
                            native_dir, chunk_idx, chunk_embeddings,
                            chunk_image_rows, chunk_detection_rows, knn_split,
                        )
                        chunk_embeddings, chunk_image_rows, chunk_detection_rows = [], [], []
                        chunk_idx = expected_chunk

                    spec = NATIVE_SPLIT_SPECS[knn_split]
                    out_name = chunk_path(native_dir, chunk_idx).name
                    row_in_chunk = image_row % NATIVE_CHUNK_SIZE
                    chunk_embeddings.append(
                        result.native_embedding.to(dtype=NATIVE_DTYPE).cpu().contiguous()
                    )
                    common = {
                        "split": knn_split,
                        "data_source": spec["data_source"],
                        "protocol": spec["protocol"],
                        "image_row": image_row,
                        "file_name": file_name,
                        "image_path": image_path,
                        "embedding_chunk": out_name,
                        "embedding_row_in_chunk": row_in_chunk,
                    }
                    image_record = {**common, "num_detections": len(result.detections)}
                    chunk_image_rows.append(image_record)
                    native_image_rows.append(image_record)
                    for det_idx, det in enumerate(result.detections):
                        detection_record = {**common, "detection_index_in_image": det_idx, **det}
                        chunk_detection_rows.append(detection_record)
                        native_detection_rows.append(detection_record)
                image_row += 1

    if need_native and chunk_embeddings:
        write_native_chunk(
            native_dir, chunk_idx, chunk_embeddings,
            chunk_image_rows, chunk_detection_rows, knn_split,
        )
    if need_native:
        write_native_indexes(native_dir, native_image_rows, native_detection_rows)
    if prior_file is not None:
        prior_file.close()

    elapsed = round(time.perf_counter() - t0, 3)
    summary: dict[str, Any] = {
        "split": split,
        "num_images": num_images,
        "elapsed_sec": elapsed,
    }
    if need_roi:
        summary["roi"] = save_roi_cache(
            roi_path, split, roi_fp8_chunks, roi_scale_chunks, roi_rows, num_images, elapsed,
        )
    else:
        summary["roi"] = {"output": str(roi_path.resolve()), "status": "kept_existing"}
    if args.with_prior:
        summary["prior"] = {"csv": str(prior_path.resolve()), "num_rows": prior_rows}
    if need_native or native_ready(native_dir):
        n_images = len(native_image_rows) if native_image_rows else None
        summary["native"] = {"dir": str(native_dir.resolve()), "num_images": n_images}
    return summary


# ===========================================================================
# Stage C-1: concept-head supervision and training
# ===========================================================================

def unpack_features(block: dict) -> torch.Tensor:
    """ROI features are stored as float8 + a per-row scale; restore float32."""
    if "features_fp8" in block:
        return block["features_fp8"].float() * block["scales"].view(-1, 1, 1, 1)
    return block["features"].float()


def training_data_classes(bundle: dict) -> list[str]:
    """Class names in semantic training_data (flat or per_class + metadata wrapper)."""
    if not isinstance(bundle, dict):
        return []
    per_class = bundle.get("per_class", bundle)
    if not isinstance(per_class, dict):
        return []
    names = [
        name for name, entry in per_class.items()
        if isinstance(entry, dict) and "id" in entry
    ]
    ordered = [c for c in PROFILE.label_classes if c in names]
    ordered += [c for c in names if c not in ordered]
    return ordered


def bundle_feature_channels(bundle: dict) -> int | None:
    """Channel dim of the first feature block in training_data.pt."""
    per_class = bundle.get("per_class", bundle)
    if not isinstance(per_class, dict):
        return None
    for entry in per_class.values():
        if not isinstance(entry, dict):
            continue
        for key in ("id", "prox", "unknown", "imagenet_o_fp"):
            block = entry.get(key)
            if not isinstance(block, dict):
                continue
            if "features_fp8" in block or "features" in block:
                return int(unpack_features(block).shape[1])
    return None


def roi_cache_channels(path: Path) -> int | None:
    """Channel dim stored on a stage-B ROI cache, without loading the fp8 tensor if possible."""
    if not path.is_file():
        return None
    header = _unpickle_torch_header(path)
    if isinstance(header, dict):
        ch = header.get("feature_channels")
        if ch is not None:
            return int(ch)
        fp8 = header.get("features_fp8")
        if torch.is_tensor(fp8) and fp8.ndim >= 2:
            return int(fp8.shape[1])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("feature_channels") is not None:
        return int(payload["feature_channels"])
    fp8 = payload.get("features_fp8")
    if torch.is_tensor(fp8) and fp8.ndim >= 2:
        return int(fp8.shape[1])
    return None


def first_roi_channels() -> int | None:
    for filename in ROI_SPLITS.values():
        ch = roi_cache_channels(ROI_DIR / filename)
        if ch is not None:
            return ch
    return None


def _channel_mismatch_exit(training_ch: int, roi_ch: int) -> None:
    raise SystemExit(
        f"channel mismatch: training_data.pt has {training_ch} channels, "
        f"ROI caches have {roi_ch} "
        f"({PROFILE.detector}/{PROFILE.name} extractor is {PROFILE.feature_channels}-d). "
        "The concept heads and ROI caches were built with different feature maps. "
        "Rebuild semantic training_data with the current run.py extractor, "
        "or restore ROI caches that match the training features. "
        "--retrain cannot fix this until those two match."
    )


def block_num_samples(block: dict | None) -> int:
    if not isinstance(block, dict):
        return 0
    feats = block.get("features_fp8")
    if feats is None:
        feats = block.get("features")
    return int(feats.shape[0]) if torch.is_tensor(feats) and feats.ndim >= 1 else 0


def slice_width(s: slice, n: int) -> int:
    start = 0 if s.start is None else int(s.start)
    stop = n if s.stop is None else int(s.stop)
    return max(stop - start, 0)


def _append_supervision(
    xs: list[torch.Tensor],
    ys: list[torch.Tensor],
    block: dict,
    target_slice: slice,
    n_concepts: int,
    *,
    has_masks: bool = True,
) -> None:
    x = unpack_features(block)
    if len(x) == 0:
        return
    masks = block["heatmaps"].float() if has_masks else torch.ones(len(x), 1, 7, 7)
    y = torch.zeros(len(x), n_concepts, *masks.shape[-2:])
    y[:, target_slice] = masks
    xs.append(x)
    ys.append(y)


def build_class_data(bundle: dict, class_name: str):
    """Assemble ID / prox / unknown supervision for one class.

    VOC classes usually have part-based `id` + `prox` + `unknown`. Some BDD
    classes ship without a `prox` block; those use whole-object ID vs unknown.
    """
    per_class = bundle.get("per_class", bundle)
    entry = per_class[class_name]
    id_block = entry["id"]
    prox_block = entry.get("prox")
    unknown_block = entry.get("unknown", entry.get("imagenet_o_fp"))
    if unknown_block is None:
        raise ValueError(f"{class_name}: missing unknown block")

    if "per_class" in bundle:
        part_based = bool(entry.get("is_part_based", True))
    else:
        part_based = prox_block is not None
    prox_concepts = list(prox_block.get("concept_order") or []) if prox_block else []
    if part_based and (prox_block is None or not prox_concepts):
        part_based = False

    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    if part_based:
        concept_order = (
            [f"id::{name}" for name in id_block["concept_order"]]
            + [f"prox::{name}" for name in prox_concepts]
            + ["unknown"]
        )
        n_id = len(id_block["concept_order"])
        n_prox = len(prox_concepts)
        id_slice = slice(0, n_id)
        prox_slice = slice(n_id, n_id + n_prox)
        unknown_slice = slice(n_id + n_prox, n_id + n_prox + 1)
        groups = [id_slice, prox_slice, unknown_slice]
        for block, target_slice, has_masks in [
            (id_block, id_slice, True),
            (prox_block, prox_slice, True),
            (unknown_block, unknown_slice, False),
        ]:
            _append_supervision(xs, ys, block, target_slice, len(concept_order), has_masks=has_masks)
    else:
        concept_order = [f"id::{class_name}", "unknown"]
        id_slice, unknown_slice = slice(0, 1), slice(1, 2)
        groups = [id_slice, unknown_slice]
        x_id = unpack_features(id_block)
        positive = [
            i for i, row in enumerate(id_block["metadata"])
            if row.get("is_positive", row.get("matched_gt_iou", 1.0) >= 0.5)
        ]
        positive_set = set(positive)
        negative = [i for i in range(len(x_id)) if i not in positive_set]
        if not positive:
            ch = int(x_id.shape[1]) if len(x_id) else int(unpack_features(unknown_block).shape[1])
            return (
                torch.empty(0, ch),
                torch.empty(0, len(concept_order), 7, 7),
                concept_order,
                groups,
            )
        id_selected = {
            "features": x_id[positive],
            "heatmaps": torch.ones((len(positive), 1, 7, 7)),
        }
        _append_supervision(xs, ys, id_selected, id_slice, len(concept_order))
        if negative:
            fp_selected = {
                "features": x_id[negative],
                "heatmaps": torch.ones((len(negative), 1, 7, 7)),
            }
            _append_supervision(xs, ys, fp_selected, unknown_slice, len(concept_order))
        _append_supervision(xs, ys, unknown_block, unknown_slice, len(concept_order), has_masks=False)

    if not xs:
        ref = id_block if block_num_samples(id_block) else unknown_block
        ch = int(unpack_features(ref).shape[1]) if block_num_samples(ref) else 0
        return (
            torch.empty(0, ch),
            torch.empty(0, len(concept_order), 7, 7),
            concept_order,
            groups,
        )
    return torch.cat(xs), torch.cat(ys), concept_order, groups


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(16, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(16, channels)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(F.gelu(self.norm1(self.conv1(x))))
        h = F.gelu(self.norm2(self.conv2(h)))
        return F.gelu(h + x)


class ConceptHead(nn.Module):
    """896x7x7 detector features -> one logit map per concept (7x7)."""

    def __init__(self, in_channels: int, n_concepts: int, hidden: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1), nn.GroupNorm(16, hidden), nn.GELU()
        )
        self.block1 = ResidualBlock(hidden)
        self.block2 = ResidualBlock(hidden)
        self.head = nn.Conv2d(hidden, n_concepts, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.block2(self.block1(self.stem(x))))


def pool_logits(logits: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """(N, C, 7, 7) -> (N, C): smooth max over the 49 spatial cells."""
    return torch.logsumexp(logits.flatten(2) / tau, dim=-1) * tau


def soft_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-concept overlap loss; robust to the tiny area of most part masks."""
    pred = torch.sigmoid(logits)
    intersection = (pred * target).sum((0, 2, 3))
    denominator = pred.sum((0, 2, 3)) + target.sum((0, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def group_ce(logits: torch.Tensor, target: torch.Tensor, groups: list[slice]) -> torch.Tensor:
    """Make the right group (id / prox / unknown) win, not just the right channel."""
    n_concepts = int(logits.shape[1])
    active = [g for g in groups if slice_width(g, n_concepts) > 0]
    if len(active) < 2:
        return logits.new_zeros(())
    scores = pool_logits(logits)
    group_scores = torch.stack([scores[:, g].amax(1) for g in active], dim=1)
    group_mass = torch.stack([target[:, g].flatten(1).sum(1) for g in active], dim=1)
    valid = group_mass.sum(1) > 0
    if not bool(valid.any()):
        return logits.new_zeros(())
    return F.cross_entropy(group_scores[valid], group_mass[valid].argmax(1))


def spatial_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Push absent concepts down, both pooled and on the object's pixels."""
    absent = target.flatten(2).sum(-1) == 0
    pooled = pool_logits(logits)
    pooled_loss = F.softplus(pooled[absent]).mean() if absent.any() else logits.new_zeros(())
    object_mask = target.amax(1, keepdim=True)
    mask = absent[:, :, None, None] * object_mask
    pixel_loss = (F.softplus(logits) * mask).sum() / mask.sum().clamp_min(1.0)
    return 0.5 * (pooled_loss + pixel_loss)


def total_loss(logits, target, groups) -> torch.Tensor:
    return soft_dice(logits, target) + 0.05 * group_ce(logits, target, groups) + 0.25 * spatial_loss(logits, target)


def train_head(class_name, x, y, groups, args) -> ConceptHead:
    """90/10 train/val split, class-balanced sampling, early stopping on val loss."""
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(x))
    cut = max(1, int(0.9 * len(order)))
    train_idx, val_idx = order[:cut], order[cut:]

    # The three groups have very different sizes (e.g. 1671 id vs 192 unknown),
    # so sample each group with probability ~ 1 / its size.
    group_of_row = torch.zeros(len(train_idx), dtype=torch.long)
    for group_id, group in enumerate(groups):
        group_of_row[y[train_idx, group].flatten(1).sum(1) > 0] = group_id
    counts = torch.bincount(group_of_row, minlength=len(groups)).float().clamp_min(1)
    sampler_gen = torch.Generator()
    sampler_gen.manual_seed(int(args.seed))
    sampler = WeightedRandomSampler(
        1.0 / counts[group_of_row], len(train_idx), replacement=True, generator=sampler_gen,
    )

    head_batch = min(tune_head_batch(int(x.shape[1]), int(y.shape[1]), args), len(train_idx))
    train_loader = DataLoader(
        TensorDataset(x[train_idx], y[train_idx]), head_batch, sampler=sampler
    )
    val_loader = DataLoader(TensorDataset(x[val_idx], y[val_idx]), head_batch)

    device = torch.device(args.device)
    head = ConceptHead(x.shape[1], y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=5e-4)

    best_loss, best_state, stale = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        for xb, yb in train_loader:
            batch_loss = total_loss(head(xb.to(device)), yb.to(device), groups)
            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            optimizer.step()

        head.eval()
        val_sum = 0.0
        with torch.inference_mode():
            for xb, yb in val_loader:
                val_sum += float(total_loss(head(xb.to(device)), yb.to(device), groups)) * len(xb)
        val_loss = val_sum / max(len(val_idx), 1)
        print(f"  [{class_name}] epoch {epoch:03d}/{args.epochs} val_loss={val_loss:.5f}", flush=True)

        if val_loss < best_loss - 1e-4:
            best_loss, stale = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"  [{class_name}] early stop at epoch {epoch}", flush=True)
                break

    head.load_state_dict(best_state)
    return head.eval()


# ===========================================================================
# Stage C-2: score the cached ROIs into SPK4 activations
# ===========================================================================

def canonical_pred_class(row: dict[str, Any]) -> str | None:
    """Map detector/cache labels onto PROFILE / head names (e.g. pedestrian -> person)."""
    raw = row.get("pred_class") or row.get("class") or row.get("detector_label")
    if raw is None:
        return None
    text = str(raw).strip()
    lowered = text.lower()
    aliased = PROFILE.label_aliases.get(lowered, PROFILE.label_aliases.get(text, text))
    return aliased


def score_roi_cache(
    cache_path: Path, split_name: str, heads: dict, device: str, batch_size: int = HEAD_BATCH,
) -> pd.DataFrame:
    """Run each class's head over the ROIs the detector predicted as that class."""
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    metadata, fp8, scales = cache["metadata"], cache["features_fp8"], cache["scales"]

    rows_by_class: dict[str, list[int]] = {}
    dropped: dict[str, int] = {}
    for i, row in enumerate(metadata):
        predicted = canonical_pred_class(row)
        if predicted in heads:
            rows_by_class.setdefault(predicted, []).append(i)
        else:
            dropped[str(predicted)] = dropped.get(str(predicted), 0) + 1
    if dropped:
        print(
            f"  {split_name}: dropped {sum(dropped.values())} ROIs with no head {dropped}",
            flush=True,
        )

    torch_device = torch.device(device)
    records = []
    for class_name, row_indices in rows_by_class.items():
        head, concept_order = heads[class_name]
        id_channels = [i for i, n in enumerate(concept_order) if n.startswith("id::")]
        prox_channels = [i for i, n in enumerate(concept_order) if n.startswith("prox::")]
        unknown_channel = concept_order.index("unknown")

        head_ch = int(head.stem[0].weight.shape[1])
        for start in range(0, len(row_indices), batch_size):
            chunk = row_indices[start:start + batch_size]
            # float8 tensors do not support fancy indexing on CPU, hence the stack.
            xb = torch.stack([fp8[i].float() * scales[i] for i in chunk]).to(torch_device)
            if xb.shape[1] != head_ch:
                raise SystemExit(
                    f"{cache_path}: ROI features are {xb.shape[1]}-d but {class_name}_head.pt "
                    f"expects {head_ch}-d. Delete mismatched *_head.pt and retrain only if "
                    f"training_data.pt also has {xb.shape[1]} channels."
                )
            with torch.inference_mode():
                # sigmoid(pooled logit) = "how strongly is this concept present?"
                activations = torch.sigmoid(pool_logits(head(xb))).cpu().numpy()

            for i, activation in zip(chunk, activations):
                row = metadata[i]
                x1, y1, x2, y2 = map(float, row["bbox_xyxy"])
                box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                image_area = max(float(row["width"]) * float(row["height"]), 1e-6)
                records.append({
                    "class": class_name,
                    "data_source": split_name,
                    "known_max": float(activation[id_channels].max()),
                    "unknown": float(activation[unknown_channel]),
                    "proxy_max": float(activation[prox_channels].max()) if prox_channels else 0.0,
                    "relative_area": box_area / image_area,
                    "image_path": row.get("image_path", ""),
                    "file_name": row["file_name"],
                    "detector_confidence": float(row["detector_confidence"]),
                    "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                })
    return pd.DataFrame(records)


# ===========================================================================
# Stage C-3: Isolation Forest per class, then FPR95
# ===========================================================================

def iou(box_a, box_b) -> float:
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return overlap / max(area_a + area_b - overlap, 1e-8)


def keep_true_positives(frame: pd.DataFrame, gt_index_path: Path) -> pd.DataFrame:
    """Keep only ID-train ROIs that really are the class the detector claimed.

    The ROI cache stores every detection, including the detector's mistakes. If
    we fitted the Isolation Forest on those too, "normal" would include garbage
    and OOD detection would degrade -- so we require IoU >= 0.5 with a
    ground-truth box of the same class.
    """
    ground_truth = json.loads(gt_index_path.read_text())
    columns = ["class", "file_name", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    keep = []
    for class_name, file_name, x1, y1, x2, y2 in frame[columns].to_numpy():
        raw = str(file_name)
        candidates = (
            ground_truth.get(Path(canonical_file_name(raw)).stem)
            or ground_truth.get(Path(raw).stem)
            or []
        )
        best = max(
            (iou([x1, y1, x2, y2], gt["bbox_xyxy"]) for gt in candidates if gt["class"] == class_name),
            default=0.0,
        )
        keep.append(best >= 0.5)
    return frame.loc[keep]


def fpr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Percent of OOD ROIs scoring at least as ID-like as the 5th-pct ID ROI.

    Isolation Forest `decision_function` is high = normal, so the threshold that
    keeps 95% of ID data is the 5th percentile of the ID scores.
    """
    if len(id_scores) == 0 or len(ood_scores) == 0:
        return float("nan")
    threshold = np.percentile(id_scores, 5)
    return 100.0 * float((ood_scores >= threshold).sum()) / len(ood_scores)


def auroc_pct(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """AUROC (%) with Isolation Forest scores (high = ID-like)."""
    if len(id_scores) == 0 or len(ood_scores) == 0:
        return float("nan")
    y = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    s = np.concatenate([id_scores, ood_scores])
    return 100.0 * float(roc_auc_score(y, s))


def drop_lowest_id_val(scores: np.ndarray, fraction: float = ID_VAL_OUTLIER_FRACTION) -> np.ndarray:
    """Drop the least ID-like ID-val scores (Isolation Forest: low = outlier)."""
    drop = min(int(np.floor(fraction * len(scores))), max(len(scores) - 5, 0))
    if drop <= 0:
        return scores
    return np.sort(scores)[drop:]


def spk_knn_mean_distances(train: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Mean Euclidean kNN distance in standardized SPK4 space (m-hood style)."""
    if len(train) == 0 or len(query) == 0:
        return np.zeros(len(query), dtype=np.float64)
    scaler = StandardScaler().fit(train)
    bank = scaler.transform(train)
    q = scaler.transform(query)
    use_k = max(1, min(int(k), len(bank)))
    nn = NearestNeighbors(n_neighbors=use_k, metric="euclidean", n_jobs=-1).fit(bank)
    return nn.kneighbors(q, return_distance=True)[0].mean(axis=1)


def drop_highest_id_val_knn(
    scores: np.ndarray, knn_dist: np.ndarray, fraction: float = ID_VAL_OUTLIER_FRACTION,
) -> np.ndarray:
    """Drop ID-val rows with the largest SPK kNN distance (farther from ID-train)."""
    if len(scores) != len(knn_dist):
        raise ValueError(f"scores/knn length mismatch: {len(scores)} vs {len(knn_dist)}")
    drop = min(int(np.floor(fraction * len(scores))), max(len(scores) - 5, 0))
    if drop <= 0:
        return scores
    keep = np.ones(len(scores), dtype=bool)
    keep[np.argsort(knn_dist)[-drop:]] = False
    return scores[keep]


def id_val_outlier_filter_label(mode: str) -> str:
    if mode == "iforest":
        return "drop each class's lowest 5% Isolation Forest scores"
    if mode == "spk_knn":
        return (
            "drop each class's highest 5% SPK kNN distances "
            f"(m-hood Euclidean kNN on standardized {SPK_KNN_OUTLIER_COLS}; "
            "same geometry as stage-D KNN on SPK4, not native image embeddings)"
        )
    return "none (all ID-val detections of the predicted class)"


def class_split_features(
    frame: pd.DataFrame,
    class_name: str,
    feature_cols: list[str],
    *,
    spk_outlier_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    part = frame.loc[frame["class"] == class_name]
    check_cols = list(dict.fromkeys(feature_cols + (spk_outlier_cols or [])))
    mask = np.isfinite(part[check_cols].to_numpy(dtype=np.float64)).all(axis=1)
    part = part.loc[mask]
    x = part[feature_cols].to_numpy(dtype=np.float64)
    spk = part[spk_outlier_cols].to_numpy(dtype=np.float64) if spk_outlier_cols else None
    return x, spk


def evaluate(
    activations: pd.DataFrame,
    class_names: list[str],
    id_train_tp: pd.DataFrame,
    feature_cols: list[str],
    outlier_mode: str = "none",
    seed: int = 42,
    spk_knn_k: int = SPK_OUTLIER_KNN_K,
) -> dict:
    """One Isolation Forest per class; report per-class and pooled FPR95 / AUROC."""
    if outlier_mode not in ID_VAL_OUTLIER_MODES:
        raise ValueError(f"unknown outlier_mode {outlier_mode!r}")
    per_class, pooled = {}, {"id_val": [], "near_ood": [], "far_ood": []}

    spk_outlier_cols = SPK_KNN_OUTLIER_COLS if outlier_mode == "spk_knn" else None
    id_val_frame = activations[activations["data_source"] == "id_val"]
    for class_name in class_names:
        train, train_spk = class_split_features(
            id_train_tp, class_name, feature_cols, spk_outlier_cols=spk_outlier_cols,
        )
        id_val, id_val_spk = class_split_features(
            id_val_frame, class_name, feature_cols, spk_outlier_cols=spk_outlier_cols,
        )
        near, _ = class_split_features(
            activations[activations["data_source"] == "near_ood"],
            class_name,
            feature_cols,
            spk_outlier_cols=spk_outlier_cols,
        )
        far, _ = class_split_features(
            activations[activations["data_source"] == "far_ood"],
            class_name,
            feature_cols,
            spk_outlier_cols=spk_outlier_cols,
        )
        if len(train) < 5 or len(id_val) < 5:
            per_class[class_name] = {
                "skipped": True,
                "n_train": len(train),
                "n_id_val": len(id_val),
                "n_near_ood": len(near),
                "n_far_ood": len(far),
            }
            print(
                f"  {class_name:8s} SKIPPED  n_train={len(train)} n_id_val={len(id_val)}  "
                f"OOD near={len(near)} far={len(far)} excluded from FPR",
                flush=True,
            )
            continue
        train_for_if = train
        train_spk_for_knn = train_spk
        if len(train) > 1500:  # cap keeps fitting fast; 1500 is plenty for 4-5 features
            keep = np.random.default_rng(seed).choice(len(train), 1500, replace=False)
            train_for_if = train[keep]
            if train_spk is not None:
                train_spk_for_knn = train_spk[keep]

        scaler = StandardScaler().fit(train_for_if)
        forest = IsolationForest(
            n_estimators=200, contamination=0.05,
            max_samples=min(512, len(train)), random_state=seed, n_jobs=-1,
        ).fit(scaler.transform(train_for_if))

        scores = {
            name: forest.decision_function(scaler.transform(data)) if len(data) else np.zeros(0)
            for name, data in [("id_val", id_val), ("near_ood", near), ("far_ood", far)]
        }
        # ID-val also contains detector mistakes. We have no ground truth for it,
        # so optionally drop the least ID-like 5% (IF scores or SPK4 kNN distance).
        n_id_val_raw = len(scores["id_val"])
        if outlier_mode == "iforest":
            scores["id_val"] = drop_lowest_id_val(scores["id_val"])
        elif outlier_mode == "spk_knn":
            if id_val_spk is None or train_spk_for_knn is None:
                raise ValueError(f"{class_name}: missing aligned SPK4 rows for spk_knn outlier removal")
            knn_dist = spk_knn_mean_distances(train_spk_for_knn, id_val_spk, spk_knn_k)
            scores["id_val"] = drop_highest_id_val_knn(scores["id_val"], knn_dist)
        for name in pooled:
            pooled[name].append(scores[name])

        near_fpr = fpr95(scores["id_val"], scores["near_ood"])
        far_fpr = fpr95(scores["id_val"], scores["far_ood"])
        near_auc = auroc_pct(scores["id_val"], scores["near_ood"])
        far_auc = auroc_pct(scores["id_val"], scores["far_ood"])
        per_class[class_name] = {
            "n_train": len(train),
            "n_id_val": n_id_val_raw, "n_id_val_kept": len(scores["id_val"]),
            "n_near_ood": len(near), "n_far_ood": len(far),
            "near_fpr95": near_fpr, "far_fpr95": far_fpr,
            "mean_fpr95": float(np.nanmean([near_fpr, far_fpr])),
            "near_auroc": near_auc, "far_auroc": far_auc,
        }
        print(
            f"  {class_name:8s} near={near_fpr:6.2f}  far={far_fpr:6.2f}  "
            f"n_near={len(near)} n_far={len(far)}",
            flush=True,
        )

    # Pooled: concatenate the per-class IF scores, then one global threshold.
    cat = {name: np.concatenate(parts) if parts else np.zeros(0) for name, parts in pooled.items()}
    near_fpr = fpr95(cat["id_val"], cat["near_ood"])
    far_fpr = fpr95(cat["id_val"], cat["far_ood"])
    finite = [x for x in (near_fpr, far_fpr) if np.isfinite(x)]
    mean_fpr = float(np.mean(finite)) if finite else float("nan")
    near_auc = auroc_pct(cat["id_val"], cat["near_ood"])
    far_auc = auroc_pct(cat["id_val"], cat["far_ood"])
    return {
        "method": "spk full" if KNN_COL in feature_cols else "spk local",
        "features": list(feature_cols),
        "ood_protocol": "near_far_ood",
        "id_train_filter": "same predicted class and IoU >= 0.5 with ground truth",
        "id_val_outlier_mode": outlier_mode,
        "id_val_filter": id_val_outlier_filter_label(outlier_mode),
        "per_class": per_class,
        "pooled": {
            "near_fpr95": near_fpr, "far_fpr95": far_fpr,
            "mean_fpr95": mean_fpr,
            "near_auroc": near_auc, "far_auroc": far_auc,
            "n_id_val": len(cat["id_val"]),
            "n_near_ood": len(cat["near_ood"]), "n_far_ood": len(cat["far_ood"]),
        },
    }


def load_native_embeddings(native_dir: Path) -> tuple[torch.Tensor, list[str]]:
    embs, names = [], []
    chunk_dir = native_dir / "chunks"
    if not chunk_dir.is_dir():
        raise FileNotFoundError(native_dir)
    for path in sorted(chunk_dir.glob("chunk_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        embs.append(payload["image_embeddings"].float())
        names.extend(canonical_file_name(row["file_name"]) for row in payload["image_rows"])
    if not embs:
        raise FileNotFoundError(f"No native chunks in {native_dir}")
    return torch.cat(embs, dim=0), names


@torch.inference_mode()
def score_classwise_native_knn(
    query_emb: torch.Tensor,
    query_names: list[str],
    gallery_emb: torch.Tensor,
    gallery_names: list[str],
    membership: dict[str, set[str]],
    class_names: list[str],
    k: int = NATIVE_K,
    leave_one_out: bool = False,
    chunk_size: int = 512,
) -> pd.DataFrame:
    """Mean cosine distance to the k nearest ID-train images that contain the class."""
    query = F.normalize(query_emb.float(), p=2, dim=1)
    gallery = F.normalize(gallery_emb.float(), p=2, dim=1)
    gallery_index = {name: i for i, name in enumerate(gallery_names)}
    records = {"file_name": query_names}

    for class_name in class_names:
        class_names_in_gallery = [n for n in membership.get(class_name, ()) if n in gallery_index]
        class_idx = [gallery_index[n] for n in class_names_in_gallery]
        distances = np.full(len(query_names), np.nan, dtype=np.float64)
        if len(class_idx) < 2:
            records[f"knn_dist__{class_name}"] = distances
            continue
        class_gallery = gallery[class_idx]
        class_gallery_names = class_names_in_gallery
        name_to_local = {n: i for i, n in enumerate(class_gallery_names)}
        use_k = min(k, len(class_idx) - (1 if leave_one_out else 0))
        use_k = max(use_k, 1)
        for start in range(0, len(query), chunk_size):
            end = min(start + chunk_size, len(query))
            sims = query[start:end] @ class_gallery.T
            if leave_one_out:
                for local_i, qname in enumerate(query_names[start:end]):
                    gallery_i = name_to_local.get(qname)
                    if gallery_i is not None:
                        sims[local_i, gallery_i] = -float("inf")
            topk = torch.topk(sims, k=use_k, dim=1).values
            distances[start:end] = (1.0 - topk.mean(dim=1)).cpu().numpy()
        records[f"knn_dist__{class_name}"] = distances
    return pd.DataFrame(records)


def attach_native_knn(activations: pd.DataFrame, knn_by_source: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = activations.copy()
    frame["_image_key"] = frame["file_name"].map(canonical_file_name)
    values = np.full(len(frame), np.nan, dtype=np.float64)
    for data_source, part in frame.groupby("data_source", sort=False):
        scores = knn_by_source.get(str(data_source))
        if scores is None:
            continue
        merged = part.merge(scores, on="_image_key", how="left", sort=False)
        part_vals = np.full(len(merged), np.nan, dtype=np.float64)
        for class_name in frame["class"].dropna().unique():
            mask = merged["class"].astype(str).eq(str(class_name))
            column = f"knn_dist__{class_name}"
            if mask.any() and column in merged.columns:
                part_vals[mask.to_numpy()] = merged.loc[mask, column].to_numpy(dtype=np.float64)
        values[frame["data_source"].astype(str).eq(str(data_source)).to_numpy()] = part_vals
    frame[KNN_COL] = values
    if frame[KNN_COL].notna().any():
        med = float(np.nanmedian(frame[KNN_COL].to_numpy(dtype=np.float64)))
        frame[KNN_COL] = frame[KNN_COL].fillna(med)
    frame.drop(columns=["_image_key"], inplace=True)
    return frame


def ensure_native_embeddings(args: argparse.Namespace) -> None:
    """Build image-level neck embeddings if missing; never overwrite ROI caches."""
    missing = [split for split in SPLITS if not native_ready(native_split_dir(split))]
    if not missing:
        print("  native embeddings present, reusing", flush=True)
        return
    print(
        f"  native embeddings missing for {missing}; one detector forward per "
        "image for P3/P4/P5 mean+std (ROI caches are not overwritten)",
        flush=True,
    )
    previous = args.with_native
    args.with_native = True
    args._skip_roi_write = True
    engine = build_engine(args.device)
    _ensure_extract_batch(engine, args)
    try:
        for split in missing:
            extract_split(split, engine, args)
    finally:
        args.with_native = previous
        args._skip_roi_write = False
        del engine


def build_native_knn_tables(
    id_train_tp: pd.DataFrame, class_names: list[str]
) -> dict[str, pd.DataFrame]:
    membership: dict[str, set[str]] = {}
    for class_name, part in id_train_tp.groupby("class"):
        membership[str(class_name)] = set(part["file_name"].map(canonical_file_name))

    train_emb, train_names = load_native_embeddings(NATIVE_ROOT / "id_train")
    print(
        f"  native kNN: {len(train_names)} ID-train images, k={NATIVE_K}, "
        f"class-conditional cosine distance",
        flush=True,
    )
    tables: dict[str, pd.DataFrame] = {}
    for data_source in ROI_SPLITS:
        native_dir = NATIVE_ROOT / data_source
        emb, names = load_native_embeddings(native_dir)
        table = score_classwise_native_knn(
            emb, names, train_emb, train_names, membership, class_names,
            k=NATIVE_K, leave_one_out=(data_source == "id_train"),
        )
        table = table.rename(columns={"file_name": "_image_key"})
        tables[data_source] = table
        print(f"  native kNN {data_source}: {len(table):,} images", flush=True)
    return tables


def load_saved_head(path: Path, device: str) -> tuple[ConceptHead, list[str]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    concept_order = payload["concept_order"]
    state = payload["state_dict"]
    in_channels = int(state["stem.0.weight"].shape[1])
    n_concepts = int(state["head.weight"].shape[0])
    head = ConceptHead(in_channels, n_concepts)
    head.load_state_dict(state)
    return head.to(device).eval(), concept_order


# ===========================================================================
# Glue
# ===========================================================================

PATIENCE = 10
SEED = 42
SEEDS = (42, 43, 44)
CANONICAL_OUTLIER = "without_outlier_removal"
LEGACY_HEAD_FILES = ("activations.csv", "results.json", "timing.json")
SPK_VARIANT_METHODS = ("MDS", "BAM", "KNN", "iForest")
SPK_VARIANT_LABELS = {"MDS": "SPK MDS", "BAM": "SPK BAM", "KNN": "SPK KNN", "iForest": "SPK IF"}
BASELINE_SPLIT_MAP = {"id_train": "train", "id_val": "id_val", "near_ood": "near_ood", "far_ood": "far_ood"}
DEFAULT_BAM_DENSITY_SWEEP = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0)
EXPERIMENTS_DRIVE_DEFAULT = Path("/content/drive/MyDrive/experiments")
ASSETS_SHARED_DRIVE_DEFAULT = Path("/content/drive/MyDrive/assets/shared")


def resolve_experiments_root(args: argparse.Namespace) -> Path | None:
    """Return the Drive experiments root, or None when backup is disabled."""
    if getattr(args, "no_backup", False):
        return None
    if args.experiments_dir is not None:
        root = args.experiments_dir.expanduser().resolve()
    elif EXPERIMENTS_DRIVE_DEFAULT.is_dir():
        root = EXPERIMENTS_DRIVE_DEFAULT
    else:
        return None
    if not root.is_dir():
        print(f"  backup skipped: {root} is not a directory", flush=True)
        return None
    return root


def resolve_assets_shared_root(args: argparse.Namespace) -> Path | None:
    """Return Drive assets/shared root, or None when Drive publish is disabled."""
    if getattr(args, "no_backup", False):
        return None
    if args.assets_dir is not None:
        root = args.assets_dir.expanduser().resolve()
    elif ASSETS_SHARED_DRIVE_DEFAULT.is_dir():
        root = ASSETS_SHARED_DRIVE_DEFAULT
    else:
        return None
    if not root.is_dir():
        print(f"  assets publish skipped: {root} is not a directory", flush=True)
        return None
    return root


def assets_gt_index_dest(assets_shared_root: Path) -> Path:
    """Same path mount_data.sh reads for eval-only SPK baselines."""
    return assets_shared_root / "datasets" / "id" / PROFILE.name / f"gt_{PROFILE.name}.json"


def experiment_subdir(experiments_root: Path) -> Path:
    """experiments/{detector}-{dataset}/ — same layout as mount_data.sh expects."""
    return experiments_root / f"{PROFILE.detector}-{PROFILE.name}"


def _human_bytes(n: int) -> str:
    size = float(max(n, 0))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)}{unit}"
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}TB"


def _sync_into(src: Path, dest: Path) -> tuple[int, int]:
    """Copy file or directory *contents* into dest (never dest/src.name nesting)."""
    if not src.exists():
        return 0, 0
    dest.mkdir(parents=True, exist_ok=True)
    copied, nbytes = 0, 0
    paths = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file())
    for path in paths:
        rel = path.name if src.is_file() else path.relative_to(src)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.is_file():
            src_stat, dst_stat = path.stat(), out.stat()
            if src_stat.st_size == dst_stat.st_size and src_stat.st_mtime <= dst_stat.st_mtime:
                continue
        shutil.copy2(path, out)
        copied += 1
        nbytes += path.stat().st_size
    return copied, nbytes


def _backup_stage(experiments_root: Path, label: str, src: Path, dest_name: str) -> None:
    exp = experiment_subdir(experiments_root)
    if not src.exists():
        print(f"  backup {label}: skip (missing {src})", flush=True)
        return
    if src.is_file():
        dest = exp / dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file():
            src_stat, dst_stat = src.stat(), dest.stat()
            if src_stat.st_size == dst_stat.st_size and src_stat.st_mtime <= dst_stat.st_mtime:
                print(f"  backup {label}: up to date at {dest}", flush=True)
                return
        shutil.copy2(src, dest)
        print(
            f"  backup {label}: 1 file(s), {_human_bytes(src.stat().st_size)} -> {dest}",
            flush=True,
        )
        return
    dest = exp / dest_name
    copied, nbytes = _sync_into(src, dest)
    if copied:
        print(
            f"  backup {label}: {copied} file(s), {_human_bytes(nbytes)} -> {dest}",
            flush=True,
        )
    else:
        print(f"  backup {label}: up to date at {dest}", flush=True)


def backup_gt_index(experiments_root: Path) -> None:
    if GT_INDEX.is_file():
        _backup_stage(experiments_root, "gt index", GT_INDEX, GT_INDEX.name)


def publish_gt_index_to_assets(assets_shared_root: Path) -> None:
    """Upload gt_{dataset}.json to Drive assets when missing (for mount_data.sh --eval-only)."""
    if not GT_INDEX.is_file():
        print(f"  assets gt index: skip (missing {GT_INDEX})", flush=True)
        return
    dest = assets_gt_index_dest(assets_shared_root)
    if dest.is_file():
        print(f"  assets gt index: already at {dest}", flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(GT_INDEX, dest)
    print(
        f"  assets gt index: uploaded {_human_bytes(dest.stat().st_size)} -> {dest}",
        flush=True,
    )


def backup_roi(experiments_root: Path) -> None:
    _backup_stage(experiments_root, "roi", ROI_DIR, "roi")


def backup_native_knn(experiments_root: Path) -> None:
    if not NATIVE_ROOT.is_dir() or not any(NATIVE_ROOT.rglob("*")):
        return
    _backup_stage(experiments_root, "native_knn", NATIVE_ROOT, "native_knn")


def backup_concept_head_ood(experiments_root: Path, out_root: Path, seeds: list[int]) -> None:
    """Single seed -> flat concept_head_ood/ (yolo-voc on Drive); multi-seed keeps seed_*/."""
    dest = experiment_subdir(experiments_root) / "concept_head_ood"
    dest.mkdir(parents=True, exist_ok=True)
    copied, nbytes = 0, 0
    if len(seeds) == 1:
        seed_src = seed_dir(out_root, seeds[0])
        if seed_src.is_dir():
            n, b = _sync_into(seed_src, dest)
            copied += n
            nbytes += b
        for name in ("timing.json", "pooled_mean_std.json"):
            src = out_root / name
            if not src.is_file():
                continue
            out = dest / name
            if out.is_file():
                src_stat, dst_stat = src.stat(), out.stat()
                if src_stat.st_size == dst_stat.st_size and src_stat.st_mtime <= dst_stat.st_mtime:
                    continue
            shutil.copy2(src, out)
            copied += 1
            nbytes += src.stat().st_size
    else:
        copied, nbytes = _sync_into(out_root, dest)
    if copied:
        print(
            f"  backup concept_head_ood: {copied} file(s), {_human_bytes(nbytes)} -> {dest}",
            flush=True,
        )
    elif out_root.is_dir() and any(out_root.rglob("*")):
        print(f"  backup concept_head_ood: up to date at {dest}", flush=True)


def default_bam_density() -> float:
    return 50.0 if PROFILE.name == "bdd" else 5.0


def variant_activation_jobs(args: argparse.Namespace) -> list[tuple[int, Path]]:
    jobs: list[tuple[int, Path]] = []
    flat = args.out_root / "activations.csv"
    for seed in args.seeds:
        path = seed_dir(args.out_root, seed) / "activations.csv"
        if path.is_file():
            jobs.append((int(seed), path))
        elif len(args.seeds) == 1 and flat.is_file():
            jobs.append((int(seed), flat))
    if not jobs:
        raise SystemExit(f"no activations.csv under {args.out_root} (run stage C first)")
    return jobs


def activations_to_baseline_data(
    activations: pd.DataFrame, feature_cols: list[str], *, train_tp_only: bool = True,
) -> dict[str, dict]:
    missing = [c for c in feature_cols if c not in activations.columns]
    if missing:
        raise SystemExit(f"missing SPK columns in activations.csv: {missing}")
    class_map = {name: i for i, name in enumerate(sorted(activations["class"].astype(str).unique()))}
    data: dict[str, dict] = {}
    for src, dst in BASELINE_SPLIT_MAP.items():
        part = activations[activations["data_source"] == src].copy()
        if dst == "train" and train_tp_only:
            if not GT_INDEX.is_file():
                raise SystemExit(f"GT index required for TP filter: {GT_INDEX}")
            part = keep_true_positives(part, GT_INDEX)
        part = part.loc[np.isfinite(part[feature_cols].to_numpy(dtype=np.float64)).all(axis=1)].copy()
        if not len(part):
            continue
        x = part[feature_cols].to_numpy(dtype=np.float64)
        labels = part["class"].astype(str).map(class_map).to_numpy(dtype=np.int64)
        ids = (
            part.index.astype(str) + ":" + part["class"].astype(str) + ":" + part["file_name"].astype(str)
        ).to_numpy(dtype=str)
        data[dst] = dict(logits=x, x=x, labels=labels, ids=ids, names=np.array(feature_cols, dtype=str))
    if "train" not in data or not len(data["train"]["x"]):
        raise SystemExit("empty train split after SPK variant filtering")
    if "id_val" not in data or not len(data["id_val"]["x"]):
        raise SystemExit("empty id_val split after SPK variant filtering")
    return data


def make_variant_args(args: argparse.Namespace, seed: int, bam_density: float | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        mds_labels="predicted",
        covariance=args.variant_covariance,
        knn_mode=args.variant_knn_mode,
        knn_k=args.variant_knn_k,
        bam_density=args.bam_density if bam_density is None else bam_density,
        bam_max_boxes=256,
        bam_cluster="minibatch",
        iforest_scope="classwise",
        trees=200,
        iforest_samples=512,
        seed=seed,
        jobs=args.variant_jobs,
        batch_size=512,
        scale_percentile=65.0,
        scale_degenerate="error",
        msp_denominator="all",
        msp_activation="softmax",
        temperature=1.0,
    )


def _score_variant_method(
    method: str, data: dict, train_x: np.ndarray, train_y: np.ndarray, baseline_args: SimpleNamespace,
) -> dict[str, dict]:
    eval_names = [k for k in data if k != "train"]
    ood_splits = [k for k in eval_names if k != "id_val"]
    model = ood_baseline.fit_model(method, train_x, train_y, baseline_args)
    scores = {split: ood_baseline.score(method, model, data[split], baseline_args)[0] for split in eval_names}
    return {split: ood_baseline.metrics(scores["id_val"], scores[split]) for split in ood_splits}


def run_stage_d(args: argparse.Namespace) -> None:
    """Stage D: MDS/BAM/KNN/iForest on saved activations -> spk_variants/."""
    SPK_VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = variant_activation_jobs(args)
    feature_cols = list(args.spk_features)
    if args.sweep_bam_density is not None:
        densities = list(args.sweep_bam_density)
        sweep_dir = SPK_VARIANTS_DIR / "bam_density_sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        print(f"\n=== BAM density sweep ({len(feature_cols)}D) -> {sweep_dir} ===", flush=True)
        for seed, path in jobs:
            data = activations_to_baseline_data(pd.read_csv(path), feature_cols)
            train_x, train_y = data["train"]["x"], data["train"]["labels"]
            for density in densities:
                try:
                    metrics = _score_variant_method(
                        "BAM", data, train_x, train_y, make_variant_args(args, seed, density),
                    )
                    near, far = metrics["near_ood"], metrics["far_ood"]
                    rows.append({
                        "seed": seed, "bam_density": float(density),
                        "near_fpr95_pct": near["fpr95_pct"], "far_fpr95_pct": far["fpr95_pct"],
                        "mean_fpr95_pct": float(np.mean([near["fpr95_pct"], far["fpr95_pct"]])),
                        "near_auroc_pct": near["auroc_pct"], "far_auroc_pct": far["auroc_pct"],
                    })
                    print(
                        f"  seed {seed} density={density:g}  near={near['fpr95_pct']:.2f}  "
                        f"far={far['fpr95_pct']:.2f}",
                        flush=True,
                    )
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    print(f"  seed {seed} density={density:g}  FAILED: {exc}", flush=True)
        if rows:
            with open(sweep_dir / "sweep.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return

    methods = list(args.spk_variant_methods)
    rows: list[dict] = []
    report: dict[str, Any] = {
        "feature_set": feature_cols, "train_tp_only": True, "outlier_removal": "none",
        "jobs": [{"seed": s, "activations": str(p)} for s, p in jobs], "methods": {}, "errors": {},
    }
    by_method = {SPK_VARIANT_LABELS[m]: {} for m in methods}
    print(f"\n=== SPK variants ({len(feature_cols)}D) -> {SPK_VARIANTS_DIR} ===", flush=True)
    print(f"jobs={len(jobs)}  bam_density={args.bam_density}  knn_k={args.variant_knn_k}", flush=True)
    for seed, path in jobs:
        print(f"\n--- seed {seed} {path.name} ---", flush=True)
        data = activations_to_baseline_data(pd.read_csv(path), feature_cols)
        train_x, train_y = data["train"]["x"], data["train"]["labels"]
        baseline_args = make_variant_args(args, seed)
        eval_names = [k for k in data if k != "train"]
        ood_splits = [k for k in eval_names if k != "id_val"]
        for method in methods:
            label = SPK_VARIANT_LABELS[method]
            try:
                model = ood_baseline.fit_model(method, train_x, train_y, baseline_args)
                scores = {
                    split: ood_baseline.score(method, model, data[split], baseline_args)[0]
                    for split in eval_names
                }
                seed_result = {"evaluations": {"full_id": {}}}
                for split in ood_splits:
                    metrics = ood_baseline.metrics(scores["id_val"], scores[split])
                    seed_result["evaluations"]["full_id"][split] = metrics
                    rows.append(dict(method=label, seed=seed, protocol="full_id", split=split, **metrics))
                    by_method[label].setdefault(split, {"fpr95_pct": [], "auroc_pct": []})
                    if metrics.get("fpr95_pct") is not None:
                        by_method[label][split]["fpr95_pct"].append(float(metrics["fpr95_pct"]))
                    if metrics.get("auroc_pct") is not None:
                        by_method[label][split]["auroc_pct"].append(float(metrics["auroc_pct"]))
                report["methods"].setdefault(label, {})[str(seed)] = seed_result
                print(f"  [{label}] ok", flush=True)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                report["errors"][f"{label}/seed{seed}"] = str(exc)
                print(f"  [{label}] FAILED: {exc}", flush=True)

    (SPK_VARIANTS_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if rows:
        with open(SPK_VARIANTS_DIR / "summary.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    pooled_rows = []
    print(f"\n=== {len(jobs)}-seed mean ± std ===", flush=True)
    header = f"{'method':10s} {'near FPR95':14s} {'far FPR95':14s} {'near AUROC':14s} {'far AUROC':14s}"
    print(header, flush=True)
    for method in methods:
        label = SPK_VARIANT_LABELS[method]
        near = by_method[label].get("near_ood", {"fpr95_pct": [], "auroc_pct": []})
        far = by_method[label].get("far_ood", {"fpr95_pct": [], "auroc_pct": []})
        print(
            f"{label:10s} {_fmt_mean_std(near['fpr95_pct']):14s} {_fmt_mean_std(far['fpr95_pct']):14s} "
            f"{_fmt_mean_std(near['auroc_pct']):14s} {_fmt_mean_std(far['auroc_pct']):14s}",
            flush=True,
        )
        pooled_rows.append({
            "method": label, "outlier_removal": "none",
            "near_fpr95": _fmt_mean_std(near["fpr95_pct"]).strip(),
            "far_fpr95": _fmt_mean_std(far["fpr95_pct"]).strip(),
            "near_auroc": _fmt_mean_std(near["auroc_pct"]).strip(),
            "far_auroc": _fmt_mean_std(far["auroc_pct"]).strip(),
        })
    (SPK_VARIANTS_DIR / "pooled_mean_std.json").write_text(json.dumps(pooled_rows, indent=2) + "\n")


def backup_spk_variants(experiments_root: Path) -> None:
    if SPK_VARIANTS_DIR.is_dir() and any(SPK_VARIANTS_DIR.rglob("*")):
        _backup_stage(experiments_root, "spk_variants", SPK_VARIANTS_DIR, "spk_variants")


def seed_dir(root: Path, seed: int) -> Path:
    return root / f"seed_{seed}"


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def migrate_legacy_heads(out_root: Path, seed: int) -> None:
    """Move flat concept_head_ood/*_head.pt into seed_{seed}/ (one-time layout change)."""
    dest = seed_dir(out_root, seed)
    if dest.is_dir() and any(dest.glob("*_head.pt")):
        return
    heads = sorted(p for p in out_root.glob("*_head.pt") if p.parent == out_root)
    if not heads:
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  migrating {len(heads)} legacy heads -> {dest}", flush=True)
    for path in heads:
        shutil.move(str(path), str(dest / path.name))
    for name in LEGACY_HEAD_FILES:
        src = out_root / name
        if src.is_file():
            shutil.move(str(src), str(dest / name))


def _fmt_mean_std(values: list[float]) -> str:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return "nan"
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return f"{float(np.mean(arr)):.2f} ± {std:.2f}"


def write_seed_pool(out_root: Path, reports: dict[int, dict]) -> dict:
    """Mean±std over head seeds. Headline protocol: no ID-val outlier removal."""
    methods = ["spk local", "spk full"]
    keys = ["near_fpr95", "far_fpr95", "mean_fpr95", "near_auroc", "far_auroc"]
    table = []
    print(
        f"\n=== {len(reports)}-seed mean ± std (concept heads; outlier removal: none) ===",
        flush=True,
    )
    print(
        f"{'method':12s} {'near FPR95':14s} {'far FPR95':14s} "
        f"{'near AUROC':14s} {'far AUROC':14s}",
        flush=True,
    )
    for method_name in methods:
        collected = {k: [] for k in keys}
        for report in reports.values():
            pooled = report["methods"][method_name][CANONICAL_OUTLIER]["pooled"]
            for k in keys:
                collected[k].append(pooled.get(k, float("nan")))
        row = {
            "method": method_name,
            "outlier_removal": "none",
            "seeds": sorted(reports),
            **{k: _fmt_mean_std(collected[k]) for k in keys},
        }
        table.append(row)
        print(
            f"{method_name:12s} {row['near_fpr95']:14s} {row['far_fpr95']:14s} "
            f"{row['near_auroc']:14s} {row['far_auroc']:14s}",
            flush=True,
        )
    payload = {"outlier_removal": "none", "seeds": sorted(reports), "methods": table}
    path = out_root / "pooled_mean_std.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}", flush=True)
    return payload


def load_or_train_heads(args: argparse.Namespace) -> tuple[dict, list[str]]:
    """逐类复用已保存的概念头，只训练缺失或要求重训的类别。"""
    training_data = torch.load(
        resolve_training_data(TRAINING_DATA), map_location="cpu", weights_only=False
    )
    available = training_data_classes(training_data)
    class_names = args.classes or available
    missing = sorted(set(class_names) - set(available))
    if missing:
        raise SystemExit(f"classes not in training_data.pt: {missing}")
    if not class_names:
        raise SystemExit("training_data.pt has no class entries with an id block")

    print(f"  {len(class_names)} class(es) on {args.device}")
    roi_ch = first_roi_channels()
    td_ch = bundle_feature_channels(training_data)
    if roi_ch is not None:
        print(f"  ROI caches: {roi_ch} channels (profile {PROFILE.feature_channels})", flush=True)
    if td_ch is not None:
        print(f"  training_data.pt: {td_ch} channels", flush=True)
    heads = {}
    n_reused = 0
    for class_name in class_names:
        head_path = args.out / f"{class_name}_head.pt"
        if head_path.is_file() and not args.retrain:
            head, concept_order = load_saved_head(head_path, args.device)
            head_ch = int(head.stem[0].weight.shape[1])
            if roi_ch is not None and head_ch != roi_ch:
                print(
                    f"  [{class_name}] skip reuse {head_path.name}: "
                    f"head {head_ch}-d vs ROI {roi_ch}-d",
                    flush=True,
                )
            else:
                heads[class_name] = (head, concept_order)
                n_reused += 1
                print(f"  [{class_name}] reused {head_path.name} ({len(concept_order)} concepts)", flush=True)
                continue
        if td_ch is not None and roi_ch is not None and td_ch != roi_ch:
            _channel_mismatch_exit(td_ch, roi_ch)
        x, y, concept_order, groups = build_class_data(training_data, class_name)
        if len(x) == 0:
            print(f"  [{class_name}] SKIPPED (no usable ID training ROIs)", flush=True)
            continue
        if roi_ch is not None and int(x.shape[1]) != roi_ch:
            _channel_mismatch_exit(int(x.shape[1]), roi_ch)
        print(f"  [{class_name}] {len(x):,} ROIs, {len(concept_order)} concepts", flush=True)
        head = train_head(class_name, x, y, groups, args)
        heads[class_name] = (head, concept_order)
        torch.save({"state_dict": head.state_dict(), "concept_order": concept_order,
                    "class_name": class_name, "seed": int(args.seed)}, head_path)
        del x, y
    class_names = [name for name in class_names if name in heads]
    if not class_names:
        raise SystemExit("no concept heads available (all classes empty or skipped)")
    print(
        f"  heads: {len(class_names)} class(es); reused {n_reused}, "
        f"trained {len(class_names) - n_reused}",
        flush=True,
    )
    del training_data

    return heads, class_names


def score_all_roi_caches(heads: dict, args: argparse.Namespace) -> pd.DataFrame:
    """依次评分四个数据划分，再合并为一张表。"""
    print("  scoring cached ROIs")
    frames = []
    score_batch = max(
        (tune_head_batch(head.stem[0].weight.shape[1], head.head.weight.shape[0], args)
         for head, _ in heads.values()),
        default=HEAD_BATCH,
    )
    for split_name, filename in ROI_SPLITS.items():
        frame = score_roi_cache(
            ROI_DIR / filename, split_name, heads, args.device, batch_size=score_batch,
        )
        print(f"  {split_name}: {len(frame):,} ROIs", flush=True)
        if not frame.empty:
            print(f"    {frame['class'].value_counts().to_dict()}", flush=True)
        frames.append(frame)
    activations = pd.concat(frames, ignore_index=True)
    print(
        "  scored class x split:\n"
        f"{activations.groupby(['data_source', 'class']).size().unstack(fill_value=0)}",
        flush=True,
    )
    return activations


def load_or_score_activations(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    """已有评分时直接读取；否则准备概念头并对 ROI 缓存评分。"""
    activations_path = args.out / "activations.csv"
    if args.retrain:
        args.rescore = True
    reuse_activations = activations_path.is_file() and not args.rescore

    if reuse_activations:
        activations = pd.read_csv(activations_path)
        present = set(activations["class"].astype(str))
        class_names = args.classes or [c for c in PROFILE.label_classes if c in present]
        print(f"  reused {activations_path} ({len(activations):,} rows, {len(class_names)} classes)", flush=True)
    else:
        heads, class_names = load_or_train_heads(args)
        activations = score_all_roi_caches(heads, args)
    return activations, class_names


def prepare_knn_activations(activations, class_names, args):
    """筛出训练集真阳性，并把图像级 kNN 距离附到每个检测框。"""
    id_train = activations[activations["data_source"] == "id_train"]
    id_train_tp = keep_true_positives(id_train, GT_INDEX)
    print(f"  ID-train true positives: {len(id_train_tp):,} of {len(id_train):,}", flush=True)
    if len(id_train) and len(id_train_tp) == 0:
        raise SystemExit(
            f"ID-train TP filter kept 0/{len(id_train)} ROIs using {GT_INDEX}. "
            "The GT index is almost certainly the wrong dataset (shared data/id/gt.json "
            "leftover). Delete it or let stage A rebuild data/id/gt_{dataset}.json."
        )

    if KNN_COL in activations.columns and activations[KNN_COL].notna().any():
        print(f"  reused {KNN_COL} from activations.csv (skip native embedding build)", flush=True)
        return activations, id_train_tp

    print("  ensuring image-level native embeddings", flush=True)
    ensure_native_embeddings(args)

    if getattr(args, "_knn_tables", None) is None:
        args._knn_tables = build_native_knn_tables(id_train_tp, class_names)
    knn_tables = args._knn_tables
    tp_index = id_train_tp.index
    activations = attach_native_knn(activations, knn_tables)
    id_train_tp = activations.loc[tp_index]
    n_knn = int(activations[KNN_COL].notna().sum())
    print(f"  attached {KNN_COL} to {n_knn:,}/{len(activations):,} activation rows", flush=True)
    return activations, id_train_tp


def evaluate_methods(activations, class_names, id_train_tp, args) -> dict:
    """Evaluate spk local/full under each requested ID-val outlier-removal mode."""
    methods = [
        ("spk local", SPK4_COLS),
        ("spk full", SPK4_COLS + [KNN_COL]),
    ]
    outlier_modes = list(dict.fromkeys(args.id_val_outlier_modes))
    unknown = [m for m in outlier_modes if m not in ID_VAL_OUTLIER_MODES]
    if unknown:
        raise SystemExit(f"unknown --id-val-outlier-modes {unknown} (choose from {list(ID_VAL_OUTLIER_MODES)})")
    missing_spk = [c for c in SPK_KNN_OUTLIER_COLS if c not in activations.columns]
    if "spk_knn" in outlier_modes and missing_spk:
        raise SystemExit(
            f"--id-val-outlier-modes spk_knn requires SPK columns {missing_spk} in activations.csv"
        )
    report: dict[str, Any] = {
        "dataset": PROFILE.name,
        "detector": PROFILE.detector,
        "seed": int(args.seed),
        "ood_protocol": "near_far_ood",
        "id_train_filter": "same predicted class and IoU >= 0.5 with ground truth",
        "id_val_outlier_modes": outlier_modes,
        "spk_knn_outlier": {
            "feature_cols": list(SPK_KNN_OUTLIER_COLS),
            "k": int(args.id_val_outlier_knn_k),
            "distance": "mean Euclidean distance to k nearest ID-train TP rows (StandardScaler + m-hood)",
        },
        "native_knn": {
            "column": KNN_COL,
            "k": NATIVE_K,
            "definition": PROFILE.native_feature_definition,
            "membership": "ID-train true-positive images",
        },
        "methods": {},
    }
    print("  FPR95 (lower is better)")
    for method_name, feature_cols in methods:
        report["methods"][method_name] = {}
        for mode in outlier_modes:
            setting_name = ID_VAL_OUTLIER_SETTING[mode]
            print(
                f"\n=== {method_name}  features={feature_cols}  "
                f"id_val outlier removal: {mode} ===",
                flush=True,
            )
            result = evaluate(
                activations, class_names, id_train_tp, feature_cols,
                outlier_mode=mode, seed=int(args.seed),
                spk_knn_k=int(args.id_val_outlier_knn_k),
            )
            pooled = result["pooled"]
            print(
                f"pooled  near={pooled['near_fpr95']:.2f}  far={pooled['far_fpr95']:.2f}  "
                f"mean={pooled['mean_fpr95']:.2f}",
                flush=True,
            )
            report["methods"][method_name][setting_name] = result

    print("\n=== pooled summary ===")
    print(f"{'method':12s} {'outlier_mode':20s} {'near':8s} {'far':8s} {'mean':8s}")
    for method_name, _ in methods:
        for mode in outlier_modes:
            setting_name = ID_VAL_OUTLIER_SETTING[mode]
            pooled = report["methods"][method_name][setting_name]["pooled"]
            print(
                f"{method_name:12s} {mode:20s} "
                f"{pooled['near_fpr95']:8.2f} {pooled['far_fpr95']:8.2f} {pooled['mean_fpr95']:8.2f}"
            )

    return report


def train_and_eval(args: argparse.Namespace) -> dict:
    """阶段 C：准备评分 → 添加 kNN 特征 → 评估 → 保存结果。"""
    activations, class_names = load_or_score_activations(args)
    activations, id_train_tp = prepare_knn_activations(activations, class_names, args)
    activations.to_csv(args.out / "activations.csv", index=False)
    report = evaluate_methods(activations, class_names, id_train_tp, args)
    (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}/results.json and {args.out}/activations.csv")
    return report


def _planned_classes(args: argparse.Namespace) -> list[str]:
    if args.classes:
        return list(args.classes)
    try:
        bundle = torch.load(
            resolve_training_data(TRAINING_DATA), map_location="cpu", weights_only=False
        )
    except (FileNotFoundError, IsADirectoryError, OSError):
        return list(PROFILE.label_classes)
    ordered = training_data_classes(bundle) if isinstance(bundle, dict) else []
    return ordered or list(PROFILE.label_classes)


def print_skip_plan(args: argparse.Namespace) -> None:
    """Say which stages will run vs reuse files already under --root."""
    classes = _planned_classes(args)
    out_root = getattr(args, "out_root", args.out)
    seeds = list(getattr(args, "seeds", [getattr(args, "seed", SEED)]))
    print("=== skip plan (--force / --retrain / --rescore to override) ===", flush=True)
    print(
        f"  A  GT index           {'skip' if gt_index_is_current(GT_INDEX, DATASET_DIR) else f'run   ({GT_INDEX.name})'}",
        flush=True,
    )
    b_run = False
    b_bits: list[str] = []
    for split in args.splits:
        need_roi, need_native, need_prior = split_needs_extract(split, args)
        path = ROI_DIR / SPLIT_OUTPUT_NAMES[split]
        if need_roi or need_native or need_prior:
            b_run = True
            why = []
            if need_roi:
                why.append("roi")
            if need_native:
                why.append("native")
            if need_prior:
                why.append("prior")
            b_bits.append(f"{split}:{'+'.join(why)}")
            continue
        stats = roi_cache_counts(path) if path.is_file() else {"status": "missing"}
        b_bits.append(
            f"{split}:skip({stats.get('num_images')} img, {stats.get('num_detections')} det)"
        )
    status = "run " if b_run else "skip"
    print(f"  B  ROI extraction     {status}  ({'; '.join(b_bits)})", flush=True)
    native_miss = [split for split in SPLITS if not native_ready(native_split_dir(split))]
    if native_miss:
        print(f"  C  native embeddings  run   (missing {native_miss})", flush=True)
    else:
        print("  C  native embeddings  skip  (chunks on disk)", flush=True)
    print(f"  C  head seeds         {seeds}", flush=True)
    for seed in seeds:
        seed_out = seed_dir(out_root, seed)
        heads_ok = [c for c in classes if (seed_out / f"{c}_head.pt").is_file()]
        heads_miss = [c for c in classes if c not in heads_ok]
        activations_path = seed_out / "activations.csv"
        if args.retrain:
            head_plan = f"retrain {len(classes)} classes"
        elif heads_miss:
            head_plan = f"train {heads_miss}; reuse {heads_ok or 'none'}"
        else:
            head_plan = f"skip ({len(heads_ok)}/{len(classes)} *_head.pt)"
        if args.rescore or args.retrain or not activations_path.is_file():
            score_plan = "run"
        else:
            score_plan = f"skip ({activations_path.name} exists)"
        print(f"    seed {seed}: heads {head_plan}; score {score_plan}", flush=True)
    modes = getattr(args, "id_val_outlier_modes", ["none", "spk_knn"])
    print(
        f"  C  FPR95 eval         run   (per seed; headline outlier=none; also {list(modes)})",
        flush=True,
    )


def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s ({seconds:.1f}s)"
    if minutes:
        return f"{minutes}m {secs:02d}s ({seconds:.1f}s)"
    return f"{seconds:.1f}s"


def parse_args() -> argparse.Namespace:
    """读取命令行参数，并按所选数据集补齐默认值。"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--detector", choices=("yolo", "frcnn", "rtdetr"), default="yolo",
                        help="yolo, frcnn, or rtdetr (checkpoint, ROI channels, native pooling)")
    parser.add_argument("--dataset", choices=("voc", "bdd"), default="voc",
                        help="voc or bdd (tar prefixes, classes)")
    parser.add_argument("--root", type=Path, default=None,
                        help="Bundle root holding model/ and data/ (default: the script's directory)")
    parser.add_argument("--splits", nargs="+", default=None,
                        help="Which splits to extract in stage B (default: all for --dataset)")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Subset of classes (default: every class in training_data.pt)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--max-images", type=int, default=0, help="0 = all images (smoke runs: 200)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Stage C output dir (default: <root>/data/concept_head_ood)")
    parser.add_argument("--force", action="store_true", help="Re-extract ROI caches that already exist")
    parser.add_argument(
        "--stage",
        nargs="+",
        choices=("A", "B", "C", "D"),
        default=None,
        help="Run only these stages (default: A B C D). Example: --stage D runs SPK variant baselines only",
    )
    parser.add_argument("--extract-only", action="store_true",
                        help="Stop after stage B (ROI extraction); same as --stage A B")
    parser.add_argument(
        "--id-val-outlier-modes",
        nargs="+",
        choices=list(ID_VAL_OUTLIER_MODES),
        default=["none", "spk_knn"],
        help="Stage C ID-val calibration filters (default: none spk_knn). "
             "spk_knn drops each class's highest 5%% kNN distances fit on standardized SPK4; "
             "iforest drops lowest 5%% Isolation Forest scores. Headline tables use none.",
    )
    parser.add_argument(
        "--id-val-outlier-knn-k",
        type=int,
        default=SPK_OUTLIER_KNN_K,
        help="k for spk_knn outlier removal (default: 5; m-hood mean distance to k neighbors)",
    )
    parser.add_argument("--skip-spk-variants", action="store_true",
                        help="Skip stage D (MDS/BAM/KNN/iForest on SPK features)")
    parser.add_argument("--bam-density", type=float, default=None,
                        help="Stage D BAM density (default: 5 voc, 50 bdd)")
    parser.add_argument("--sweep-bam-density", nargs="*", type=float, default=None, metavar="D",
                        help="Stage D: sweep BAM densities only (default grid: 1 2 3 5 10 20 50)")
    parser.add_argument("--spk-features", nargs="+",
                        choices=["known_max", "unknown", "proxy_max", "relative_area", "native_knn"],
                        default=None, help="Stage D feature columns (default: 5D spk full)")
    parser.add_argument("--spk-variant-methods", nargs="+", choices=list(SPK_VARIANT_METHODS),
                        default=list(SPK_VARIANT_METHODS))
    parser.add_argument("--variant-covariance", choices=["empirical", "ledoit-wolf"], default="empirical")
    parser.add_argument("--variant-knn-mode", choices=["mhood", "sun"], default="mhood")
    parser.add_argument("--variant-knn-k", type=int, default=5)
    parser.add_argument("--variant-jobs", type=int, default=4,
                        help="CPU threads for stage D sklearn baselines")
    parser.add_argument("--with-native", action="store_true",
                        help="Also write native-kNN image embeddings (needed for `spk global` / `spk full`)")
    parser.add_argument("--with-prior", action="store_true",
                        help="Also write detection-prior CSVs")
    parser.add_argument("--retrain", action="store_true",
                        help="Redo concept heads even if {class}_head.pt already exists")
    parser.add_argument("--rescore", action="store_true",
                        help="Redo ROI scoring even if activations.csv already exists")
    parser.add_argument("--vram-frac", type=float, default=VRAM_FRACTION,
                        help="Target fraction of GPU memory for auto batch size (default: 0.70)")
    parser.add_argument("--extract-batch", type=int, default=0,
                        help="Stage B image batch; 0 = auto from --vram-frac")
    parser.add_argument("--head-batch", type=int, default=0,
                        help="Stage C head train/score batch; 0 = auto from --vram-frac")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Single concept-head seed (default: 42)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Run several concept-head seeds (e.g. --seeds 42 43 44)")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=None,
        help="Drive experiments root for stage backups "
             "(default: /content/drive/MyDrive/experiments when mounted)",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=None,
        help="Drive assets/shared root for gt_{dataset}.json publish "
             "(default: /content/drive/MyDrive/assets/shared when mounted)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not copy stage outputs to Drive (experiments/ or assets/shared/)",
    )
    args = parser.parse_args()
    set_profile(args.detector, args.dataset)
    if args.root is not None:
        set_root(args.root)
    else:
        refresh_paths()
    if args.extract_only and args.stage is not None:
        raise SystemExit("pass --stage or --extract-only, not both")
    if args.extract_only:
        args.stages = ["A", "B"]
    elif args.stage is not None:
        args.stages = [str(s).upper() for s in args.stage]
    else:
        args.stages = ["A", "B", "C", "D"]
    if args.skip_spk_variants and "D" in args.stages:
        args.stages = [s for s in args.stages if s != "D"]
    if args.bam_density is None:
        args.bam_density = default_bam_density()
    if args.spk_features is None:
        args.spk_features = list(SPK_FULL_COLS)
    if args.sweep_bam_density is not None and not args.sweep_bam_density:
        args.sweep_bam_density = list(DEFAULT_BAM_DENSITY_SWEEP)
    if args.splits is None:
        args.splits = list(SPLITS)
    else:
        unknown = [split for split in args.splits if split not in SPLITS]
        if unknown:
            raise SystemExit(
                f"unknown --splits {unknown} for --detector {args.detector} "
                f"--dataset {args.dataset} (choose from {list(SPLITS)})"
            )
    if args.seeds:
        args.seeds = [int(s) for s in args.seeds]
    else:
        args.seeds = [int(args.seed)]
    args.seed = int(args.seeds[0])
    if args.out is None:
        args.out = ARCH_DIR / "concept_head_ood"
    args.out_root = args.out
    return args


def main() -> None:
    args = parse_args()
    stages = set(args.stages)
    run_a = "A" in stages
    run_b = "B" in stages
    run_c = "C" in stages
    run_d = "D" in stages

    if run_c:
        args.out_root.mkdir(parents=True, exist_ok=True)
        migrate_legacy_heads(args.out_root, args.seeds[0])

    if run_c and not stage_c_can_reuse_activations(args):
        try:
            training = resolve_training_data(TRAINING_DATA)
        except FileNotFoundError:
            training = TRAINING_DATA
        if not training.is_file():
            raise FileNotFoundError(
                f"{TRAINING_DATA} (needed to train/rescore concept heads; "
                "eval-only reuse of activations.csv does not need this file — "
                "copy yolo-voc.pt from Drive semantic_training_data/ or omit --rescore/--retrain)"
            )

    args.extract_batch_resolved = int(args.extract_batch) if args.extract_batch > 0 else 0
    args._head_batch_cache = {}
    args._knn_tables = None
    print(
        f"detector={PROFILE.detector}  dataset={PROFILE.name}  root={ROOT}\n"
        f"  stages={args.stages}\n"
        f"  checkpoint={CHECKPOINT}\n"
        f"  training={TRAINING_DATA}\n"
        f"  roi={ROI_DIR}\n"
        f"  gt={GT_INDEX}\n"
        f"  out={args.out_root}/seed_{{s}}\n"
        f"  max_det={PROFILE.max_det}  roi_ch={PROFILE.feature_channels}  "
        f"vram_frac={args.vram_frac:.0%}  seeds={list(args.seeds)}",
        flush=True,
    )
    experiments_root = resolve_experiments_root(args)
    assets_shared_root = resolve_assets_shared_root(args)
    if experiments_root is not None:
        print(f"  experiments backup -> {experiment_subdir(experiments_root)}/", flush=True)
    else:
        print(
            "  experiments backup disabled (use --experiments-dir or mount Drive; --no-backup to silence)",
            flush=True,
        )
    if assets_shared_root is not None:
        print(f"  assets gt index -> {assets_gt_index_dest(assets_shared_root)}", flush=True)
    else:
        print(
            "  assets gt publish disabled (use --assets-dir or mount Drive; --no-backup to silence)",
            flush=True,
        )
    if run_b or run_c:
        print_skip_plan(args)
    t_all = time.perf_counter()
    timings: dict[str, float] = {}

    if run_a:
        print("=== [A] GT index ===")
        t0 = time.perf_counter()
        if gt_index_is_current(GT_INDEX, DATASET_DIR):
            print(f"  {GT_INDEX} matches {PROFILE.name} ID-train, skipping")
        else:
            if GT_INDEX.is_file():
                print(f"  {GT_INDEX} exists but does not match {PROFILE.name} ID-train; rebuilding")
            build_gt_index(DATASET_DIR, GT_INDEX)
        timings["A_gt_index"] = time.perf_counter() - t0
        print(f"  stage A wall {_fmt_duration(timings['A_gt_index'])}")
        if experiments_root is not None:
            backup_gt_index(experiments_root)
        if assets_shared_root is not None:
            publish_gt_index_to_assets(assets_shared_root)

    if run_b:
        print("=== [B] ROI extraction ===")
        t0 = time.perf_counter()
        engine = None
        summaries = []
        for split in args.splits:
            # 只有需要重新提取时才加载模型；已有缓存由 extract_split 直接复用。
            if any(split_needs_extract(split, args)) and engine is None:
                engine = build_engine(args.device)
                _ensure_extract_batch(engine, args)
            summaries.append(extract_split(split, engine, args))
        del engine
        timings["B_roi_extract"] = time.perf_counter() - t0
        summary_path = write_extraction_summary(summaries, timings["B_roi_extract"])
        print(f"  wrote {summary_path}")
        print(f"  stage B wall {_fmt_duration(timings['B_roi_extract'])}")
        if experiments_root is not None:
            backup_roi(experiments_root)
            backup_native_knn(experiments_root)

    if not run_c:
        if not run_d:
            timings["total"] = time.perf_counter() - t_all
            print("\n=== wall-clock ===")
            if run_a:
                print(f"  A  GT index        {_fmt_duration(timings['A_gt_index'])}")
            if run_b:
                print(f"  B  ROI extract     {_fmt_duration(timings['B_roi_extract'])}")
            print(f"  total              {_fmt_duration(timings['total'])}")
            return

    if run_c:
        print("=== [C] concept heads + OOD eval ===")
        t0 = time.perf_counter()
        seed_reports: dict[int, dict] = {}
        seed_times: dict[str, float] = {}
        for seed in args.seeds:
            args.seed = int(seed)
            args.out = seed_dir(args.out_root, seed)
            args.out.mkdir(parents=True, exist_ok=True)
            seed_everything(seed)
            print(f"\n----- seed {seed} -> {args.out} -----", flush=True)
            t_seed = time.perf_counter()
            seed_reports[seed] = train_and_eval(args)
            seed_times[f"C_seed_{seed}"] = time.perf_counter() - t_seed
            print(f"  seed {seed} wall {_fmt_duration(seed_times[f'C_seed_{seed}'])}", flush=True)
        timings.update(seed_times)
        timings["C_train_eval"] = time.perf_counter() - t0
        print(f"  stage C wall {_fmt_duration(timings['C_train_eval'])}")
        if len(seed_reports) > 1:
            write_seed_pool(args.out_root, seed_reports)
        timing_path = args.out_root / "timing.json"
        timing_path.write_text(json.dumps(timings, indent=2) + "\n")
        print(f"wrote {timing_path}")
        if experiments_root is not None:
            backup_concept_head_ood(experiments_root, args.out_root, list(args.seeds))
            backup_native_knn(experiments_root)

    if run_d:
        print("=== [D] SPK variants ===")
        t0 = time.perf_counter()
        run_stage_d(args)
        timings["D_spk_variants"] = time.perf_counter() - t0
        print(f"  stage D wall {_fmt_duration(timings['D_spk_variants'])}")
        if experiments_root is not None:
            backup_spk_variants(experiments_root)

    timings["total"] = time.perf_counter() - t_all
    print("\n=== wall-clock ===")
    if run_a:
        print(f"  A  GT index        {_fmt_duration(timings.get('A_gt_index', 0.0))}")
    if run_b:
        print(f"  B  ROI extract     {_fmt_duration(timings.get('B_roi_extract', 0.0))}")
    if run_c:
        for seed in args.seeds:
            print(f"  C  seed {seed}        {_fmt_duration(timings.get(f'C_seed_{seed}', 0.0))}")
        print(f"  C  train+eval      {_fmt_duration(timings.get('C_train_eval', 0.0))}")
    if run_d:
        print(f"  D  spk variants    {_fmt_duration(timings.get('D_spk_variants', 0.0))}")
    print(f"  total              {_fmt_duration(timings['total'])}")


if __name__ == "__main__":
    main()
