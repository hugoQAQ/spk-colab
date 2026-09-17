#!/usr/bin/env python3
"""Concept-head OOD experiment (YOLO or Faster R-CNN; VOC or BDD).

Three stages, run in order:

  A  GT INDEX   decode YOLO `.txt` labels in ID-train tars -> data/id/gt.json
  B  EXTRACT    one detector forward per image; cache float8 ROI features
                (YOLO 896x7x7, FRCNN 256x7x7)
                -> data/{detector}/{dataset}/roi/{id_train,id_val,near_ood,far_ood}.pt
  C  TRAIN+EVAL concept heads from data/{detector}/{dataset}/training_data.pt,
                SPK4 + native kNN, classwise Isolation Forest
                -> data/{detector}/{dataset}/concept_head_ood/

Checkpoints live under model/{detector}/ so one Colab session can hold YOLO and FRCNN.
Image tars under data/id and data/ood are shared across detectors.

Both detectors keep at most 30 boxes per image after NMS (`max_det=30`).

Examples
--------
    DETECTOR=yolo DATASET=voc bash mount_data.sh
    python run.py --root /content/spk --detector yolo --dataset voc

    DETECTOR=yolo DATASET=bdd bash mount_data.sh
    python run.py --root /content/spk --detector yolo --dataset bdd --max-images 200 --epochs 6

    DETECTOR=frcnn DATASET=voc bash mount_data.sh
    python run.py --root /content/spk --detector frcnn --dataset voc
    # checkpoint: /content/spk/model/frcnn/voc_vanilla.pth
    # outputs:    /content/spk/data/frcnn/voc/concept_head_ood/
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision.ops import nms, roi_align
from tqdm.auto import tqdm

# `__file__` is absent when the script is piped into a remote kernel (Colab CLI),
# so fall back to the cwd and let --root override in every case.
ROOT = Path(globals().get("__file__", "run.py")).resolve().parent

CHECKPOINT = ROOT / "model/yolo/voc_vanilla.pt"
FRCNN_CFG = ROOT / "model/frcnn/frcnn_fx/FX_vanilla_voc.yaml"
DATASET_DIR = ROOT / "data/id"
OOD_DIR = ROOT / "data/ood"
ARCH_DIR = ROOT / "data/yolo/voc"
ROI_DIR = ARCH_DIR / "roi"
GT_INDEX = ROOT / "data/id/gt.json"
TRAINING_DATA = ARCH_DIR / "training_data.pt"
NATIVE_ROOT = ARCH_DIR / "native_knn"
PRIOR_DIR = ARCH_DIR / "detection_prior_rows"

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


VOC_PROFILE = DatasetProfile(
    name="voc",
    checkpoint="voc_vanilla.pt",
    splits=("voc_train", "voc_val", "near_ood", "far_ood"),
    id_splits=frozenset({"voc_train", "voc_val"}),
    split_tar_prefix={
        "voc_train": "voc_yolo_train",
        "voc_val": "voc_yolo_val",
        "near_ood": "near_ood",
        "far_ood": "far_ood",
    },
    split_output_names={
        "voc_train": "id_train.pt",
        "voc_val": "id_val.pt",
        "near_ood": "near_ood.pt",
        "far_ood": "far_ood.pt",
    },
    split_to_knn={
        "voc_train": "voc_id_train",
        "voc_val": "voc_id_val",
        "near_ood": "near_ood_voc",
        "far_ood": "far_ood_voc",
    },
    split_to_prior={
        "voc_train": ("train_detector_rows.csv", "id_train_tp"),
        "voc_val": ("val_detector_rows.csv", "id_val_tp"),
        "near_ood": ("near_detector_rows.csv", "near_ood_fp"),
        "far_ood": ("far_detector_rows.csv", "far_ood_fp"),
    },
    native_split_specs={
        "voc_id_train": {"data_source": "id_train_tp", "protocol": "voc_fp8_train_pool"},
        "voc_id_val": {"data_source": "id_val_tp", "protocol": "voc_id"},
        "near_ood_voc": {"data_source": "near_ood_fp", "protocol": "near_far_ood"},
        "far_ood_voc": {"data_source": "far_ood_fp", "protocol": "near_far_ood"},
    },
    label_classes=tuple(VOC20),
    eval_classes=VOC14,
    gt_train_glob="voc_yolo_train-*.tar",
    gt_fixed_size=None,
    roi_cache_type="voc_fp8_roi_features",
)

BDD_PROFILE = DatasetProfile(
    name="bdd",
    checkpoint="bdd_vanilla.pt",
    splits=("bdd_train", "bdd_val", "near_ood", "far_ood"),
    id_splits=frozenset({"bdd_train", "bdd_val"}),
    split_tar_prefix={
        "bdd_train": "bdd_train",
        "bdd_val": "bdd_val",
        "near_ood": "near_ood",
        "far_ood": "far_ood",
    },
    split_output_names={
        "bdd_train": "id_train.pt",
        "bdd_val": "id_val.pt",
        "near_ood": "near_ood.pt",
        "far_ood": "far_ood.pt",
    },
    split_to_knn={
        "bdd_train": "bdd_id_train",
        "bdd_val": "bdd_id_val",
        "near_ood": "near_ood_bdd",
        "far_ood": "far_ood_bdd",
    },
    split_to_prior={
        "bdd_train": ("train_detector_rows.csv", "id_train_tp"),
        "bdd_val": ("val_detector_rows.csv", "id_val_tp"),
        "near_ood": ("near_detector_rows.csv", "near_ood_fp"),
        "far_ood": ("far_detector_rows.csv", "far_ood_fp"),
    },
    native_split_specs={
        "bdd_id_train": {"data_source": "id_train_tp", "protocol": "bdd_fp8_train_pool"},
        "bdd_id_val": {"data_source": "id_val_tp", "protocol": "bdd_id"},
        "near_ood_bdd": {"data_source": "near_ood_fp", "protocol": "near_far_ood"},
        "far_ood_bdd": {"data_source": "far_ood_fp", "protocol": "near_far_ood"},
    },
    label_classes=tuple(BDD10),
    eval_classes=frozenset(BDD10),
    gt_train_glob="bdd_train-*.tar",
    gt_fixed_size=(1280, 720),
    roi_cache_type="bdd_fp8_roi_features",
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

PROFILES: dict[tuple[str, str], DatasetProfile] = {
    ("yolo", "voc"): VOC_PROFILE,
    ("yolo", "bdd"): BDD_PROFILE,
    ("frcnn", "voc"): FRCNN_VOC_PROFILE,
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
    global TRAINING_DATA, NATIVE_ROOT, PRIOR_DIR, FRCNN_CFG, ARCH_DIR
    CHECKPOINT = ROOT / "model" / PROFILE.detector / PROFILE.checkpoint
    DATASET_DIR = ROOT / "data/id"
    OOD_DIR = ROOT / "data/ood"
    ARCH_DIR = ROOT / "data" / PROFILE.detector / PROFILE.name
    ROI_DIR = ARCH_DIR / "roi"
    GT_INDEX = ROOT / "data/id/gt.json"
    TRAINING_DATA = ARCH_DIR / "training_data.pt"
    NATIVE_ROOT = ARCH_DIR / "native_knn"
    PRIOR_DIR = ARCH_DIR / "detection_prior_rows"
    FRCNN_CFG = ROOT / PROFILE.frcnn_config


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
NATIVE_K = 5
ID_VAL_OUTLIER_FRACTION = 0.05

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
    if peak <= budget:
        best = min_batch
        while hi < max_batch:
            nxt = min(hi * 2, max_batch)
            peak = _peak(nxt)
            if peak is None or peak > budget:
                lo, hi = best, nxt
                break
            best, lo, hi = nxt, nxt, nxt
            if nxt == max_batch:
                lo, hi = nxt, nxt
                break
    else:
        print(
            f"  {label}: batch={min_batch} already uses {peak / 1e9:.1f} GB "
            f"(budget {budget / 1e9:.1f} GB of {total / 1e9:.1f} GB); keeping {min_batch}",
            flush=True,
        )
        return min_batch

    if lo < hi:
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            peak = _peak(mid)
            if peak is not None and peak <= budget:
                best, lo = mid, mid
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
        dummy = np.full((800, 1200, 3), 114, dtype=np.uint8)

        def trial(batch: int) -> None:
            engine.infer_batch_bgr([dummy] * batch)

        chosen = choose_batch_size(
            trial, args.device, args.vram_frac,
            min_batch=1, max_batch=8, default=1, label="extract batch",
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
        bbox, score, label = pred.split((4, 1, 1), dim=-1)
        detections: list[dict[str, Any]] = []
        for row in torch.cat([bbox, score, label], dim=-1):
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

    @torch.inference_mode()
    def infer_batch_bgr(self, images_bgr: list[np.ndarray]) -> list[UnifiedImageResult]:
        if not images_bgr:
            return []
        batched: list[dict[str, Any]] = []
        orig_hw: list[tuple[int, int]] = []
        for image_bgr in images_bgr:
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            orig_hw.append((height, width))
            tensor = torch.as_tensor(image.transpose(2, 0, 1), device=self.device)
            batched.append({"image": tensor, "height": height, "width": width})

        images = self.model.preprocess_image(batched)
        features = self.model.backbone(images.tensor)
        outputs = self.model.inference(batched)

        results: list[UnifiedImageResult] = []
        for batch_idx, output in enumerate(outputs):
            orig_h, orig_w = orig_hw[batch_idx]
            model_h, model_w = images.image_sizes[batch_idx]
            detections = self._parse_instances(output["instances"])
            feat_i = {name: features[name][batch_idx : batch_idx + 1] for name in features}
            native = self._pool_native(feat_i)
            if detections:
                boxes = torch.tensor([d["bbox_xyxy"] for d in detections], dtype=torch.float32)
                boxes_model = boxes.clone()
                boxes_model[:, [0, 2]] *= float(model_w) / max(orig_w, 1)
                boxes_model[:, [1, 3]] *= float(model_h) / max(orig_h, 1)
                roi = self._roi_from_features(feat_i, boxes_model)
            else:
                roi = torch.empty((0, self.roi_channels, ROI_SIZE, ROI_SIZE), dtype=torch.float32)
            results.append(UnifiedImageResult(detections, native, roi))
        return results


def build_engine(device: str | torch.device):
    if PROFILE.detector == "frcnn":
        return FRCNNUnifiedForward(CHECKPOINT, device)
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


def split_needs_extract(split: str, args: argparse.Namespace) -> tuple[bool, bool, bool]:
    """Return (need_roi, need_native, need_prior) for this split."""
    roi_path = ROI_DIR / SPLIT_OUTPUT_NAMES[split]
    skip_roi_write = bool(getattr(args, "_skip_roi_write", False))
    need_roi = (not skip_roi_write) and (args.force or not roi_path.is_file())
    need_native = args.with_native and not native_ready(native_split_dir(split))
    need_prior = bool(args.with_prior)
    return need_roi, need_native, need_prior


def extract_split(split: str, engine, args: argparse.Namespace) -> dict[str, Any]:
    """Detector forward over one split; write the ROI cache (plus optional extras)."""
    roi_path = ROI_DIR / SPLIT_OUTPUT_NAMES[split]
    knn_split = SPLIT_TO_KNN[split]
    prior_name, prior_source = SPLIT_TO_PRIOR[split]
    prior_path = PRIOR_DIR / prior_name
    native_dir = native_split_dir(split)
    need_roi, need_native, need_prior = split_needs_extract(split, args)
    if not need_roi and not need_native and not need_prior:
        print(f"  {split}: caches exist, skipping (use --force to redo ROI)")
        return {"split": split, "roi": {"status": "skipped_existing"}}

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
                    chunk_image_rows.append({**common, "num_detections": len(result.detections)})
                    for det_idx, det in enumerate(result.detections):
                        chunk_detection_rows.append(
                            {**common, "detection_index_in_image": det_idx, **det}
                        )
                    native_image_rows.append(chunk_image_rows[-1])
                    native_detection_rows.extend(chunk_detection_rows[-len(result.detections):] if result.detections else [])
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
        summary["roi"] = {"output": str(roi_path.resolve()), "num_detections": int(payload["num_detections"])}
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

    compact_schema = "per_class" not in bundle
    part_based = bool(prox_block is not None) if compact_schema else bool(
        entry.get("is_part_based", True)
    )

    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    if part_based:
        if prox_block is None:
            raise ValueError(f"{class_name}: missing prox block")
        concept_order = (
            [f"id::{name}" for name in id_block["concept_order"]]
            + [f"prox::{name}" for name in prox_block["concept_order"]]
            + ["unknown"]
        )
        n_id = len(id_block["concept_order"])
        n_prox = len(prox_block["concept_order"])
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
            raise ValueError(f"{class_name}: no positive ID samples")
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
    scores = pool_logits(logits)
    group_scores = torch.stack([scores[:, g].amax(1) for g in groups], dim=1)
    group_mass = torch.stack([target[:, g].flatten(1).sum(1) for g in groups], dim=1)
    valid = group_mass.sum(1) > 0
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
    sampler = WeightedRandomSampler(1.0 / counts[group_of_row], len(train_idx), replacement=True)

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

def score_roi_cache(
    cache_path: Path, split_name: str, heads: dict, device: str, batch_size: int = HEAD_BATCH,
) -> pd.DataFrame:
    """Run each class's head over the ROIs the detector predicted as that class."""
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    metadata, fp8, scales = cache["metadata"], cache["features_fp8"], cache["scales"]

    rows_by_class: dict[str, list[int]] = {}
    for i, row in enumerate(metadata):
        predicted = row.get("pred_class") or row.get("class")
        if predicted in heads:
            rows_by_class.setdefault(predicted, []).append(i)

    torch_device = torch.device(device)
    records = []
    for class_name, row_indices in rows_by_class.items():
        head, concept_order = heads[class_name]
        id_channels = [i for i, n in enumerate(concept_order) if n.startswith("id::")]
        prox_channels = [i for i, n in enumerate(concept_order) if n.startswith("prox::")]
        unknown_channel = concept_order.index("unknown")

        for start in range(0, len(row_indices), batch_size):
            chunk = row_indices[start:start + batch_size]
            # float8 tensors do not support fancy indexing on CPU, hence the stack.
            xb = torch.stack([fp8[i].float() * scales[i] for i in chunk]).to(torch_device)
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
        candidates = ground_truth.get(Path(file_name).stem, [])
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


def drop_lowest_id_val(scores: np.ndarray, fraction: float = ID_VAL_OUTLIER_FRACTION) -> np.ndarray:
    """Drop the least ID-like ID-val scores (Isolation Forest: low = outlier)."""
    drop = min(int(np.floor(fraction * len(scores))), max(len(scores) - 5, 0))
    if drop <= 0:
        return scores
    return np.sort(scores)[drop:]


def evaluate(
    activations: pd.DataFrame,
    class_names: list[str],
    id_train_tp: pd.DataFrame,
    feature_cols: list[str],
    drop_id_val_outliers: bool,
) -> dict:
    """One Isolation Forest per class; report per-class and pooled FPR95."""
    per_class, pooled = {}, {"id_val": [], "near_ood": [], "far_ood": []}

    for class_name in class_names:
        def split(frame: pd.DataFrame) -> np.ndarray:
            return frame.loc[frame["class"] == class_name, feature_cols].dropna().to_numpy()

        train = split(id_train_tp)
        id_val = split(activations[activations["data_source"] == "id_val"])
        near = split(activations[activations["data_source"] == "near_ood"])
        far = split(activations[activations["data_source"] == "far_ood"])
        if len(train) < 5 or len(id_val) < 5:
            per_class[class_name] = {"skipped": True, "n_train": len(train), "n_id_val": len(id_val)}
            continue
        if len(train) > 1500:  # cap keeps fitting fast; 1500 is plenty for 4-5 features
            train = train[np.random.default_rng(42).choice(len(train), 1500, replace=False)]

        scaler = StandardScaler().fit(train)
        forest = IsolationForest(
            n_estimators=200, contamination=0.05,
            max_samples=min(512, len(train)), random_state=42, n_jobs=-1,
        ).fit(scaler.transform(train))

        scores = {
            name: forest.decision_function(scaler.transform(data)) if len(data) else np.zeros(0)
            for name, data in [("id_val", id_val), ("near_ood", near), ("far_ood", far)]
        }
        # ID-val also contains detector mistakes. We have no ground truth for it,
        # so optionally drop the 5% the forest itself finds least ID-like.
        n_id_val_raw = len(scores["id_val"])
        if drop_id_val_outliers:
            scores["id_val"] = drop_lowest_id_val(scores["id_val"])
        for name in pooled:
            pooled[name].append(scores[name])

        near_fpr = fpr95(scores["id_val"], scores["near_ood"])
        far_fpr = fpr95(scores["id_val"], scores["far_ood"])
        per_class[class_name] = {
            "n_train": len(train),
            "n_id_val": n_id_val_raw, "n_id_val_kept": len(scores["id_val"]),
            "n_near_ood": len(near), "n_far_ood": len(far),
            "near_fpr95": near_fpr, "far_fpr95": far_fpr,
            "mean_fpr95": float(np.nanmean([near_fpr, far_fpr])),
        }
        print(f"  {class_name:8s} near={near_fpr:6.2f}  far={far_fpr:6.2f}", flush=True)

    # Pooled: concatenate the per-class IF scores, then one global threshold.
    cat = {name: np.concatenate(parts) if parts else np.zeros(0) for name, parts in pooled.items()}
    near_fpr = fpr95(cat["id_val"], cat["near_ood"])
    far_fpr = fpr95(cat["id_val"], cat["far_ood"])
    return {
        "method": "spk full" if KNN_COL in feature_cols else "spk local",
        "features": list(feature_cols),
        "ood_protocol": "near_far_ood",
        "id_train_filter": "same predicted class and IoU >= 0.5 with ground truth",
        "id_val_filter": (
            "drop each class's lowest 5% Isolation Forest scores"
            if drop_id_val_outliers else "none (all ID-val detections of the predicted class)"
        ),
        "per_class": per_class,
        "pooled": {
            "near_fpr95": near_fpr, "far_fpr95": far_fpr,
            "mean_fpr95": float(np.nanmean([near_fpr, far_fpr])),
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
    split_dirs = {
        "id_train": NATIVE_ROOT / "id_train",
        "id_val": NATIVE_ROOT / "id_val",
        "near_ood": NATIVE_ROOT / "near_ood",
        "far_ood": NATIVE_ROOT / "far_ood",
    }
    for data_source, native_dir in split_dirs.items():
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


def train_and_eval(args: argparse.Namespace) -> None:
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
        training_data = torch.load(
            resolve_training_data(TRAINING_DATA), map_location="cpu", weights_only=False
        )
        class_names = args.classes or list(training_data)
        missing = sorted(set(class_names) - set(training_data))
        if missing:
            raise SystemExit(f"classes not in training_data.pt: {missing}")

        print(f"  {len(class_names)} class(es) on {args.device}")
        heads = {}
        for class_name in class_names:
            head_path = args.out / f"{class_name}_head.pt"
            if head_path.is_file() and not args.retrain:
                head, concept_order = load_saved_head(head_path, args.device)
                heads[class_name] = (head, concept_order)
                print(f"  [{class_name}] reused {head_path.name} ({len(concept_order)} concepts)", flush=True)
                continue
            x, y, concept_order, groups = build_class_data(training_data, class_name)
            print(f"  [{class_name}] {len(x):,} ROIs, {len(concept_order)} concepts", flush=True)
            head = train_head(class_name, x, y, groups, args)
            heads[class_name] = (head, concept_order)
            torch.save({"state_dict": head.state_dict(), "concept_order": concept_order,
                        "class_name": class_name}, head_path)
            del x, y
        del training_data

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
            frames.append(frame)
        activations = pd.concat(frames, ignore_index=True)
        del heads

    print("  ensuring image-level native embeddings")
    ensure_native_embeddings(args)

    id_train = activations[activations["data_source"] == "id_train"]
    id_train_tp = keep_true_positives(id_train, GT_INDEX)
    print(f"  ID-train true positives: {len(id_train_tp):,} of {len(id_train):,}", flush=True)

    knn_tables = build_native_knn_tables(id_train_tp, class_names)
    tp_index = id_train_tp.index
    activations = attach_native_knn(activations, knn_tables)
    id_train_tp = activations.loc[tp_index]
    n_knn = int(activations[KNN_COL].notna().sum())
    print(f"  attached {KNN_COL} to {n_knn:,}/{len(activations):,} activation rows", flush=True)
    activations.to_csv(activations_path, index=False)

    methods = [
        ("spk local", SPK4_COLS),
        ("spk full", SPK4_COLS + [KNN_COL]),
    ]
    outlier_settings = [
        ("without_outlier_removal", False),
        ("with_outlier_removal", True),
    ]
    report: dict[str, Any] = {
        "dataset": PROFILE.name,
        "detector": PROFILE.detector,
        "ood_protocol": "near_far_ood",
        "id_train_filter": "same predicted class and IoU >= 0.5 with ground truth",
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
        for setting_name, drop in outlier_settings:
            label = "drop_lowest_5pct" if drop else "none"
            print(f"\n=== {method_name}  features={feature_cols}  id_val outlier removal: {label} ===")
            result = evaluate(
                activations, class_names, id_train_tp, feature_cols, drop_id_val_outliers=drop,
            )
            pooled = result["pooled"]
            print(
                f"pooled  near={pooled['near_fpr95']:.2f}  far={pooled['far_fpr95']:.2f}  "
                f"mean={pooled['mean_fpr95']:.2f}"
            )
            report["methods"][method_name][setting_name] = result

    print("\n=== pooled summary ===")
    print(f"{'method':12s} {'outlier_removal':20s} {'near':8s} {'far':8s} {'mean':8s}")
    for method_name, _ in methods:
        for setting_name, drop in outlier_settings:
            pooled = report["methods"][method_name][setting_name]["pooled"]
            label = "drop_lowest_5pct" if drop else "none"
            print(
                f"{method_name:12s} {label:20s} "
                f"{pooled['near_fpr95']:8.2f} {pooled['far_fpr95']:8.2f} {pooled['mean_fpr95']:8.2f}"
            )

    (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}/results.json and {args.out}/activations.csv")
    return report


def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s ({seconds:.1f}s)"
    if minutes:
        return f"{minutes}m {secs:02d}s ({seconds:.1f}s)"
    return f"{seconds:.1f}s"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--detector", choices=("yolo", "frcnn"), default="yolo",
                        help="yolo or frcnn (checkpoint, ROI channels, native pooling)")
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
    args = parser.parse_args()
    set_profile(args.detector, args.dataset)
    if args.root is not None:
        set_root(args.root)
    else:
        refresh_paths()
    if args.splits is None:
        args.splits = list(SPLITS)
    else:
        unknown = [split for split in args.splits if split not in SPLITS]
        if unknown:
            raise SystemExit(
                f"unknown --splits {unknown} for --detector {args.detector} "
                f"--dataset {args.dataset} (choose from {list(SPLITS)})"
            )
    if args.out is None:
        args.out = ARCH_DIR / "concept_head_ood"

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    args.seed = SEED
    args.out.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if PROFILE.detector == "frcnn" and not FRCNN_CFG.is_file():
        raise FileNotFoundError(FRCNN_CFG)
    if not resolve_training_data(TRAINING_DATA).is_file():
        raise FileNotFoundError(TRAINING_DATA)

    args.extract_batch_resolved = int(args.extract_batch) if args.extract_batch > 0 else 0
    args._head_batch_cache = {}
    print(
        f"detector={PROFILE.detector}  dataset={PROFILE.name}  root={ROOT}\n"
        f"  checkpoint={CHECKPOINT}\n"
        f"  training={TRAINING_DATA}\n"
        f"  roi={ROI_DIR}\n"
        f"  out={args.out}\n"
        f"  max_det={PROFILE.max_det}  roi_ch={PROFILE.feature_channels}  "
        f"vram_frac={args.vram_frac:.0%}",
        flush=True,
    )
    t_all = time.perf_counter()
    timings: dict[str, float] = {}

    print("=== [A] GT index ===")
    t0 = time.perf_counter()
    if GT_INDEX.is_file():
        print(f"  {GT_INDEX} exists, skipping")
    else:
        build_gt_index(DATASET_DIR, GT_INDEX)
    timings["A_gt_index"] = time.perf_counter() - t0
    print(f"  stage A wall {_fmt_duration(timings['A_gt_index'])}")

    print("=== [B] ROI extraction ===")
    t0 = time.perf_counter()
    engine = None
    summaries = []
    for split in args.splits:
        need_roi, need_native, need_prior = split_needs_extract(split, args)
        if not need_roi and not need_native and not need_prior:
            summaries.append(extract_split(split, None, args))
            continue
        if engine is None:
            engine = build_engine(args.device)
            _ensure_extract_batch(engine, args)
        summaries.append(extract_split(split, engine, args))
    del engine
    timings["B_roi_extract"] = time.perf_counter() - t0
    (ROI_DIR / "extraction_summary.json").write_text(
        json.dumps(
            {
                "detector": PROFILE.detector,
                "dataset": PROFILE.name,
                "checkpoint": str(CHECKPOINT.resolve()),
                "max_det": PROFILE.max_det,
                "splits": summaries,
                "elapsed_sec": timings["B_roi_extract"],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"  stage B wall {_fmt_duration(timings['B_roi_extract'])}")

    print("=== [C] concept heads + OOD eval ===")
    t0 = time.perf_counter()
    results = train_and_eval(args)
    timings["C_train_eval"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - t_all
    print(f"  stage C wall {_fmt_duration(timings['C_train_eval'])}")

    print("\n=== wall-clock ===")
    print(f"  A  GT index        {_fmt_duration(timings['A_gt_index'])}")
    print(f"  B  ROI extract     {_fmt_duration(timings['B_roi_extract'])}")
    print(f"  C  train+eval      {_fmt_duration(timings['C_train_eval'])}")
    print(f"  total              {_fmt_duration(timings['total'])}")

    timing_path = args.out / "timing.json"
    timing_path.write_text(json.dumps(timings, indent=2) + "\n")
    if isinstance(results, dict):
        results["timing_sec"] = timings
        (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {timing_path}")


if __name__ == "__main__":
    main()
