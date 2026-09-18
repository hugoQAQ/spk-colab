#!/usr/bin/env python3
"""Extract Faster R-CNN classification logits (one dict per split, keyed by class).

Self-contained: no import from run.py. Reads image tars, runs Detectron2 FX FRCNN,
applies the same conf/NMS/max_det settings as run.py, and saves per-split artifacts.

Each split file (e.g. voc-train.pt / bdd-train.pt) is a dict:
  class_names  list[str]          model logit order (length C = K foreground + background)
  background_index  int           last column index (= K)
  by_class     dict[str, dict]    predicted class -> rows for that class
    <class>:
      logits         float32 [N, C]   pre-softmax class logits (incl. background column)
      pred_labels    int64   [N]      foreground class index (= logit column 0..K-1)
      confidences    float32 [N]      softmax prob of pred foreground class
      file_names     list[str] [N]
      bboxes_xyxy    float32 [N, 4]    original-image coordinates
      detection_ids  list[str] [N]

Also includes flat concatenation across classes:
  logits, pred_labels, confidences, file_names, bboxes_xyxy, detection_ids

Examples
--------
    # Colab: clone repo (frcnn_fx ships in-repo), mount weights + image tars, then extract
    git clone <repo-url> /content/spk-colab
    cd /content/spk-colab
    bash mount_frcnn_logits.sh
    python frcnn_logits_extraction.py --dataset voc --root /content/spk

    # BDD
    DATASET=bdd bash mount_frcnn_logits.sh
    python frcnn_logits_extraction.py --dataset bdd --root /content/spk
    python frcnn_logits_extraction.py --dataset bdd --splits bdd-train bdd-val --max-images 200
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Match run.py detector settings.
CONF = 0.25
IOU = 0.7
MAX_DET = 30
DECODE_BATCH = 8
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

VOC14 = frozenset({
    "bird", "bottle", "car", "cat", "cow", "dog", "horse", "person", "sheep",
    "bicycle", "boat", "bus", "chair", "train",
})

BDD10 = frozenset({
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
    "traffic_light", "traffic_sign",
})

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

BACKGROUND_NAME = "background"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    checkpoint: str
    detector_class_names: tuple[str, ...]
    split_specs: dict[str, dict[str, str]]
    label_aliases: dict[str, str] = field(default_factory=dict)
    eval_classes: frozenset[str] | None = None


DATASETS: dict[str, DatasetConfig] = {
    "voc": DatasetConfig(
        name="voc",
        checkpoint="voc_vanilla.pth",
        detector_class_names=FRCNN_VOC20,
        split_specs={
            "voc-train": {"prefix": "voc_yolo_train", "search": "id"},
            "voc-val": {"prefix": "voc_yolo_val", "search": "id"},
            "near-ood": {"prefix": "near_ood_voc", "search": "ood"},
            "far-ood": {"prefix": "far_ood", "search": "ood"},
        },
        label_aliases={
            "airplane": "aeroplane",
            "motorcycle": "motorbike",
            "dining table": "diningtable",
            "potted plant": "pottedplant",
            "couch": "sofa",
            "tv": "tvmonitor",
        },
        eval_classes=VOC14,
    ),
    "bdd": DatasetConfig(
        name="bdd",
        checkpoint="bdd_vanilla.pth",
        detector_class_names=FRCNN_BDD10,
        split_specs={
            "bdd-train": {"prefix": "bdd_train_10k", "search": "id"},
            "bdd-val": {"prefix": "bdd_val", "search": "id"},
            "near-ood": {"prefix": "near_ood_bdd", "search": "ood"},
            "far-ood": {"prefix": "far_ood", "search": "ood"},
        },
        label_aliases={
            "pedestrian": "person",
            "traffic light": "traffic_light",
            "traffic sign": "traffic_sign",
        },
        eval_classes=BDD10,
    ),
}


@dataclass(frozen=True)
class ImageItem:
    file_name: str
    tar_member: str
    tar_path: Path


class SplitImageSource:
    def __init__(self, split: str, root: Path, dataset: DatasetConfig, max_images: int = 0) -> None:
        spec = dataset.split_specs[split]
        search_dir = root / "data" / spec["search"]
        self.tar_paths = sorted(search_dir.glob(f"{spec['prefix']}-*.tar"))
        if not self.tar_paths:
            raise FileNotFoundError(
                f"No {spec['prefix']}-*.tar for split {split!r} under {search_dir}"
            )
        self.max_images = max_images
        self._tar: tarfile.TarFile | None = None
        self._members_by_tar: dict[Path, list[tarfile.TarInfo]] = {}

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None

    def __enter__(self) -> SplitImageSource:
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
        total = sum(len(self._get_members(p)) for p in self.tar_paths)
        return min(total, self.max_images) if self.max_images > 0 else total

    def iter_items(self) -> Iterator[ImageItem]:
        count = 0
        for tar_path in self.tar_paths:
            for member in self._get_members(tar_path):
                if self.max_images > 0 and count >= self.max_images:
                    return
                yield ImageItem(Path(member.name).name, member.name, tar_path)
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


def iter_image_batches(source: SplitImageSource, batch_size: int) -> Iterator[list[ImageItem]]:
    batch: list[ImageItem] = []
    for item in source.iter_items():
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def canonical_class(raw_label: str, aliases: dict[str, str]) -> str:
    text = str(raw_label).strip()
    lowered = text.lower()
    return aliases.get(lowered, aliases.get(text, text))


class FRCNNLogitsExtractor:
    """Detectron2 FX Faster R-CNN with pre-softmax class logits per detection."""

    def __init__(
        self,
        checkpoint: Path,
        cfg_path: Path,
        device: str | torch.device,
        *,
        detector_class_names: tuple[str, ...],
        label_aliases: dict[str, str] | None = None,
        eval_classes: frozenset[str] | None = None,
    ) -> None:
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import get_cfg
        from detectron2.modeling import build_model

        self.device = torch.device(device)
        self.label_aliases = label_aliases or {}
        self.eval_classes = eval_classes
        self.checkpoint_path = str(checkpoint.resolve())
        self.foreground_names = tuple(str(n) for n in detector_class_names)
        self.class_names = list(self.foreground_names) + [BACKGROUND_NAME]
        self.background_index = len(self.foreground_names)

        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"{cfg_path} (bundled frcnn_fx/ should ship with this repo; "
                "mount_frcnn_logits.sh also copies it under model/frcnn/frcnn_fx/)"
            )
        fx_dir = cfg_path.parent
        if str(fx_dir) not in sys.path:
            sys.path.insert(0, str(fx_dir))
        import utils.fxrcnn  # noqa: F401,E402  registers FXGeneralizedRCNN

        cfg = get_cfg()
        cfg.merge_from_file(str(cfg_path))
        cfg.MODEL.WEIGHTS = str(checkpoint.resolve())
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.foreground_names)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONF
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = IOU
        cfg.TEST.DETECTIONS_PER_IMAGE = MAX_DET
        cfg.MODEL.DEVICE = str(self.device)
        cfg.freeze()
        self.model = build_model(cfg)
        self.model.eval()
        DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

    def _class_name(self, foreground_idx: int) -> str:
        if not 0 <= foreground_idx < len(self.foreground_names):
            raise IndexError(f"foreground class index {foreground_idx} out of range")
        raw = self.foreground_names[foreground_idx].strip().lower()
        return canonical_class(raw, self.label_aliases)

    @torch.inference_mode()
    def _infer_one_bgr(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        orig_h, orig_w = image_bgr.shape[:2]
        tensor = torch.as_tensor(image_bgr.transpose(2, 0, 1).copy(), device=self.device)
        batched = [{"image": tensor, "height": orig_h, "width": orig_w}]

        outputs = self.model(batched)
        inst = outputs[0]["instances"]
        if inst is None or len(inst) == 0:
            return []
        if not hasattr(inst, "pred_logits"):
            raise RuntimeError(
                "FXGeneralizedRCNN did not attach pred_logits; "
                "ensure bundled spk-colab/frcnn_fx is on sys.path"
            )

        boxes = inst.pred_boxes.tensor.detach().cpu()
        scores = inst.scores.detach().cpu()
        pred_classes = inst.pred_classes.detach().cpu()
        pred_logits = inst.pred_logits.detach().cpu()

        detections: list[dict[str, Any]] = []
        for i in range(len(inst)):
            fg_idx = int(pred_classes[i].item())
            pred_class = self._class_name(fg_idx)
            if self.eval_classes is not None and pred_class not in self.eval_classes:
                continue
            detections.append({
                "bbox_xyxy": boxes[i],
                "confidence": float(scores[i].item()),
                "pred_label": fg_idx,
                "pred_class": pred_class,
                "detector_label": self.foreground_names[fg_idx],
                "logits": pred_logits[i].float(),
            })
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections[:MAX_DET]

    @torch.inference_mode()
    def infer_batch_bgr(self, images_bgr: list[np.ndarray]) -> list[list[dict[str, Any]]]:
        # One image at a time: FPN pads batches to max H/W; Far-OOD images can be 4k.
        return [self._infer_one_bgr(image_bgr) for image_bgr in images_bgr]


def accumulate_rows(
    split: str,
    class_names: list[str],
    background_index: int,
    buckets: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    by_class: dict[str, Any] = {}
    flat_logits, flat_labels, flat_conf, flat_names, flat_boxes, flat_ids = [], [], [], [], [], []

    for cls_name, rows in sorted(buckets.items()):
        if not rows:
            continue
        logits = torch.stack([r["logits"] for r in rows]).numpy().astype(np.float32)
        pred_labels = np.array([r["pred_label"] for r in rows], dtype=np.int64)
        confidences = np.array([r["confidence"] for r in rows], dtype=np.float32)
        file_names = [r["file_name"] for r in rows]
        bboxes = torch.stack([r["bbox_xyxy"] for r in rows]).numpy().astype(np.float32)
        detection_ids = [r["detection_id"] for r in rows]
        by_class[cls_name] = {
            "logits": logits,
            "pred_labels": pred_labels,
            "confidences": confidences,
            "file_names": file_names,
            "bboxes_xyxy": bboxes,
            "detection_ids": detection_ids,
            "count": len(rows),
        }
        flat_logits.append(logits)
        flat_labels.append(pred_labels)
        flat_conf.append(confidences)
        flat_names.extend(file_names)
        flat_boxes.append(bboxes)
        flat_ids.extend(detection_ids)

    payload: dict[str, Any] = {
        "split": split,
        "class_names": class_names,
        "background_index": background_index,
        "num_classes": len(class_names),
        "by_class": by_class,
        "num_detections": sum(v["count"] for v in by_class.values()),
    }
    if flat_logits:
        payload["logits"] = np.concatenate(flat_logits, axis=0)
        payload["pred_labels"] = np.concatenate(flat_labels, axis=0)
        payload["confidences"] = np.concatenate(flat_conf, axis=0)
        payload["file_names"] = flat_names
        payload["bboxes_xyxy"] = np.concatenate(flat_boxes, axis=0)
        payload["detection_ids"] = flat_ids
    else:
        payload["logits"] = np.zeros((0, len(class_names)), dtype=np.float32)
        payload["pred_labels"] = np.zeros((0,), dtype=np.int64)
        payload["confidences"] = np.zeros((0,), dtype=np.float32)
        payload["file_names"] = []
        payload["bboxes_xyxy"] = np.zeros((0, 4), dtype=np.float32)
        payload["detection_ids"] = []
    return payload


def extract_split(
    split: str,
    root: Path,
    dataset: DatasetConfig,
    extractor: FRCNNLogitsExtractor,
    out_dir: Path,
    decode_batch: int,
    max_images: int,
) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    det_counter = 0
    t0 = time.perf_counter()

    with SplitImageSource(split, root, dataset, max_images=max_images) as source:
        num_images = source.count_images()
        with tqdm(total=num_images, desc=split, unit="img") as pbar:
            for batch_items in iter_image_batches(source, decode_batch):
                images_bgr = []
                for item in batch_items:
                    bgr = source.load_bgr(item)
                    if bgr is None:
                        raise FileNotFoundError(f"failed to load {item.file_name}")
                    images_bgr.append(bgr)
                for item, dets in zip(batch_items, extractor.infer_batch_bgr(images_bgr)):
                    for det in dets:
                        det_counter += 1
                        row = {
                            **det,
                            "file_name": item.file_name,
                            "detection_id": f"{split}:{det_counter}",
                        }
                        buckets[det["pred_class"]].append(row)
                    pbar.update(1)
                    pbar.set_postfix(dets=det_counter, refresh=False)

    payload = accumulate_rows(
        split, extractor.class_names, extractor.background_index, buckets
    )
    payload["dataset"] = dataset.name
    payload["detector"] = "frcnn"
    payload["num_images"] = num_images
    payload["elapsed_sec"] = round(time.perf_counter() - t0, 3)
    payload["checkpoint"] = extractor.checkpoint_path

    out_path = out_dir / f"{split}.pt"
    torch.save(payload, out_path)
    print(
        f"  wrote {out_path}  dets={payload['num_detections']}  "
        f"classes={len(payload['by_class'])}  wall={payload['elapsed_sec']:.1f}s",
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("/content/spk"))
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="voc", help="voc or bdd")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to extract (default: all for --dataset)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dir (default: <root>/data/frcnn/<dataset>/logits)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--decode-batch",
        type=int,
        default=DECODE_BATCH,
        help="Tar decode batch size; FRCNN still infers one image at a time (default: 8)",
    )
    parser.add_argument("--max-images", type=int, default=0, help="0 = all images")
    args = parser.parse_args()
    dataset = DATASETS[args.dataset]
    splits = args.splits or list(dataset.split_specs)
    unknown = [s for s in splits if s not in dataset.split_specs]
    if unknown:
        raise SystemExit(
            f"unknown --splits {unknown} for --dataset {args.dataset} "
            f"(choose from {list(dataset.split_specs)})"
        )

    from frcnn_fx.bootstrap import config_path as frcnn_config_path

    checkpoint = args.root / "model/frcnn" / dataset.checkpoint
    cfg_path = frcnn_config_path(args.root, dataset.name)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    out_dir = args.out or (args.root / "data/frcnn" / dataset.name / "logits")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"frcnn_logits_extraction  dataset={dataset.name}  root={args.root}\n"
        f"  checkpoint={checkpoint}\n"
        f"  config={cfg_path}\n"
        f"  out={out_dir}\n"
        f"  splits={splits}\n"
        f"  device={args.device}  decode_batch={args.decode_batch}",
        flush=True,
    )

    extractor = FRCNNLogitsExtractor(
        checkpoint,
        cfg_path,
        args.device,
        detector_class_names=dataset.detector_class_names,
        label_aliases=dataset.label_aliases,
        eval_classes=dataset.eval_classes,
    )
    print(
        f"  model classes ({len(extractor.class_names)}): {extractor.class_names}\n"
        f"  background_index={extractor.background_index}",
        flush=True,
    )

    summary = {}
    for split in splits:
        summary[split] = {
            "num_detections": extract_split(
                split,
                args.root,
                dataset,
                extractor,
                out_dir,
                args.decode_batch,
                args.max_images,
            )["num_detections"]
        }
    print("\n=== done ===", flush=True)
    for split, info in summary.items():
        print(f"  {split}: {info['num_detections']} detections -> {out_dir}/{split}.pt", flush=True)


if __name__ == "__main__":
    main()
