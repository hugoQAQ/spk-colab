#!/usr/bin/env python3
"""Extract YOLO classification logits (one dict per split, keyed by class).

Self-contained: no import from run.py. Reads image tars, runs YOLO forward,
applies the same conf/NMS/max_det settings as run.py, and saves per-split
artifacts under --out.

Each split file (e.g. voc-train.pt / bdd-train.pt) is a dict:
  class_names  list[str]          model class order (length C)
  by_class     dict[str, dict]    predicted class -> rows for that class
    <class>:
      logits         float32 [N, C]   pre-sigmoid class logits (logit of sigmoid score)
      pred_labels    int64   [N]      argmax column index (= model class id)
      confidences    float32 [N]      max class score after sigmoid
      file_names     list[str] [N]
      bboxes_xyxy    float32 [N, 4]    original-image coordinates
      detection_ids  list[str] [N]

Also includes flat concatenation across classes:
  logits, pred_labels, confidences, file_names, bboxes_xyxy, detection_ids

Examples
--------
    # VOC
    bash mount_logits_voc.sh
    python logits_extraction.py --dataset voc --root /content/spk

    # BDD
    bash mount_logits_bdd.sh
    python logits_extraction.py --dataset bdd --root /content/spk
    python logits_extraction.py --dataset bdd --splits bdd-train bdd-val --max-images 200
"""
from __future__ import annotations

import argparse
import tarfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from tqdm.auto import tqdm

# Match run.py detector settings.
CONF = 0.25
IOU = 0.7
IMGSZ = 640
MAX_DET = 30
EXTRACT_BATCH = 16
VRAM_FRACTION = 0.70
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

VOC14 = frozenset({
    "bird", "bottle", "car", "cat", "cow", "dog", "horse", "person", "sheep",
    "bicycle", "boat", "bus", "chair", "train",
})

BDD10 = frozenset({
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
    "traffic_light", "traffic_sign",
})


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    checkpoint: str
    split_specs: dict[str, dict[str, str]]
    label_aliases: dict[str, str] = field(default_factory=dict)
    eval_classes: frozenset[str] | None = None


DATASETS: dict[str, DatasetConfig] = {
    "voc": DatasetConfig(
        name="voc",
        checkpoint="voc_vanilla.pt",
        split_specs={
            "voc-train": {"prefix": "voc_yolo_train", "search": "id"},
            "voc-val": {"prefix": "voc_yolo_val", "search": "id"},
            "near-ood": {"prefix": "near_ood_voc", "search": "ood"},
            "far-ood": {"prefix": "far_ood", "search": "ood"},
        },
        eval_classes=VOC14,
    ),
    "bdd": DatasetConfig(
        name="bdd",
        checkpoint="bdd_vanilla.pt",
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


def letterbox_rgb(image_rgb: np.ndarray) -> tuple[torch.Tensor, dict[str, Any]]:
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
    }


def xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    xyxy = xywh.clone()
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return xyxy


def scale_boxes_to_orig(boxes_xyxy: torch.Tensor, info: dict[str, Any]) -> torch.Tensor:
    """Map letterbox xyxy back to original image coords (handles shape [4] or [N, 4])."""
    squeeze = boxes_xyxy.ndim == 1
    boxes = boxes_xyxy.clone().float().reshape(-1, 4)
    ratio, pad_l, pad_t = info["ratio"], info["pad_l"], info["pad_t"]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_l) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_t) / ratio
    return boxes.squeeze(0) if squeeze else boxes


def _cuda_index(device: str | torch.device) -> int | None:
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


def to_pre_sigmoid(scores: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert YOLO sigmoid class scores to pre-sigmoid logits."""
    s = scores.float().clamp(eps, 1.0 - eps)
    return torch.log(s / (1.0 - s))


def canonical_class(raw_label: str, aliases: dict[str, str]) -> str:
    text = str(raw_label).strip()
    lowered = text.lower()
    return aliases.get(lowered, aliases.get(text, text))


class YOLOLogitsExtractor:
    def __init__(
        self,
        checkpoint: Path,
        device: str | torch.device,
        *,
        label_aliases: dict[str, str] | None = None,
        eval_classes: frozenset[str] | None = None,
    ) -> None:
        from ultralytics import YOLO

        wrapper = YOLO(str(checkpoint))
        self.names = wrapper.names
        self.class_names = [str(self.names[i]) for i in range(len(self.names))]
        self.label_aliases = label_aliases or {}
        self.eval_classes = eval_classes
        self.model = wrapper.model.to(device).eval()
        self.device = torch.device(device)
        self.detect_input_indices = (16, 19, 22)
        self.detect_layer_index = 23
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def _forward(self, batch_tensor: torch.Tensor) -> torch.Tensor:
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
        return preds.float()

    def _nms_with_logits(self, pred: torch.Tensor) -> list[dict[str, Any]]:
        """pred: (4+nc, N_anchors) letterbox xywh + sigmoid cls scores (run.py layout)."""
        if pred.ndim != 2:
            raise ValueError(f"expected (4+nc, N), got {tuple(pred.shape)}")
        boxes_xywh = pred[:4]
        cls_scores = pred[4:]
        scores, labels = cls_scores.max(dim=0)
        keep = scores > CONF
        if not keep.any():
            return []
        xyxy = xywh_to_xyxy(boxes_xywh.T[keep])
        scores = scores[keep]
        labels = labels[keep]
        cls_kept = cls_scores[:, keep].T
        class_offset = 7680.0
        keep_idx = nms(xyxy + labels.unsqueeze(1).to(xyxy.dtype) * class_offset, scores, IOU)
        keep_idx = keep_idx[:MAX_DET]
        detections: list[dict[str, Any]] = []
        for idx in keep_idx.tolist():
            label = int(labels[idx].item())
            raw_label = self.class_names[label]
            pred_class = canonical_class(raw_label, self.label_aliases)
            if self.eval_classes is not None and pred_class not in self.eval_classes:
                continue
            detections.append({
                "bbox_xyxy": xyxy[idx].detach().cpu(),
                "confidence": float(scores[idx].item()),
                "pred_label": label,
                "pred_class": pred_class,
                "detector_label": raw_label,
                "logits": to_pre_sigmoid(cls_kept[idx]).detach().cpu(),
            })
        return detections

    @torch.inference_mode()
    def infer_batch_bgr(self, images_bgr: list[np.ndarray]) -> list[list[dict[str, Any]]]:
        if not images_bgr:
            return []
        tensors, infos = [], []
        for image_bgr in images_bgr:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            tensor, info = letterbox_rgb(rgb)
            tensors.append(tensor)
            infos.append(info)
        batch_tensor = torch.cat(tensors, dim=0)
        preds = self._forward(batch_tensor)
        if preds.ndim != 3:
            raise ValueError(f"unexpected pred shape {tuple(preds.shape)}")
        if preds.shape[1] > preds.shape[2]:
            preds = preds.transpose(1, 2)
        results: list[list[dict[str, Any]]] = []
        for batch_idx, info in enumerate(infos):
            dets = self._nms_with_logits(preds[batch_idx])
            for det in dets:
                det["bbox_xyxy"] = scale_boxes_to_orig(det["bbox_xyxy"], info)
            results.append(dets)
        return results


def tune_extract_batch(extractor: YOLOLogitsExtractor, device: str, vram_frac: float, manual: int) -> int:
    if manual > 0:
        print(f"  extract batch: {manual} (manual)", flush=True)
        return manual
    dummy = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)

    def trial(batch: int) -> None:
        extractor.infer_batch_bgr([dummy] * batch)

    chosen = choose_batch_size(
        trial, device, vram_frac,
        min_batch=1, max_batch=256, default=EXTRACT_BATCH, label="extract batch",
    )
    if _cuda_index(device) is not None:
        torch.cuda.empty_cache()
    return max(chosen, 1)


def accumulate_rows(
    split: str,
    class_names: list[str],
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
    extractor: YOLOLogitsExtractor,
    out_dir: Path,
    batch_size: int,
    max_images: int,
) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    det_counter = 0
    t0 = time.perf_counter()

    with SplitImageSource(split, root, dataset, max_images=max_images) as source:
        num_images = source.count_images()
        with tqdm(total=num_images, desc=split, unit="img") as pbar:
            for batch_items in iter_image_batches(source, batch_size):
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

    payload = accumulate_rows(split, extractor.class_names, buckets)
    payload["dataset"] = dataset.name
    payload["num_images"] = num_images
    payload["elapsed_sec"] = round(time.perf_counter() - t0, 3)
    payload["checkpoint"] = str(extractor.model.__class__.__name__)

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
        "--out", type=Path, default=None,
        help="Output dir (default: <root>/data/yolo/<dataset>/logits)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--batch-size", type=int, default=0,
        help="Image batch size; 0 = auto from --vram-frac (default)",
    )
    parser.add_argument(
        "--vram-frac", type=float, default=VRAM_FRACTION,
        help="Target fraction of GPU memory for auto batch size (default: 0.70)",
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

    checkpoint = args.root / "model/yolo" / dataset.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    out_dir = args.out or (args.root / "data/yolo" / dataset.name / "logits")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"logits_extraction  dataset={dataset.name}  root={args.root}\n"
        f"  checkpoint={checkpoint}\n"
        f"  out={out_dir}\n"
        f"  splits={splits}\n"
        f"  device={args.device}  vram_frac={args.vram_frac:.0%}",
        flush=True,
    )

    extractor = YOLOLogitsExtractor(
        checkpoint,
        args.device,
        label_aliases=dataset.label_aliases,
        eval_classes=dataset.eval_classes,
    )
    print(f"  model classes ({len(extractor.class_names)}): {extractor.class_names}", flush=True)
    batch_size = tune_extract_batch(extractor, args.device, args.vram_frac, args.batch_size)

    summary = {}
    for split in splits:
        summary[split] = {
            "num_detections": extract_split(
                split, args.root, dataset, extractor, out_dir, batch_size, args.max_images
            )["num_detections"]
        }
    print("\n=== done ===", flush=True)
    for split, info in summary.items():
        print(f"  {split}: {info['num_detections']} detections -> {out_dir}/{split}.pt", flush=True)


if __name__ == "__main__":
    main()
