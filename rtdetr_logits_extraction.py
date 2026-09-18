#!/usr/bin/env python3
"""Extract RT-DETR classification logits (one dict per split, keyed by class).

Self-contained: no import from run.py. Reads image tars, runs Ultralytics RT-DETR,
keeps last-decoder-layer pre-sigmoid class logits, then applies the same
conf / max_det settings as run.py. RT-DETR is NMS-free (no IoU NMS).

Each split file (e.g. voc-train.pt / bdd-train.pt) is a dict:
  class_names  list[str]          model class order (length C)
  by_class     dict[str, dict]    predicted class -> rows for that class
    <class>:
      logits         float32 [N, C]   pre-sigmoid class logits
      pred_labels    int64   [N]      argmax column index (= model class id)
      confidences    float32 [N]      max class score after sigmoid
      file_names     list[str] [N]
      bboxes_xyxy    float32 [N, 4]    original-image coordinates
      detection_ids  list[str] [N]

Also includes flat concatenation across classes:
  logits, pred_labels, confidences, file_names, bboxes_xyxy, detection_ids

Examples
--------
    bash mount_rtdetr_logits.sh
    python rtdetr_logits_extraction.py --dataset voc --root /content/spk

    DATASET=bdd bash mount_rtdetr_logits.sh
    python rtdetr_logits_extraction.py --dataset bdd --root /content/spk
    python rtdetr_logits_extraction.py --dataset bdd --splits bdd-train bdd-val --max-images 200
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
from tqdm.auto import tqdm

# Match run.py detector settings. RT-DETR is NMS-free; IOU is unused.
CONF = 0.25
IMGSZ = 640
MAX_DET = 30
EXTRACT_BATCH = 8
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


def scale_fill_rgb(image_rgb: np.ndarray) -> tuple[torch.Tensor, dict[str, Any]]:
    """Ultralytics RT-DETR LetterBox(scale_fill=True): stretch to a square canvas."""
    h, w = image_rgb.shape[:2]
    resized = cv2.resize(image_rgb, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0)
    return tensor.unsqueeze(0), {"orig_hw": (h, w)}


def xywh_norm_to_xyxy_orig(xywh: torch.Tensor, orig_hw: tuple[int, int]) -> torch.Tensor:
    """Map RT-DETR cxcywh in [0, 1] of the square input to original-image xyxy."""
    squeeze = xywh.ndim == 1
    boxes = xywh.clone().float().reshape(-1, 4)
    orig_h, orig_w = orig_hw
    cx, cy, bw, bh = boxes.unbind(dim=1)
    x1 = (cx - bw / 2) * orig_w
    y1 = (cy - bh / 2) * orig_h
    x2 = (cx + bw / 2) * orig_w
    y2 = (cy + bh / 2) * orig_h
    xyxy = torch.stack([x1, y1, x2, y2], dim=1)
    return xyxy.squeeze(0) if squeeze else xyxy


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
    max_batch: int = 64,
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
    s = scores.float().clamp(eps, 1.0 - eps)
    return torch.log(s / (1.0 - s))


def canonical_class(raw_label: str, aliases: dict[str, str]) -> str:
    text = str(raw_label).strip()
    lowered = text.lower()
    return aliases.get(lowered, aliases.get(text, text))


def _last_decoder_layer(tensor: torch.Tensor) -> torch.Tensor:
    """(ndl, bs, nq, D) -> (bs, nq, D); already (bs, nq, D) is returned as-is."""
    if tensor.ndim == 4:
        return tensor[-1]
    if tensor.ndim == 3:
        return tensor
    raise ValueError(f"unexpected decoder tensor shape {tuple(tensor.shape)}")


class RTDETRLogitsExtractor:
    def __init__(
        self,
        checkpoint: Path,
        device: str | torch.device,
        *,
        label_aliases: dict[str, str] | None = None,
        eval_classes: frozenset[str] | None = None,
    ) -> None:
        from ultralytics import RTDETR

        wrapper = RTDETR(str(checkpoint))
        self.names = wrapper.names
        self.class_names = [str(self.names[i]) for i in range(len(self.names))]
        self.label_aliases = label_aliases or {}
        self.eval_classes = eval_classes
        self.checkpoint_path = str(checkpoint.resolve())
        self.model = wrapper.model.to(device).eval()
        self.device = torch.device(device)
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def _forward(self, batch_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return last-layer boxes (bs, nq, 4) cxcywh-in-[0,1] and class logits (bs, nq, C)."""
        use_amp = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
            out = self.model(batch_tensor.to(self.device))
        if isinstance(out, (list, tuple)) and len(out) >= 2 and isinstance(out[1], (list, tuple)):
            extra = out[1]
            boxes = _last_decoder_layer(extra[0]).float()
            logits = _last_decoder_layer(extra[1]).float()
            return boxes, logits
        preds = out[0] if isinstance(out, (list, tuple)) else out
        if preds.ndim != 3 or preds.shape[-1] < 4 + len(self.class_names):
            raise ValueError(f"unexpected RT-DETR output shape {tuple(getattr(preds, 'shape', ()))}")
        boxes = preds[..., :4].float()
        logits = to_pre_sigmoid(preds[..., 4 : 4 + len(self.class_names)].float())
        return boxes, logits

    def _decode_queries(
        self, boxes_xywh: torch.Tensor, logits: torch.Tensor, orig_hw: tuple[int, int]
    ) -> list[dict[str, Any]]:
        """boxes_xywh / logits: (nq, 4) and (nq, C) for one image."""
        scores = logits.sigmoid()
        conf, labels = scores.max(dim=-1)
        keep = conf > CONF
        if not keep.any():
            return []
        conf = conf[keep]
        labels = labels[keep]
        logits_kept = logits[keep]
        boxes_kept = boxes_xywh[keep]
        order = torch.argsort(conf, descending=True)[:MAX_DET]
        detections: list[dict[str, Any]] = []
        for idx in order.tolist():
            label = int(labels[idx].item())
            raw_label = self.class_names[label]
            pred_class = canonical_class(raw_label, self.label_aliases)
            if self.eval_classes is not None and pred_class not in self.eval_classes:
                continue
            detections.append({
                "bbox_xyxy": xywh_norm_to_xyxy_orig(boxes_kept[idx], orig_hw).detach().cpu(),
                "confidence": float(conf[idx].item()),
                "pred_label": label,
                "pred_class": pred_class,
                "detector_label": raw_label,
                "logits": logits_kept[idx].detach().cpu(),
            })
        return detections

    @torch.inference_mode()
    def infer_batch_bgr(self, images_bgr: list[np.ndarray]) -> list[list[dict[str, Any]]]:
        if not images_bgr:
            return []
        tensors, infos = [], []
        for image_bgr in images_bgr:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            tensor, info = scale_fill_rgb(rgb)
            tensors.append(tensor)
            infos.append(info)
        batch_tensor = torch.cat(tensors, dim=0)
        boxes, logits = self._forward(batch_tensor)
        if boxes.shape[0] != len(infos) or logits.shape[0] != len(infos):
            raise ValueError(
                f"batch mismatch: boxes {tuple(boxes.shape)} logits {tuple(logits.shape)} "
                f"images {len(infos)}"
            )
        results: list[list[dict[str, Any]]] = []
        for batch_idx, info in enumerate(infos):
            results.append(self._decode_queries(boxes[batch_idx], logits[batch_idx], info["orig_hw"]))
        return results


def tune_extract_batch(extractor: RTDETRLogitsExtractor, device: str, vram_frac: float, manual: int) -> int:
    if manual > 0:
        print(f"  extract batch: {manual} (manual)", flush=True)
        return manual
    dummy = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)

    def trial(batch: int) -> None:
        extractor.infer_batch_bgr([dummy] * batch)

    chosen = choose_batch_size(
        trial, device, vram_frac,
        min_batch=1, max_batch=32, default=EXTRACT_BATCH, label="extract batch",
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
    extractor: RTDETRLogitsExtractor,
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
    payload["detector"] = "rtdetr"
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
        "--out", type=Path, default=None,
        help="Output dir (default: <root>/data/rtdetr/<dataset>/logits)",
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
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip splits whose output .pt already exists (unless --force)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if output .pt exists",
    )
    args = parser.parse_args()
    dataset = DATASETS[args.dataset]
    splits = args.splits or list(dataset.split_specs)
    unknown = [s for s in splits if s not in dataset.split_specs]
    if unknown:
        raise SystemExit(
            f"unknown --splits {unknown} for --dataset {args.dataset} "
            f"(choose from {list(dataset.split_specs)})"
        )

    checkpoint = args.root / "model/rtdetr" / dataset.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    out_dir = args.out or (args.root / "data/rtdetr" / dataset.name / "logits")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"rtdetr_logits_extraction  dataset={dataset.name}  root={args.root}\n"
        f"  checkpoint={checkpoint}\n"
        f"  out={out_dir}\n"
        f"  splits={splits}\n"
        f"  device={args.device}  vram_frac={args.vram_frac:.0%}",
        flush=True,
    )

    extractor = RTDETRLogitsExtractor(
        checkpoint,
        args.device,
        label_aliases=dataset.label_aliases,
        eval_classes=dataset.eval_classes,
    )
    print(f"  model classes ({len(extractor.class_names)}): {extractor.class_names}", flush=True)
    batch_size = tune_extract_batch(extractor, args.device, args.vram_frac, args.batch_size)

    summary = {}
    for split in splits:
        out_path = out_dir / f"{split}.pt"
        if args.skip_existing and not args.force and out_path.is_file():
            cached = torch.load(out_path, map_location="cpu", weights_only=False)
            n = int(cached.get("num_detections", 0))
            print(f"  skip {split}: {out_path} exists (dets={n})", flush=True)
            summary[split] = {"num_detections": n, "skipped": True}
            continue
        summary[split] = {
            "num_detections": extract_split(
                split, args.root, dataset, extractor, out_dir, batch_size, args.max_images
            )["num_detections"],
            "skipped": False,
        }
    print("\n=== done ===", flush=True)
    for split, info in summary.items():
        tag = " (skipped)" if info.get("skipped") else ""
        print(
            f"  {split}: {info['num_detections']} detections -> {out_dir}/{split}.pt{tag}",
            flush=True,
        )


if __name__ == "__main__":
    main()
