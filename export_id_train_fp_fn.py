#!/usr/bin/env python3
"""Pack typical ID-train false positives and false negatives into one tar.

A detection is a true positive when its predicted class matches a ground-truth
box at IoU >= 0.5 (same rule as run.py keep_true_positives). Everything else
the detector emitted on ID-train is a false positive. A ground-truth box with
no same-class detection at IoU >= 0.5 is a false negative.

The semantic / concept head is currently fit only on those true positives.
This export is a small, class-balanced sample of the mistakes (confident
background FPs, class confusions, pure misses, and confused misses) plus the
JPEG crops and a JSON manifest, so you can download one archive and visualize
them locally.

    python export_id_train_fp_fn.py --root /content/spk --detector yolo --dataset voc
    # writes /content/spk/diagnostics/id_train_fp_fn.tar

Then locally:
    colab download -s SESSION /content/spk/diagnostics/id_train_fp_fn.tar ./id_train_fp_fn.tar
"""
from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

import run as pipeline
from run import (
    ROI_DIR,
    VocSplitImageSource,
    canonical_file_name,
    canonical_pred_class,
    iou,
    set_profile,
    set_root,
)

IOU_TP = 0.5
SPK_COLS = ("known_max", "unknown", "proxy_max", "relative_area", "native_knn")


def gt_index_path(root: Path, dataset_name: str) -> Path:
    return root / "data" / "id" / f"gt_{dataset_name}.json"


def resolve_bundle_root(root: Path | None, dataset_name: str) -> Path:
    if root is not None:
        chosen = root.expanduser().resolve()
        gt = gt_index_path(chosen, dataset_name)
        if not gt.is_file():
            raise SystemExit(f"--root {chosen} has no {gt}")
        return chosen
    for raw in (os.environ.get("SPK_ROOT", ""), "/content/spk", "/content/spk-colab"):
        if not raw.strip():
            continue
        cand = Path(raw).expanduser().resolve()
        if gt_index_path(cand, dataset_name).is_file():
            print(f"  bundle root: {cand}", flush=True)
            return cand
    raise SystemExit("pass --root /content/spk (directory that contains data/id/gt_*.json)")


def id_train_split() -> str:
    for split, name in pipeline.PROFILE.split_output_names.items():
        if name == "id_train.pt":
            return split
    raise SystemExit("profile has no id_train.pt split")


def gt_boxes_for(ground_truth: dict, file_name: str) -> list[dict]:
    raw = str(file_name)
    return (
        ground_truth.get(Path(canonical_file_name(raw)).stem)
        or ground_truth.get(Path(raw).stem)
        or []
    )


def box_of(row: dict) -> list[float]:
    return [float(v) for v in row["bbox_xyxy"]]


def best_match(box: list[float], candidates: list[dict], *, same_class: str | None) -> dict | None:
    best: dict | None = None
    for cand in candidates:
        if same_class is not None and cand["class"] != same_class:
            continue
        score = iou(box, cand["bbox_xyxy"] if "bbox_xyxy" in cand else box_of(cand))
        if best is None or score > best["iou"]:
            best = {
                "class": cand["class"],
                "iou": float(score),
                "bbox_xyxy": [float(v) for v in (cand["bbox_xyxy"] if "bbox_xyxy" in cand else box_of(cand))],
            }
    return best


def load_detections(roi_path: Path) -> list[dict]:
    print(f"  loading ROI metadata {roi_path}", flush=True)
    cache = torch.load(roi_path, map_location="cpu", weights_only=False)
    metadata = cache["metadata"]
    del cache
    rows = []
    for row in metadata:
        pred = canonical_pred_class(row)
        if pred is None:
            continue
        x1, y1, x2, y2 = box_of(row)
        rows.append(
            {
                "class": pred,
                "file_name": row["file_name"],
                "bbox_xyxy": [x1, y1, x2, y2],
                "detector_confidence": float(row["detector_confidence"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "image_path": row.get("image_path", ""),
            }
        )
    print(f"  {len(rows):,} ID-train detections", flush=True)
    return rows


def attach_activations(rows: list[dict], activations_csv: Path | None) -> None:
    if activations_csv is None or not activations_csv.is_file():
        return
    frame = pd.read_csv(activations_csv)
    frame = frame.loc[frame["data_source"].astype(str) == "id_train"]
    lookup: dict[tuple, dict] = {}
    for rec in frame.to_dict(orient="records"):
        key = (
            str(rec["file_name"]),
            str(rec["class"]),
            round(float(rec["bbox_x1"]), 2),
            round(float(rec["bbox_y1"]), 2),
            round(float(rec["bbox_x2"]), 2),
            round(float(rec["bbox_y2"]), 2),
        )
        lookup[key] = {col: (None if pd.isna(rec.get(col)) else float(rec[col])) for col in SPK_COLS if col in rec}
    hit = 0
    for row in rows:
        x1, y1, x2, y2 = row["bbox_xyxy"]
        key = (str(row["file_name"]), row["class"], round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2))
        acts = lookup.get(key)
        row["activations"] = acts
        if acts:
            hit += 1
    print(f"  joined SPK activations on {hit:,}/{len(rows):,} detections", flush=True)


def label_detections(detections: list[dict], ground_truth: dict) -> tuple[list[dict], list[dict]]:
    by_image: dict[str, list[dict]] = {}
    for det in detections:
        by_image.setdefault(str(det["file_name"]), []).append(det)

    fps: list[dict] = []
    for det in detections:
        gts = gt_boxes_for(ground_truth, det["file_name"])
        same = best_match(det["bbox_xyxy"], gts, same_class=det["class"])
        any_gt = best_match(det["bbox_xyxy"], gts, same_class=None)
        same_iou = 0.0 if same is None else same["iou"]
        if same_iou >= IOU_TP:
            continue
        if any_gt is not None and any_gt["class"] != det["class"] and any_gt["iou"] >= IOU_TP:
            subtype = "confusion"
        elif any_gt is None or any_gt["iou"] < 0.1:
            subtype = "background"
        else:
            subtype = "low_overlap"
        fps.append(
            {
                "kind": "fp",
                "subtype": subtype,
                "class": det["class"],
                "file_name": det["file_name"],
                "bbox_xyxy": det["bbox_xyxy"],
                "detector_confidence": det["detector_confidence"],
                "iou_same_class": same_iou,
                "best_gt": any_gt,
                "activations": det.get("activations"),
                "image_path": det.get("image_path", ""),
                "width": det["width"],
                "height": det["height"],
            }
        )

    fns: list[dict] = []
    seen_images = set(by_image)
    # GT index is keyed by stem; detections use file_name. Walk every GT image
    # that also appears in the detection cache, plus stems we can map.
    stem_to_file = {}
    for file_name in seen_images:
        stem_to_file[Path(canonical_file_name(file_name)).stem] = file_name
        stem_to_file[Path(file_name).stem] = file_name

    for stem, gts in ground_truth.items():
        file_name = stem_to_file.get(stem)
        if file_name is None:
            continue
        dets = by_image.get(file_name, [])
        det_cands = [
            {"class": d["class"], "bbox_xyxy": d["bbox_xyxy"], "detector_confidence": d["detector_confidence"], "activations": d.get("activations")}
            for d in dets
        ]
        for gt in gts:
            gt_box = [float(v) for v in gt["bbox_xyxy"]]
            same = best_match(gt_box, det_cands, same_class=gt["class"])
            any_det = best_match(gt_box, det_cands, same_class=None)
            same_iou = 0.0 if same is None else same["iou"]
            if same_iou >= IOU_TP:
                continue
            if any_det is not None and any_det["class"] != gt["class"] and any_det["iou"] >= IOU_TP:
                subtype = "confused"
            elif any_det is None or any_det["iou"] < 0.1:
                subtype = "missed"
            else:
                subtype = "partial"
            area = max(0.0, gt_box[2] - gt_box[0]) * max(0.0, gt_box[3] - gt_box[1])
            fns.append(
                {
                    "kind": "fn",
                    "subtype": subtype,
                    "class": gt["class"],
                    "file_name": file_name,
                    "bbox_xyxy": gt_box,
                    "detector_confidence": None,
                    "iou_same_class": same_iou,
                    "best_detection": any_det,
                    "box_area": area,
                    "activations": None,
                }
            )
    print(f"  false positives: {len(fps):,}   false negatives: {len(fns):,}", flush=True)
    return fps, fns


def _pick_bucket(rows: list[dict], k: int, rng: np.random.Generator) -> list[dict]:
    if len(rows) <= k:
        return list(rows)
    order = rng.permutation(len(rows))
    return [rows[i] for i in order[:k]]


def sample_typical(rows: list[dict], per_class: int, seed: int, *, rank_key: str) -> list[dict]:
    """Take a mix of subtypes per class, preferring high-confidence FPs / large FNs."""
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[dict]] = {}
    for row in rows:
        by_class.setdefault(row["class"], []).append(row)

    chosen: list[dict] = []
    for class_name, group in sorted(by_class.items()):
        subtypes: dict[str, list[dict]] = {}
        for row in group:
            subtypes.setdefault(row["subtype"], []).append(row)
        for bucket in subtypes.values():
            bucket.sort(key=lambda r: float(r.get(rank_key) or 0.0), reverse=True)
        names = list(subtypes)
        if not names:
            continue
        quota = {name: per_class // len(names) for name in names}
        for name in names[: per_class % len(names)]:
            quota[name] += 1
        picked: list[dict] = []
        for name, n in quota.items():
            # Top half by rank, then a random draw from the rest so "typical" is not only the extreme tail.
            bucket = subtypes[name]
            head_n = max(1, n // 2) if n else 0
            picked.extend(bucket[:head_n])
            rest = bucket[head_n:]
            picked.extend(_pick_bucket(rest, max(0, n - head_n), rng))
        # If a subtype was empty, fill from the remaining ranked rows.
        if len(picked) < per_class:
            used = {id(r) for r in picked}
            for row in sorted(group, key=lambda r: float(r.get(rank_key) or 0.0), reverse=True):
                if id(row) in used:
                    continue
                picked.append(row)
                if len(picked) >= per_class:
                    break
        chosen.extend(picked[:per_class])
    return chosen


def crop_with_box(bgr: np.ndarray, box: list[float], color: tuple[int, int, int], label: str) -> np.ndarray:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pad = 0.35 * max(bw, bh)
    cx1, cy1 = int(max(0, np.floor(x1 - pad))), int(max(0, np.floor(y1 - pad)))
    cx2, cy2 = int(min(w, np.ceil(x2 + pad))), int(min(h, np.ceil(y2 + pad)))
    crop = bgr[cy1:cy2, cx1:cx2].copy()
    if crop.size == 0:
        crop = bgr.copy()
        cx1 = cy1 = 0
    rx1, ry1 = int(round(x1 - cx1)), int(round(y1 - cy1))
    rx2, ry2 = int(round(x2 - cx1)), int(round(y2 - cy1))
    cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), color, 2)
    cv2.putText(crop, label[:80], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return crop


class ImageCache:
    def __init__(self, split: str) -> None:
        self.source = VocSplitImageSource(split)
        self.lookup: dict[str, pipeline.ImageItem] = {}
        for item in self.source.iter_items():
            self.lookup[item.file_name] = item
            self.lookup.setdefault(Path(item.file_name).stem, item)

    def close(self) -> None:
        self.source.close()

    def load(self, file_name: str) -> np.ndarray | None:
        item = self.lookup.get(file_name) or self.lookup.get(Path(file_name).stem)
        if item is None:
            return None
        return self.source.load_bgr(item)


def pack_tar(cases: list[dict], split: str, out_path: Path, meta: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images = ImageCache(split)
    written = 0
    try:
        with tarfile.open(out_path, "w") as archive:
            manifest_cases = []
            for i, case in enumerate(cases):
                bgr = images.load(str(case["file_name"]))
                rel = f"{case['kind']}/{case['subtype']}/{case['class']}/{i:04d}.jpg"
                case_out = {k: v for k, v in case.items() if k not in {"width", "height"}}
                case_out["image"] = rel if bgr is not None else None
                manifest_cases.append(case_out)
                if bgr is None:
                    continue
                color = (0, 0, 255) if case["kind"] == "fp" else (255, 128, 0)
                label = f"{case['kind']} {case['subtype']} {case['class']}"
                crop = crop_with_box(bgr, case["bbox_xyxy"], color, label)
                ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if not ok:
                    continue
                payload = encoded.tobytes()
                info = tarfile.TarInfo(name=rel)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
                written += 1
            body = json.dumps({"meta": meta, "cases": manifest_cases}, indent=2).encode()
            info = tarfile.TarInfo(name="manifest.json")
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
    finally:
        images.close()
    print(f"wrote {out_path} ({written} images, {len(cases)} cases)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--detector", choices=("yolo", "frcnn", "rtdetr"), default="yolo")
    parser.add_argument("--dataset", choices=("voc", "bdd"), default="voc")
    parser.add_argument("--classes", nargs="+", default=None, help="Limit to these class names")
    parser.add_argument("--per-class", type=int, default=6, help="FP and FN samples kept per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Tar path (default: <root>/diagnostics/id_train_fp_fn.tar)",
    )
    parser.add_argument(
        "--activations",
        type=Path,
        default=None,
        help="activations.csv to attach SPK scores (default: concept_head_ood/seed_42/activations.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_profile(args.detector, args.dataset)
    root = resolve_bundle_root(args.root, pipeline.PROFILE.name)
    set_root(root)

    gt_path = gt_index_path(root, pipeline.PROFILE.name)
    roi_path = ROI_DIR / "id_train.pt"
    if not roi_path.is_file():
        raise SystemExit(f"missing {roi_path} (mount roi/ or run stage B)")

    activations = args.activations
    if activations is None:
        activations = (
            root / "data" / args.detector / pipeline.PROFILE.name / "concept_head_ood" / "seed_42" / "activations.csv"
        )

    ground_truth = json.loads(gt_path.read_text())
    detections = load_detections(roi_path)
    attach_activations(detections, activations)
    fps, fns = label_detections(detections, ground_truth)

    if args.classes:
        wanted = set(args.classes)
        fps = [r for r in fps if r["class"] in wanted]
        fns = [r for r in fns if r["class"] in wanted]

    sampled_fp = sample_typical(fps, args.per_class, args.seed, rank_key="detector_confidence")
    sampled_fn = sample_typical(fns, args.per_class, args.seed + 1, rank_key="box_area")
    cases = sampled_fp + sampled_fn
    print(f"  sampled {len(sampled_fp)} FP + {len(sampled_fn)} FN", flush=True)

    out = args.out or (root / "diagnostics" / "id_train_fp_fn.tar")
    pack_tar(
        cases,
        id_train_split(),
        out,
        {
            "detector": args.detector,
            "dataset": args.dataset,
            "iou_true_positive": IOU_TP,
            "per_class": args.per_class,
            "n_fp_pool": len(fps),
            "n_fn_pool": len(fns),
            "n_fp_sampled": len(sampled_fp),
            "n_fn_sampled": len(sampled_fn),
            "definition": {
                "fp": "ID-train detection with no same-class GT at IoU>=0.5",
                "fn": "ID-train GT box with no same-class detection at IoU>=0.5",
                "fp_subtypes": "confusion | background | low_overlap",
                "fn_subtypes": "confused | missed | partial",
            },
        },
    )


if __name__ == "__main__":
    main()
