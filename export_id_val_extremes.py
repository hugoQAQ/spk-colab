#!/usr/bin/env python3
"""Export ID-val detections at the high and low end of the OOD score, split by TP/FP.

Isolation Forest `decision_function` is high when a box looks in-distribution.
This script reports `ood_score = -iforest_score`, so a high OOD score is a box
the SPK model treats as OOD (below the per-class ID-val 5th-percentile threshold
used by run.py FPR95).

TP: IoU >= 0.5 with a same-class ground-truth box in the ID-val label tar.
FP: every other ID-val detection of that predicted class.

    python export_id_val_extremes.py \\
        --root /content/spk --detector yolo --dataset voc --seed 42 \\
        --per-group 8

Writes under <root>/diagnostics/id_val_extremes/<method>/:
  cases.json     activation vector, OOD score, TP/FP, image path
  images/        one boxed frame per selected detection
"""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

import run as pipeline
from export_class_diagnostics import (
    METHOD_SPECS,
    TarImageLoader,
    _json_value,
    _sanitize_name,
    attach_concepts_to_cases,
    resolve_bundle_root,
    row_match_key,
    score_rows,
    try_fit_class_iforest,
)
from run import (
    DATASET_DIR,
    IMAGE_SUFFIXES,
    KNN_COL,
    SPK4_COLS,
    _boxes_from_yolo_labels,
    canonical_file_name,
    iou,
    keep_true_positives,
    seed_dir,
    set_profile,
    set_root,
)

GROUPS = ("tp_high_ood", "tp_low_ood", "fp_high_ood", "fp_low_ood")


def val_tar_split() -> str:
    for split, filename in pipeline.PROFILE.split_output_names.items():
        if Path(filename).stem == "id_val":
            return split
    raise SystemExit(f"no id_val split on profile {pipeline.PROFILE.name}")


def load_id_val_ground_truth() -> dict[str, list[dict]]:
    """YOLO labels inside the ID-val tar (gt_*.json only indexes ID-train)."""
    split = val_tar_split()
    prefix = pipeline.PROFILE.split_tar_prefix[split]
    paths = sorted(DATASET_DIR.glob(f"{prefix}-*.tar"))
    if not paths:
        raise SystemExit(f"no {prefix}-*.tar under {DATASET_DIR} (mount with --stage-c)")

    index: dict[str, list[dict]] = {}
    class_names = pipeline.PROFILE.label_classes
    fixed_size = pipeline.PROFILE.gt_fixed_size
    for path in paths:
        with tarfile.open(path) as archive:
            members = [m for m in archive.getmembers() if m.isfile()]
            if fixed_size is not None:
                width, height = fixed_size
                labels = [m for m in members if Path(m.name).suffix.lower() == ".txt"]
                for member in tqdm(labels, desc=f"id-val gt {path.name}"):
                    handle = archive.extractfile(member)
                    if handle is None:
                        continue
                    stem = Path(member.name).stem
                    index[stem] = _boxes_from_yolo_labels(handle.read().decode(), class_names, width, height)
                continue
            images = {Path(m.name).stem: m for m in members if Path(m.name).suffix.lower() in IMAGE_SUFFIXES}
            labels = {Path(m.name).stem: m for m in members if Path(m.name).suffix.lower() == ".txt"}
            for stem, member in tqdm(sorted(labels.items()), desc=f"id-val gt {path.name}"):
                image_member = images.get(stem)
                if image_member is None:
                    continue
                image_handle = archive.extractfile(image_member)
                label_handle = archive.extractfile(member)
                if image_handle is None or label_handle is None:
                    continue
                with Image.open(image_handle) as image:
                    width, height = image.size
                index[stem] = _boxes_from_yolo_labels(label_handle.read().decode(), class_names, width, height)
    print(f"  id-val GT: {len(index):,} images from {len(paths)} tar(s)", flush=True)
    return index


def match_iou(row: pd.Series, ground_truth: dict[str, list[dict]]) -> float:
    raw = str(row["file_name"])
    candidates = (
        ground_truth.get(Path(canonical_file_name(raw)).stem)
        or ground_truth.get(Path(raw).stem)
        or []
    )
    box = [float(row["bbox_x1"]), float(row["bbox_y1"]), float(row["bbox_x2"]), float(row["bbox_y2"])]
    return max(
        (iou(box, gt["bbox_xyxy"]) for gt in candidates if gt["class"] == str(row["class"])),
        default=0.0,
    )


def activation_payload(row: pd.Series, feature_cols: list[str]) -> dict[str, Any]:
    names = [c for c in feature_cols if c in row.index]
    values = [_json_value(row[c]) for c in names]
    return {"names": names, "values": values, **{name: value for name, value in zip(names, values)}}


def pick_group(frame: pd.DataFrame, match: str, which: str, k: int) -> pd.DataFrame:
    part = frame.loc[frame["match"] == match]
    if part.empty or k <= 0:
        return part.iloc[0:0]
    # high_ood: largest ood_score (most OOD-like). low_ood: smallest.
    return part.sort_values("ood_score", ascending=(which == "low_ood")).head(k)


def select_extremes(scored: pd.DataFrame, class_names: list[str], k: int) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for class_name in class_names:
        part = scored.loc[scored["class"].astype(str) == class_name]
        for match, which, group in (
            ("tp", "high_ood", "tp_high_ood"),
            ("tp", "low_ood", "tp_low_ood"),
            ("fp", "high_ood", "fp_high_ood"),
            ("fp", "low_ood", "fp_low_ood"),
        ):
            chosen = pick_group(part, match, which, k).copy()
            if chosen.empty:
                continue
            chosen["group"] = group
            pieces.append(chosen)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def draw_box(bgr: np.ndarray, row: pd.Series) -> np.ndarray:
    img = bgr.copy()
    x1, y1, x2, y2 = map(int, [row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]])
    color = (0, 180, 0) if row["match"] == "tp" else (0, 0, 220)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = (
        f"{row['group']} {row['class']} ood={row['ood_score']:.3f} "
        f"IF={row['iforest_score']:.3f} iou={row['gt_iou']:.2f}"
    )
    cv2.putText(img, label[:110], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def case_record(row: pd.Series, feature_cols: list[str], image_rel: str | None, concepts: dict | None) -> dict:
    threshold = float(row["id_val_threshold_p5"])
    score = float(row["iforest_score"])
    return _json_value(
        {
            "case_id": row["case_id"],
            "group": row["group"],
            "match": row["match"],
            "class": str(row["class"]),
            "file_name": str(row["file_name"]),
            "bbox_xyxy": [float(row["bbox_x1"]), float(row["bbox_y1"]), float(row["bbox_x2"]), float(row["bbox_y2"])],
            "detector_confidence": float(row["detector_confidence"]),
            "gt_iou": float(row["gt_iou"]),
            "ood": {
                "ood_score": float(row["ood_score"]),
                "iforest_score": score,
                "id_val_threshold_p5": threshold,
                "considered_ood": bool(score < threshold),
                "margin_above_threshold": score - threshold,
                "note": (
                    "ood_score = -iforest_score. considered_ood means this ID-val box falls "
                    "below the class 5th-percentile IF threshold (the FPR95 operating point)."
                ),
            },
            "activation_vector": activation_payload(row, feature_cols),
            "concepts": concepts,
            "image_file": image_rel,
            "match_key": list(row_match_key(row)),
        }
    )


def export_method(
    activations: pd.DataFrame,
    id_train_tp: pd.DataFrame,
    val_gt: dict[str, list[dict]],
    class_names: list[str],
    method_slug: str,
    out_dir: Path,
    seed: int,
    per_group: int,
    device: str,
    attach_concepts: bool,
    skip_images: bool,
) -> dict[str, Any]:
    feature_cols = METHOD_SPECS[method_slug][1]
    method_dir = out_dir / method_slug
    method_dir.mkdir(parents=True, exist_ok=True)
    id_val = activations.loc[activations["data_source"] == "id_val"].copy()
    scored_parts: list[pd.DataFrame] = []
    thresholds: dict[str, Any] = {}
    for class_name in class_names:
        fitted = try_fit_class_iforest(activations, id_train_tp, class_name, feature_cols, seed)
        subset = id_val.loc[id_val["class"].astype(str) == class_name]
        if fitted is None or subset.empty:
            thresholds[class_name] = {"skipped": True, "n_id_val": int(len(subset))}
            continue
        forest, scaler, meta = fitted
        thresholds[class_name] = meta
        scored = score_rows(subset, forest, scaler, feature_cols, meta["id_val_threshold_p5"])
        scored["ood_score"] = -scored["iforest_score"]
        scored["gt_iou"] = [match_iou(row, val_gt) for _, row in scored.iterrows()]
        scored["match"] = np.where(scored["gt_iou"] >= 0.5, "tp", "fp")
        scored_parts.append(scored)
        n_tp = int((scored["match"] == "tp").sum())
        n_ood = int((scored["iforest_score"] < meta["id_val_threshold_p5"]).sum())
        print(
            f"  {class_name:12s} n={len(scored):4d} tp={n_tp:4d} fp={len(scored) - n_tp:4d} "
            f"below_p5={n_ood:4d} thr={meta['id_val_threshold_p5']:.3f}",
            flush=True,
        )

    scored_all = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    selected = select_extremes(scored_all, class_names, per_group) if not scored_all.empty else pd.DataFrame()
    if selected.empty:
        raise SystemExit(f"{method_slug}: no ID-val rows to export")

    selected = selected.reset_index(drop=True)
    selected["case_id"] = [
        f"{method_slug}_{_sanitize_name(row['class'])}_{row['group']}_{i:04d}"
        for i, row in selected.iterrows()
    ]

    entries = [{"row": row, "concepts": None, "image_relpath": None} for _, row in selected.iterrows()]
    if attach_concepts:
        attach_concepts_to_cases(entries, seed_dir(pipeline.ARCH_DIR / "concept_head_ood", seed), device)

    n_images = 0
    if not skip_images:
        images_dir = method_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        loader = TarImageLoader()
        try:
            for entry in entries:
                row = entry["row"]
                bgr = loader.load_bgr("id_val", str(row["file_name"]))
                if bgr is None:
                    continue
                rel = f"images/{row['case_id']}.jpg"
                cv2.imwrite(str(method_dir / rel), draw_box(bgr, row))
                entry["image_relpath"] = rel
                n_images += 1
        finally:
            loader.close()

    cases = [
        case_record(entry["row"], feature_cols, entry.get("image_relpath"), entry.get("concepts"))
        for entry in entries
    ]
    by_group: dict[str, list[dict]] = {name: [] for name in GROUPS}
    for case in cases:
        by_group.setdefault(str(case["group"]), []).append(case)

    payload = {
        "method_slug": method_slug,
        "method": METHOD_SPECS[method_slug][0],
        "feature_cols": feature_cols,
        "seed": seed,
        "per_group": per_group,
        "split": "id_val",
        "tp_rule": "IoU >= 0.5 with a same-class box in the ID-val YOLO labels",
        "ood_score": "negative Isolation Forest decision_function (higher = more OOD)",
        "considered_ood": "iforest_score < per-class id_val 5th percentile",
        "n_id_val_scored": int(len(scored_all)),
        "n_cases": len(cases),
        "counts": {name: len(items) for name, items in by_group.items()},
        "thresholds": thresholds,
        "cases": cases,
    }
    (method_dir / "cases.json").write_text(json.dumps(_json_value(payload), indent=2) + "\n")
    for name, items in by_group.items():
        (method_dir / f"{name}.json").write_text(
            json.dumps(_json_value({"group": name, "n_cases": len(items), "cases": items}), indent=2) + "\n"
        )
    keep_cols = [
        c
        for c in (
            "case_id", "group", "match", "class", "file_name", "gt_iou", "ood_score",
            "iforest_score", "id_val_threshold_p5", "detector_confidence", *SPK4_COLS, KNN_COL,
        )
        if c in selected.columns
    ]
    selected[keep_cols].to_csv(method_dir / "cases.csv", index=False)
    print(f"  wrote {method_dir / 'cases.json'} ({len(cases)} cases, {n_images} images)", flush=True)
    return {
        "method_slug": method_slug,
        "n_cases": len(cases),
        "n_images": n_images,
        "counts": payload["counts"],
        "output_dir": str(method_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--detector", choices=("yolo", "frcnn", "rtdetr"), default="yolo")
    parser.add_argument("--dataset", choices=("voc", "bdd"), default="voc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes", nargs="+", default=None, help="Subset of predicted classes (default: all scored)")
    parser.add_argument("--per-group", type=int, default=8, help="Examples per class for each of the 4 TP/FP x high/low groups")
    parser.add_argument("--methods", nargs="+", choices=tuple(METHOD_SPECS), default=["spk_local", "spk_full"])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attach-concepts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_profile(args.detector, args.dataset)
    bundle_root = resolve_bundle_root(args.root, pipeline.PROFILE.name)
    set_root(bundle_root)

    activations_path = seed_dir(pipeline.ARCH_DIR / "concept_head_ood", args.seed) / "activations.csv"
    if not activations_path.is_file():
        raise SystemExit(f"missing {activations_path}")
    activations = pd.read_csv(activations_path)
    id_train = activations.loc[activations["data_source"] == "id_train"]
    id_train_tp = keep_true_positives(id_train, pipeline.GT_INDEX)

    present = sorted(activations.loc[activations["data_source"] == "id_val", "class"].astype(str).unique())
    class_names = args.classes or present
    missing = sorted(set(class_names) - set(present))
    if missing:
        raise SystemExit(f"classes missing from id_val activations: {missing}")

    print("=== ID-val ground truth ===", flush=True)
    val_gt = load_id_val_ground_truth()
    out_dir = args.out_dir or (bundle_root / "diagnostics" / "id_val_extremes")
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for method_slug in args.methods:
        if method_slug == "spk_full" and KNN_COL not in activations.columns:
            print(f"skip {method_slug}: no {KNN_COL}", flush=True)
            continue
        print(f"\n=== {method_slug} ===", flush=True)
        reports.append(
            export_method(
                activations,
                id_train_tp,
                val_gt,
                class_names,
                method_slug,
                out_dir,
                args.seed,
                args.per_group,
                args.device,
                args.attach_concepts,
                args.skip_images,
            )
        )
    summary = {
        "detector": args.detector,
        "dataset": args.dataset,
        "seed": args.seed,
        "per_group": args.per_group,
        "activations_csv": str(activations_path),
        "methods": reports,
    }
    (out_dir / "summary.json").write_text(json.dumps(_json_value(summary), indent=2) + "\n")
    print(f"\nwrote {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
