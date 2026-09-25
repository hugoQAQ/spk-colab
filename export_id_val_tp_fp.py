#!/usr/bin/env python3
"""Collect ID-validation true positives and false positives for the failure-case figure.

A detection is a true positive when its predicted class matches an ID-val
ground-truth box at IoU >= 0.5. Every other ID-val detection of that class is
a false positive. Isolation Forest scores use the same fit as run.py evaluate
(outlier_mode=none): forest on ID-train true positives, threshold = id_val
5th percentile.

The full scored table is always written. Images are a seeded sample, stratified
across Isolation Forest score bins, so a later canvas can show the same five
channels as the OOD failure figure (known_max, proxy_max, unknown,
relative_area, native_knn).

On a Colab VM, after mount_data.sh --stage-c:

    python export_id_val_tp_fp.py --root /content/spk --detector yolo --dataset voc --seed 42

Writes <root>/diagnostics/id_val_tp_fp/ and, when Drive is mounted, copies
id_val_tp_fp.tgz to MyDrive/experiments/{detector}-{dataset}/diagnostics/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

import run as pipeline
from export_class_diagnostics import (
    METHOD_SPECS,
    TarImageLoader,
    _json_value,
    _sanitize_name,
    activation_dict,
    resolve_bundle_root,
    score_rows,
    try_fit_class_iforest,
)
from export_id_val_extremes import load_id_val_ground_truth, match_iou
from run import KNN_COL, keep_true_positives, seed_dir, set_profile, set_root

SCORE_COLS = ("known_max", "unknown", "proxy_max", "relative_area", "native_knn")


def drive_experiments_dir(detector: str, dataset: str) -> Path | None:
    name = f"{detector}-{dataset}"
    for base in (
        Path("/content/drive/MyDrive/experiments"),
        Path("/content/drive/My Drive/experiments"),
    ):
        path = base / name
        if path.is_dir():
            return path
    return None


def stratified_sample(frame: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Up to k rows, spread across Isolation Forest score bins."""
    finite = frame.loc[np.isfinite(frame["iforest_score"].to_numpy())]
    if finite.empty or k <= 0 or len(finite) <= k:
        return finite if not frame.empty else frame
    n_bins = min(5, k, len(finite))
    try:
        bins = pd.qcut(finite["iforest_score"], q=n_bins, duplicates="drop")
    except ValueError:
        bins = pd.Series(0, index=finite.index)
    frame = finite
    groups = [part for _, part in frame.groupby(bins, observed=True)]
    rng = np.random.default_rng(seed)
    base, extra = divmod(k, len(groups))
    picked: list[pd.DataFrame] = []
    for i, part in enumerate(groups):
        take = min(len(part), base + (1 if i < extra else 0))
        if take <= 0:
            continue
        choice = rng.choice(len(part), size=take, replace=False)
        picked.append(part.iloc[np.sort(choice)])
    if not picked:
        return frame.iloc[0:0]
    return pd.concat(picked, ignore_index=True)


def draw_box(bgr: np.ndarray, row: pd.Series) -> np.ndarray:
    img = bgr.copy()
    x1, y1, x2, y2 = map(int, [row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]])
    color = (0, 160, 0) if row["match"] == "tp" else (0, 0, 220)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = (
        f"{row['match']} {row['class']} IF={row['iforest_score']:.3f} "
        f"km={row.get('known_max', float('nan')):.2f} "
        f"px={row.get('proxy_max', float('nan')):.2f} "
        f"unk={row.get('unknown', float('nan')):.2f}"
    )
    cv2.putText(
        img, label[:110], (max(x1, 0), max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
    )
    return img


def case_record(row: pd.Series, feature_cols: list[str], image_rel: str | None) -> dict[str, Any]:
    score = float(row["iforest_score"])
    threshold = float(row["id_val_threshold_p5"])
    return _json_value(
        {
            "case_id": row["case_id"],
            "match": row["match"],
            "class": str(row["class"]),
            "data_source": "id_val",
            "file_name": str(row["file_name"]),
            "bbox_xyxy": [float(row["bbox_x1"]), float(row["bbox_y1"]), float(row["bbox_x2"]), float(row["bbox_y2"])],
            "detector_confidence": float(row["detector_confidence"]),
            "gt_iou": float(row["gt_iou"]),
            "activations": activation_dict(row, feature_cols),
            "feature_cols": list(feature_cols),
            "ood": {
                "iforest_score": score,
                "id_val_threshold_p5": threshold,
                "iforest_margin": float(row["iforest_margin"]),
                "accepted_as_id": bool(score >= threshold),
                "definition": (
                    "accepted_as_id: Isolation Forest decision_function >= per-class "
                    "id_val 5th percentile (same operating point as run.py evaluate)"
                ),
            },
            "image_file": image_rel,
        }
    )


def score_id_val(
    activations: pd.DataFrame,
    id_train_tp: pd.DataFrame,
    val_gt: dict[str, list[dict]],
    class_names: list[str],
    feature_cols: list[str],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    id_val = activations.loc[activations["data_source"] == "id_val"]
    parts: list[pd.DataFrame] = []
    per_class: dict[str, Any] = {}
    for class_name in class_names:
        fitted = try_fit_class_iforest(activations, id_train_tp, class_name, feature_cols, seed)
        subset = id_val.loc[id_val["class"].astype(str) == class_name]
        if fitted is None or subset.empty:
            per_class[class_name] = {"skipped": True, "n_id_val": int(len(subset))}
            continue
        forest, scaler, meta = fitted
        scored = score_rows(subset, forest, scaler, feature_cols, meta["id_val_threshold_p5"])
        scored["gt_iou"] = [match_iou(row, val_gt) for _, row in scored.iterrows()]
        scored["match"] = np.where(scored["gt_iou"] >= 0.5, "tp", "fp")
        scored["accepted_as_id"] = scored["iforest_score"] >= scored["id_val_threshold_p5"]
        n_tp = int((scored["match"] == "tp").sum())
        meta = {
            **meta,
            "n_tp": n_tp,
            "n_fp": int(len(scored) - n_tp),
            "n_tp_accepted_as_id": int(((scored["match"] == "tp") & scored["accepted_as_id"]).sum()),
            "n_fp_accepted_as_id": int(((scored["match"] == "fp") & scored["accepted_as_id"]).sum()),
        }
        per_class[class_name] = meta
        parts.append(scored)
        print(
            f"  {class_name:12s} n={len(scored):5d} tp={n_tp:5d} fp={len(scored) - n_tp:5d} "
            f"tp_kept={meta['n_tp_accepted_as_id']:5d} fp_kept={meta['n_fp_accepted_as_id']:5d}",
            flush=True,
        )
    scored_all = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return scored_all, per_class


def select_cases(scored: pd.DataFrame, class_names: list[str], per_class: int, seed: int) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for class_index, class_name in enumerate(class_names):
        part = scored.loc[scored["class"].astype(str) == class_name]
        for match_index, match in enumerate(("tp", "fp")):
            group = part.loc[part["match"] == match]
            chosen = stratified_sample(group, per_class, seed + 17 * class_index + match_index)
            if chosen.empty:
                continue
            chosen = chosen.copy()
            chosen["case_id"] = [
                f"{_sanitize_name(class_name)}_{match}_{i:04d}" for i in range(len(chosen))
            ]
            pieces.append(chosen)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def render_images(selected: pd.DataFrame, images_dir: Path) -> dict[str, str]:
    images_dir.mkdir(parents=True, exist_ok=True)
    loader = TarImageLoader()
    relpaths: dict[str, str] = {}
    try:
        for _, row in selected.iterrows():
            bgr = loader.load_bgr("id_val", str(row["file_name"]))
            if bgr is None:
                continue
            rel = f"images/{row['case_id']}.jpg"
            cv2.imwrite(str(images_dir.parent / rel), draw_box(bgr, row))
            relpaths[str(row["case_id"])] = rel
    finally:
        loader.close()
    return relpaths


def write_method(
    method_dir: Path,
    method_slug: str,
    feature_cols: list[str],
    seed: int,
    per_class_n: int,
    per_class_meta: dict[str, Any],
    scored: pd.DataFrame,
    selected: pd.DataFrame,
    relpaths: dict[str, str],
) -> dict[str, Any]:
    method_dir.mkdir(parents=True, exist_ok=True)
    keep = [
        c
        for c in (
            "class", "match", "gt_iou", "file_name", "image_path",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "detector_confidence",
            *SCORE_COLS, "iforest_score", "id_val_threshold_p5", "iforest_margin", "accepted_as_id",
        )
        if c in scored.columns
    ]
    scored[keep].to_csv(method_dir / "all_scored_activations.csv", index=False)
    (method_dir / "thresholds.json").write_text(
        json.dumps(
            _json_value(
                {
                    "method_slug": method_slug,
                    "method": METHOD_SPECS[method_slug][0],
                    "feature_cols": feature_cols,
                    "seed": seed,
                    "tp_rule": "IoU >= 0.5 with a same-class box in the ID-val YOLO labels",
                    "per_class": per_class_meta,
                }
            ),
            indent=2,
        )
        + "\n"
    )

    cases = [
        case_record(row, feature_cols, relpaths.get(str(row["case_id"])))
        for _, row in selected.iterrows()
    ]
    for match in ("tp", "fp"):
        by_class: dict[str, list[dict]] = {}
        for case in cases:
            if case["match"] != match:
                continue
            by_class.setdefault(str(case["class"]), []).append(case)
        out = method_dir / f"{match}_by_class"
        out.mkdir(exist_ok=True)
        for class_name, items in sorted(by_class.items()):
            (out / f"{_sanitize_name(class_name)}.json").write_text(
                json.dumps(
                    _json_value({"class": class_name, "match": match, "n_cases": len(items), "cases": items}),
                    indent=2,
                )
                + "\n"
            )
    n_images = sum(1 for case in cases if case["image_file"])
    print(f"  {method_slug}: {len(scored)} scored rows, {len(cases)} sampled cases, {n_images} images", flush=True)
    return {
        "method_slug": method_slug,
        "n_scored": int(len(scored)),
        "n_tp": int((scored["match"] == "tp").sum()) if not scored.empty else 0,
        "n_fp": int((scored["match"] == "fp").sum()) if not scored.empty else 0,
        "n_sampled": len(cases),
        "n_images": n_images,
        "per_class": per_class_n,
        "output_dir": str(method_dir),
    }


def publish_tarball(out_dir: Path, detector: str, dataset: str, drive_dir: Path | None) -> str | None:
    archive = out_dir.parent / "id_val_tp_fp.tgz"
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)
    print(f"  packed {archive} ({archive.stat().st_size / 1e6:.1f} MB)", flush=True)

    dest_root = drive_dir
    if dest_root is None:
        experiments = drive_experiments_dir(detector, dataset)
        dest_root = experiments / "diagnostics" if experiments is not None else None
    if dest_root is None:
        print("  Drive experiments dir not mounted; left the archive on local disk", flush=True)
        return None
    dest_root.mkdir(parents=True, exist_ok=True)
    landed = dest_root / archive.name
    shutil.copy2(archive, landed)
    print(f"  copied to {landed}", flush=True)
    return str(landed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--detector", choices=("yolo", "frcnn", "rtdetr"), default="yolo")
    parser.add_argument("--dataset", choices=("voc", "bdd"), default="voc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument(
        "--per-class",
        type=int,
        default=20,
        help="Images per class for TP and for FP. 0 keeps every detection.",
    )
    parser.add_argument("--methods", nargs="+", choices=tuple(METHOD_SPECS), default=["spk_full"])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--drive-dir", type=Path, default=None, help="Override Drive diagnostics directory")
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-drive", action="store_true")
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
    out_dir = args.out_dir or (bundle_root / "diagnostics" / "id_val_tp_fp")
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for method_slug in args.methods:
        feature_cols = METHOD_SPECS[method_slug][1]
        if method_slug == "spk_full" and KNN_COL not in activations.columns:
            print(f"skip {method_slug}: no {KNN_COL}", flush=True)
            continue
        print(f"\n=== {method_slug} ===", flush=True)
        scored, per_class_meta = score_id_val(
            activations, id_train_tp, val_gt, class_names, feature_cols, args.seed
        )
        if scored.empty:
            raise SystemExit(f"{method_slug}: no ID-val rows scored")
        limit = len(scored) if args.per_class <= 0 else args.per_class
        selected = select_cases(scored, class_names, limit, args.seed)
        selected.insert(0, "method_slug", method_slug)
        selected["case_id"] = method_slug + "_" + selected["case_id"].astype(str)
        relpaths: dict[str, str] = {}
        if not args.skip_images:
            relpaths = render_images(selected, out_dir / method_slug / "images")
        reports.append(
            write_method(
                out_dir / method_slug,
                method_slug,
                feature_cols,
                args.seed,
                args.per_class,
                per_class_meta,
                scored,
                selected,
                relpaths,
            )
        )

    summary = {
        "detector": args.detector,
        "dataset": args.dataset,
        "seed": args.seed,
        "per_class": args.per_class,
        "tp_rule": "IoU >= 0.5 with a same-class box in the ID-val YOLO labels",
        "activations_csv": str(activations_path),
        "methods": reports,
    }
    (out_dir / "summary.json").write_text(json.dumps(_json_value(summary), indent=2) + "\n")
    print(f"\nwrote {out_dir / 'summary.json'}", flush=True)
    if not args.skip_drive:
        publish_tarball(out_dir, args.detector, args.dataset, args.drive_dir)


if __name__ == "__main__":
    main()
