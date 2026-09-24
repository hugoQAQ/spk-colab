#!/usr/bin/env python3
"""Export concept-head diagnostics and OOD false-positive (accepted FP) cases.

Single class (e.g. bird)::

    python export_class_diagnostics.py \\
        --root /content/spk --detector yolo --dataset voc \\
        --class-name bird --seed 42

All SPK local + SPK full OOD false alarms (IF score >= id_val p5 threshold)::

    python export_class_diagnostics.py \\
        --root /content/spk --detector yolo --dataset voc \\
        --all-failures --seed 42 \\
        --out-dir /content/spk/diagnostics/ood_false_positives

Per method under --out-dir:

  thresholds.json     per-class Isolation Forest calibration + FPR95
  failures.json       all accepted-FP cases (activations, OOD scores, optional concepts)
  failures_by_class/  same cases split by predicted class
  images/             one cropped visualization per case (full frame + box)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterator

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import run as pipeline
from run import (
    ConceptHead,
    GT_INDEX,
    KNN_COL,
    ROI_DIR,
    ROI_SPLITS,
    SPK4_COLS,
    SPK_FULL_COLS,
    VocSplitImageSource,
    canonical_pred_class,
    class_split_features,
    fpr95,
    keep_true_positives,
    load_saved_head,
    pool_logits,
    seed_dir,
    set_profile,
    set_root,
)

METHOD_SPECS: dict[str, tuple[str, list[str]]] = {
    "spk_local": ("spk local", list(SPK4_COLS)),
    "spk_full": ("spk full", list(SPK_FULL_COLS)),
}

OOD_SPLITS = ("near_ood", "far_ood")


def _sanitize_name(text: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", str(text))[:180]


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if np.isnan(v) else v
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    return value


def data_source_to_tar_split(data_source: str) -> str:
    mapping = {Path(v).stem: k for k, v in pipeline.PROFILE.split_output_names.items()}
    try:
        return mapping[data_source]
    except KeyError:
        raise SystemExit(f"unknown data_source {data_source!r} for dataset {pipeline.PROFILE.name}") from None


def row_match_key(row: pd.Series | dict[str, Any]) -> tuple[Any, ...]:
    if isinstance(row, pd.Series):
        row = row.to_dict()
    return (
        str(row["data_source"]),
        str(row["file_name"]),
        round(float(row["bbox_x1"]), 3),
        round(float(row["bbox_y1"]), 3),
        round(float(row["bbox_x2"]), 3),
        round(float(row["bbox_y2"]), 3),
        str(row["class"]),
    )


def try_fit_class_iforest(
    activations: pd.DataFrame,
    id_train_tp: pd.DataFrame,
    class_name: str,
    feature_cols: list[str],
    seed: int,
) -> tuple[IsolationForest, StandardScaler, dict[str, Any]] | None:
    """Mirror run.evaluate() for one class (outlier_mode=none). None if skipped."""
    id_val_frame = activations[activations["data_source"] == "id_val"]
    train, _ = class_split_features(id_train_tp, class_name, feature_cols)
    id_val, _ = class_split_features(id_val_frame, class_name, feature_cols)
    near, _ = class_split_features(
        activations[activations["data_source"] == "near_ood"], class_name, feature_cols
    )
    far, _ = class_split_features(
        activations[activations["data_source"] == "far_ood"], class_name, feature_cols
    )
    meta: dict[str, Any] = {
        "class": class_name,
        "n_train": len(train),
        "n_id_val": len(id_val),
        "n_near_ood": len(near),
        "n_far_ood": len(far),
        "feature_cols": list(feature_cols),
        "skipped": False,
    }
    if len(train) < 5 or len(id_val) < 5:
        meta["skipped"] = True
        meta["skip_reason"] = f"n_train={len(train)} n_id_val={len(id_val)} (need >= 5 each)"
        return None

    train_for_if = train
    if len(train) > 1500:
        keep = np.random.default_rng(seed).choice(len(train), 1500, replace=False)
        train_for_if = train[keep]

    scaler = StandardScaler().fit(train_for_if)
    forest = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        max_samples=min(512, len(train_for_if)),
        random_state=seed,
        n_jobs=-1,
    ).fit(scaler.transform(train_for_if))

    scores = {
        "id_val": forest.decision_function(scaler.transform(id_val)),
        "near_ood": forest.decision_function(scaler.transform(near)) if len(near) else np.zeros(0),
        "far_ood": forest.decision_function(scaler.transform(far)) if len(far) else np.zeros(0),
    }
    threshold = float(np.percentile(scores["id_val"], 5))
    near_f = fpr95(scores["id_val"], scores["near_ood"])
    far_f = fpr95(scores["id_val"], scores["far_ood"])
    meta.update(
        {
            "id_val_threshold_p5": threshold,
            "near_fpr95": near_f,
            "far_fpr95": far_f,
            "mean_fpr95": float(np.nanmean([near_f, far_f])),
        }
    )
    return forest, scaler, meta


def score_rows(
    frame: pd.DataFrame,
    forest: IsolationForest,
    scaler: StandardScaler,
    feature_cols: list[str],
    threshold: float,
) -> pd.DataFrame:
    out = frame.copy()
    x = out[feature_cols].to_numpy(dtype=np.float64)
    mask = np.isfinite(x).all(axis=1)
    scores = np.full(len(out), np.nan, dtype=np.float64)
    if mask.any():
        scores[mask] = forest.decision_function(scaler.transform(x[mask]))
    out["iforest_score"] = scores
    out["id_val_threshold_p5"] = threshold
    out["iforest_margin"] = scores - threshold
    out["ood_false_alarm"] = (
        out["data_source"].isin(OOD_SPLITS) & np.isfinite(scores) & (scores >= threshold)
    )
    return out


def activation_dict(row: pd.Series, feature_cols: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for col in SPK4_COLS:
        if col in row.index:
            out[col] = _json_value(row[col])
    if KNN_COL in row.index and KNN_COL in feature_cols:
        out[KNN_COL] = _json_value(row[KNN_COL])
    return out


def row_to_failure_case(
    row: pd.Series,
    *,
    case_id: str,
    method_slug: str,
    method_label: str,
    feature_cols: list[str],
    class_meta: dict[str, Any],
    concepts: dict[str, float] | None,
    image_relpath: str | None,
) -> dict[str, Any]:
    return _json_value(
        {
            "case_id": case_id,
            "method_slug": method_slug,
            "method": method_label,
            "class": str(row["class"]),
            "data_source": str(row["data_source"]),
            "file_name": str(row["file_name"]),
            "image_path": str(row.get("image_path", "")) or None,
            "bbox_xyxy": [
                float(row["bbox_x1"]),
                float(row["bbox_y1"]),
                float(row["bbox_x2"]),
                float(row["bbox_y2"]),
            ],
            "detector_confidence": float(row["detector_confidence"]),
            "activations": activation_dict(row, feature_cols),
            "feature_cols": list(feature_cols),
            "ood": {
                "iforest_score": float(row["iforest_score"]),
                "id_val_threshold_p5": float(row["id_val_threshold_p5"]),
                "iforest_margin": float(row["iforest_margin"]),
                "accepted_false_positive": True,
                "definition": (
                    "OOD detection (near_ood|far_ood) with IF decision_function >= "
                    "per-class id_val 5th percentile (same as run.py evaluate, outlier_mode=none)"
                ),
            },
            "class_calibration": class_meta,
            "concepts": concepts,
            "image_file": image_relpath,
            "match_key": list(row_match_key(row)),
        }
    )


def collect_failure_cases(
    activations: pd.DataFrame,
    id_train_tp: pd.DataFrame,
    class_names: list[str],
    method_slug: str,
    feature_cols: list[str],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    """Score all rows per class; return failure case dicts, thresholds meta, scored frame."""
    method_label = METHOD_SPECS[method_slug][0]
    missing_cols = [c for c in feature_cols if c not in activations.columns]
    if missing_cols:
        raise SystemExit(f"{method_slug}: missing columns in activations.csv: {missing_cols}")

    thresholds: dict[str, Any] = {
        "method_slug": method_slug,
        "method": method_label,
        "feature_cols": feature_cols,
        "seed": seed,
        "per_class": {},
    }
    scored_parts: list[pd.DataFrame] = []
    cases: list[dict[str, Any]] = []
    ordinal = 0

    for class_name in class_names:
        fitted = try_fit_class_iforest(activations, id_train_tp, class_name, feature_cols, seed)
        subset = activations.loc[activations["class"].astype(str) == class_name]
        if fitted is None:
            thresholds["per_class"][class_name] = {
                "skipped": True,
                "n_rows": len(subset),
            }
            continue
        forest, scaler, class_meta = fitted
        thresholds["per_class"][class_name] = class_meta
        scored = score_rows(subset, forest, scaler, feature_cols, class_meta["id_val_threshold_p5"])
        scored_parts.append(scored)
        failures = scored.loc[scored["ood_false_alarm"]].sort_values(
            ["data_source", "iforest_score"], ascending=[True, False]
        )
        for _, row in failures.iterrows():
            case_id = f"{method_slug}_{_sanitize_name(class_name)}_{ordinal:06d}"
            ordinal += 1
            cases.append(
                {
                    "case_id": case_id,
                    "row": row,
                    "class_meta": class_meta,
                    "concepts": None,
                    "image_relpath": None,
                }
            )

    scored_all = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    thresholds["n_failure_cases"] = len(cases)
    thresholds["n_ood_rows_scored"] = int(
        scored_all.loc[scored_all["data_source"].isin(OOD_SPLITS)].shape[0]
        if not scored_all.empty
        else 0
    )
    return cases, thresholds, scored_all


def score_full_concepts(
    cache_path: Path,
    split_name: str,
    class_name: str,
    head: ConceptHead,
    concept_order: list[str],
    device: str,
    batch_size: int = 64,
) -> dict[tuple[Any, ...], dict[str, float]]:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    metadata, fp8, scales = cache["metadata"], cache["features_fp8"], cache["scales"]
    row_indices = [i for i, row in enumerate(metadata) if canonical_pred_class(row) == class_name]
    if not row_indices:
        return {}

    torch_device = torch.device(device)
    lookup: dict[tuple[Any, ...], dict[str, float]] = {}
    head_ch = int(head.stem[0].weight.shape[1])
    with torch.inference_mode():
        for start in range(0, len(row_indices), batch_size):
            chunk = row_indices[start : start + batch_size]
            xb = torch.stack([fp8[i].float() * scales[i] for i in chunk]).to(torch_device)
            if xb.shape[1] != head_ch:
                raise SystemExit(
                    f"{cache_path}: ROI {xb.shape[1]}-d vs head {head_ch}-d for {class_name}"
                )
            pooled = torch.sigmoid(pool_logits(head(xb))).cpu().numpy()
            for i, activation in zip(chunk, pooled):
                row = metadata[i]
                x1, y1, x2, y2 = map(float, row["bbox_xyxy"])
                key = row_match_key(
                    {
                        "data_source": split_name,
                        "file_name": row["file_name"],
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                        "class": class_name,
                    }
                )
                lookup[key] = {name: float(val) for name, val in zip(concept_order, activation)}
    return lookup


def attach_concepts_to_cases(
    case_entries: list[dict[str, Any]],
    concept_root: Path,
    device: str,
) -> None:
    """Fill concepts dict on each case entry (mutates entries)."""
    by_class: dict[str, list[dict[str, Any]]] = {}
    for entry in case_entries:
        by_class.setdefault(str(entry["row"]["class"]), []).append(entry)

    for class_name, entries in by_class.items():
        head_path = concept_root / f"{class_name}_head.pt"
        if not head_path.is_file():
            print(f"  concepts: no head for {class_name}, skipping", flush=True)
            continue
        head, concept_order = load_saved_head(head_path, device)
        splits_needed = {str(e["row"]["data_source"]) for e in entries}
        lookup: dict[tuple[Any, ...], dict[str, float]] = {}
        for split_name, filename in ROI_SPLITS.items():
            if split_name not in splits_needed:
                continue
            cache = ROI_DIR / filename
            if not cache.is_file():
                continue
            lookup.update(
                score_full_concepts(cache, split_name, class_name, head, concept_order, device)
            )
        hit = 0
        for entry in entries:
            key = row_match_key(entry["row"])
            concepts = lookup.get(key)
            entry["concepts"] = concepts
            if concepts:
                hit += 1
        print(f"  concepts {class_name}: matched {hit}/{len(entries)} failure ROIs", flush=True)


class TarImageLoader:
    """Lazy tar reader per split."""

    def __init__(self) -> None:
        self._sources: dict[str, VocSplitImageSource] = {}
        self._lookups: dict[str, dict[str, pipeline.ImageItem]] = {}

    def close(self) -> None:
        for source in self._sources.values():
            source.close()
        self._sources.clear()

    def _lookup_table(self, data_source: str) -> dict[str, pipeline.ImageItem]:
        tar_split = data_source_to_tar_split(data_source)
        if tar_split not in self._lookups:
            self._lookups[tar_split] = {}
            with VocSplitImageSource(tar_split) as source:
                for item in source.iter_items():
                    self._lookups[tar_split][item.file_name] = item
                    self._lookups[tar_split].setdefault(Path(item.file_name).stem, item)
        return self._lookups[tar_split]

    def load_bgr(self, data_source: str, file_name: str) -> np.ndarray | None:
        tar_split = data_source_to_tar_split(data_source)
        table = self._lookup_table(data_source)
        item = table.get(str(file_name)) or table.get(Path(str(file_name)).stem)
        if item is None:
            return None
        if tar_split not in self._sources:
            self._sources[tar_split] = VocSplitImageSource(tar_split)
        return self._sources[tar_split].load_bgr(item)


def draw_detection(
    bgr: np.ndarray,
    row: pd.Series,
    title: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    img = bgr.copy()
    x1, y1, x2, y2 = map(int, [row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = (
        f"{title}  IF={row['iforest_score']:.3f}  "
        f"km={row.get('known_max', float('nan')):.2f} unk={row.get('unknown', float('nan')):.2f}"
    )
    cv2.putText(
        img, label[:96], (max(x1, 0), max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
    )
    return img


def render_failure_images(
    case_entries: list[dict[str, Any]],
    images_dir: Path,
    method_slug: str,
) -> int:
    images_dir.mkdir(parents=True, exist_ok=True)
    loader = TarImageLoader()
    saved = 0
    try:
        for entry in case_entries:
            row = entry["row"]
            case_id = entry["case_id"]
            bgr = loader.load_bgr(str(row["data_source"]), str(row["file_name"]))
            if bgr is None:
                continue
            title = f"{method_slug} {row['class']} {row['data_source']}"
            painted = draw_detection(bgr, row, title, (0, 0, 255))
            relpath = Path("images") / f"{case_id}.jpg"
            path = images_dir.parent / relpath
            cv2.imwrite(str(path), painted)
            entry["image_relpath"] = str(relpath)
            saved += 1
    finally:
        loader.close()
    return saved


def finalize_cases(
    case_entries: list[dict[str, Any]],
    method_slug: str,
    method_label: str,
    feature_cols: list[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in case_entries:
        out.append(
            row_to_failure_case(
                entry["row"],
                case_id=entry["case_id"],
                method_slug=method_slug,
                method_label=method_label,
                feature_cols=feature_cols,
                class_meta=entry["class_meta"],
                concepts=entry.get("concepts"),
                image_relpath=entry.get("image_relpath"),
            )
        )
    return out


def write_failure_bundle(
    out_method_dir: Path,
    method_slug: str,
    feature_cols: list[str],
    seed: int,
    thresholds: dict[str, Any],
    cases: list[dict[str, Any]],
    activations_path: Path,
) -> None:
    out_method_dir.mkdir(parents=True, exist_ok=True)
    method_label = METHOD_SPECS[method_slug][0]

    (out_method_dir / "thresholds.json").write_text(json.dumps(_json_value(thresholds), indent=2) + "\n")

    payload = {
        "version": 1,
        "method_slug": method_slug,
        "method": method_label,
        "feature_cols": feature_cols,
        "seed": seed,
        "detector": pipeline.PROFILE.detector,
        "dataset": pipeline.PROFILE.name,
        "activations_csv": str(activations_path),
        "ood_protocol": "near_far_ood",
        "id_train_filter": "same predicted class and IoU >= 0.5 with ground truth (run.py stage C)",
        "failure_definition": "near_ood|far_ood with iforest_score >= class id_val 5th percentile",
        "n_cases": len(cases),
        "cases": cases,
    }
    (out_method_dir / "failures.json").write_text(json.dumps(_json_value(payload), indent=2) + "\n")

    by_class_dir = out_method_dir / "failures_by_class"
    by_class_dir.mkdir(exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(str(case["class"]), []).append(case)
    for class_name, class_cases in sorted(grouped.items()):
        class_payload = {
            "class": class_name,
            "method_slug": method_slug,
            "n_cases": len(class_cases),
            "cases": class_cases,
        }
        (by_class_dir / f"{_sanitize_name(class_name)}.json").write_text(
            json.dumps(_json_value(class_payload), indent=2) + "\n"
        )

    print(
        f"  {method_slug}: {len(cases)} failure cases -> {out_method_dir / 'failures.json'}",
        flush=True,
    )


def export_method_failures(
    activations: pd.DataFrame,
    id_train_tp: pd.DataFrame,
    class_names: list[str],
    method_slug: str,
    out_dir: Path,
    seed: int,
    activations_path: Path,
    device: str,
    attach_concepts: bool,
    skip_images: bool,
) -> dict[str, Any]:
    feature_cols = METHOD_SPECS[method_slug][1]
    method_dir = out_dir / method_slug
    case_entries, thresholds, scored = collect_failure_cases(
        activations, id_train_tp, class_names, method_slug, feature_cols, seed
    )
    if attach_concepts and case_entries:
        attach_concepts_to_cases(case_entries, seed_dir(pipeline.ARCH_DIR / "concept_head_ood", seed), device)

    n_images = 0
    if not skip_images and case_entries:
        n_images = render_failure_images(case_entries, method_dir / "images", method_slug)

    cases = finalize_cases(
        case_entries, method_slug, METHOD_SPECS[method_slug][0], feature_cols
    )
    write_failure_bundle(method_dir, method_slug, feature_cols, seed, thresholds, cases, activations_path)

    if not scored.empty:
        scored.to_csv(method_dir / "all_scored_activations.csv", index=False)

    return {
        "method_slug": method_slug,
        "n_failure_cases": len(cases),
        "n_images_saved": n_images,
        "n_ood_scored": thresholds.get("n_ood_rows_scored", 0),
        "output_dir": str(method_dir),
    }


def plot_distributions(enriched: pd.DataFrame, class_name: str, threshold: float, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    splits = ["id_train", "id_val", "near_ood", "far_ood"]
    colors = {"id_train": "#2ca02c", "id_val": "#1f77b4", "near_ood": "#ff7f0e", "far_ood": "#d62728"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for src in splits:
        part = enriched.loc[enriched["data_source"] == src, "iforest_score"].dropna()
        if len(part):
            ax.hist(part, bins=40, alpha=0.55, label=f"{src} (n={len(part)})", color=colors[src])
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2, label=f"id_val p5={threshold:.3f}")
    ax.set_xlabel("Isolation Forest score (higher = more ID-like)")
    ax.set_title(f"{class_name}: IF score by split")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "iforest_scores_by_split.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.ravel(), SPK4_COLS):
        for src in splits:
            part = enriched.loc[enriched["data_source"] == src, col].dropna()
            if len(part):
                ax.hist(part, bins=35, alpha=0.45, label=src, color=colors[src])
        ax.set_title(col)
        ax.legend(fontsize=7)
    fig.suptitle(f"{class_name}: SPK4 feature distributions")
    fig.tight_layout()
    fig.savefig(fig_dir / "spk4_features_by_split.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for src in splits:
        part = enriched.loc[enriched["data_source"] == src]
        if len(part):
            ax.scatter(
                part["known_max"],
                part["unknown"],
                s=12,
                alpha=0.35,
                label=src,
                c=colors[src],
            )
    ax.set_xlabel("known_max")
    ax.set_ylabel("unknown")
    ax.set_title(f"{class_name}: known_max vs unknown")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "known_max_vs_unknown.png", dpi=150)
    plt.close(fig)


def save_example_images(
    enriched: pd.DataFrame,
    out_images: Path,
    top_k: int,
) -> list[dict[str, Any]]:
    out_images.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    loader = TarImageLoader()
    try:
        ood_worst = enriched.loc[enriched["ood_false_alarm"]].sort_values("iforest_score", ascending=False)
        if ood_worst.empty:
            ood_worst = enriched.loc[enriched["data_source"].isin(OOD_SPLITS)].sort_values(
                "iforest_score", ascending=False
            )
        id_ref = enriched.loc[enriched["data_source"] == "id_val"].sort_values(
            "iforest_score", ascending=False
        ).head(min(5, top_k // 3))

        rows_to_save = pd.concat([ood_worst.head(top_k), id_ref], ignore_index=True)
        for rank, (_, row) in enumerate(rows_to_save.iterrows()):
            bgr = loader.load_bgr(str(row["data_source"]), str(row["file_name"]))
            if bgr is None:
                continue
            is_ood = str(row["data_source"]) in OOD_SPLITS
            color = (0, 0, 255) if is_ood else (0, 200, 0)
            title = f"{row['data_source']} FA" if bool(row.get("ood_false_alarm")) else str(row["data_source"])
            painted = draw_detection(bgr, row, title, color)
            fname = _sanitize_name(f"{rank:03d}_{row['data_source']}_{row['file_name']}.jpg")
            path = out_images / fname
            cv2.imwrite(str(path), painted)
            manifest.append(_json_value({"image": str(path), "rank": rank, **row.to_dict()}))
    finally:
        loader.close()
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--detector", choices=("yolo", "frcnn", "rtdetr"), default="yolo")
    parser.add_argument("--dataset", choices=("voc", "bdd"), default="voc")
    parser.add_argument("--class-name", default=None, help="Single-class mode (omit with --all-failures)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--all-failures",
        action="store_true",
        help="Export every OOD accepted-FP for spk_local and spk_full (all classes)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHOD_SPECS),
        default=None,
        help="Which SPK methods to export (default: spk_local spk_full with --all-failures, else spk_local)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: diagnostics/<class> or diagnostics/ood_false_positives)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k-images", type=int, default=24, help="Single-class preview images (not all-failures)")
    parser.add_argument(
        "--attach-concepts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attach per-concept sigmoid activations to each failure case JSON",
    )
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-plots", action="store_true", help="Single-class mode only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_root(args.root)
    set_profile(args.detector, args.dataset)

    if args.all_failures and args.class_name:
        raise SystemExit("use --all-failures without --class-name, or single-class mode without --all-failures")
    if not args.all_failures and not args.class_name:
        args.class_name = "bird"

    methods = args.methods or (list(METHOD_SPECS) if args.all_failures else ["spk_local"])

    concept_root = pipeline.ARCH_DIR / "concept_head_ood"
    activations_path = seed_dir(concept_root, args.seed) / "activations.csv"
    if not activations_path.is_file():
        raise SystemExit(f"missing {activations_path} (run stage C first)")

    activations = pd.read_csv(activations_path)
    id_train = activations.loc[activations["data_source"] == "id_train"]
    id_train_tp = keep_true_positives(id_train, GT_INDEX)

    if args.all_failures:
        out_dir = args.out_dir or (args.root / "diagnostics" / "ood_false_positives")
        out_dir.mkdir(parents=True, exist_ok=True)
        class_names = sorted(activations["class"].astype(str).unique())
        print(f"=== export all OOD false positives ({len(class_names)} classes) ===", flush=True)
        method_reports = []
        for method_slug in methods:
            if method_slug == "spk_full" and KNN_COL not in activations.columns:
                print(f"  skip {method_slug}: no {KNN_COL} in activations.csv", flush=True)
                continue
            print(f"\n--- {method_slug} ---", flush=True)
            method_reports.append(
                export_method_failures(
                    activations,
                    id_train_tp,
                    class_names,
                    method_slug,
                    out_dir,
                    args.seed,
                    activations_path,
                    args.device,
                    attach_concepts=args.attach_concepts,
                    skip_images=args.skip_images,
                )
            )
        summary = {
            "mode": "all_failures",
            "detector": args.detector,
            "dataset": args.dataset,
            "seed": args.seed,
            "activations_csv": str(activations_path),
            "methods": method_reports,
        }
        (out_dir / "summary.json").write_text(json.dumps(_json_value(summary), indent=2) + "\n")
        print(f"\nwrote {out_dir / 'summary.json'}", flush=True)
        return

    class_name = args.class_name
    out_dir = args.out_dir or (args.root / "diagnostics" / class_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = activations.loc[activations["class"].astype(str) == class_name].copy()
    if subset.empty:
        raise SystemExit(f"no rows for class {class_name!r} in {activations_path}")

    method_slug = methods[0]
    feature_cols = METHOD_SPECS[method_slug][1]
    if method_slug == "spk_full" and KNN_COL not in activations.columns:
        print(f"  note: falling back to spk_local (no {KNN_COL})", flush=True)
        method_slug, feature_cols = "spk_local", list(SPK4_COLS)

    fitted = try_fit_class_iforest(activations, id_train_tp, class_name, feature_cols, args.seed)
    if fitted is None:
        raise SystemExit(f"cannot fit Isolation Forest for {class_name}")
    forest, scaler, if_meta = fitted
    enriched = score_rows(subset, forest, scaler, feature_cols, if_meta["id_val_threshold_p5"])
    enriched["ood_false_alarm"] = enriched["data_source"].isin(OOD_SPLITS) & (
        enriched["iforest_score"] >= if_meta["id_val_threshold_p5"]
    )

    enriched_path = out_dir / f"{class_name}_activations_enriched.csv"
    enriched.to_csv(enriched_path, index=False)
    print(f"wrote {enriched_path} ({len(enriched):,} rows)", flush=True)

    case_entries, _, _ = collect_failure_cases(
        activations, id_train_tp, [class_name], method_slug, feature_cols, args.seed
    )
    if args.attach_concepts and case_entries:
        attach_concepts_to_cases(case_entries, seed_dir(concept_root, args.seed), args.device)
    if not args.skip_images:
        render_failure_images(case_entries, out_dir / method_slug / "images", method_slug)
    cases = finalize_cases(case_entries, method_slug, METHOD_SPECS[method_slug][0], feature_cols)
    write_failure_bundle(
        out_dir / method_slug,
        method_slug,
        feature_cols,
        args.seed,
        {"per_class": {class_name: if_meta}, "method_slug": method_slug},
        cases,
        activations_path,
    )

    summary = {
        "mode": "single_class",
        "class": class_name,
        "method": method_slug,
        "iforest": if_meta,
        "enriched_csv": str(enriched_path),
        "n_failure_cases": len(cases),
    }

    if not args.skip_plots:
        plot_distributions(enriched, class_name, if_meta["id_val_threshold_p5"], out_dir / "figures")
        summary["figures"] = str(out_dir / "figures")

    if not args.skip_images:
        summary["preview_images"] = save_example_images(enriched, out_dir / "preview_images", args.top_k_images)

    (out_dir / "summary.json").write_text(json.dumps(_json_value(summary), indent=2) + "\n")
    print(f"wrote {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
