#!/usr/bin/env python3
"""Run BAM / KNN / MDS on SPK activation features (from activations.csv).

Uses ood_baseline.fit_model / score / metrics with SPK feature vectors instead of
detector logits.  Training defaults to ID-train true positives (IoU >= 0.5), matching
run.py concept-head evaluation.

Example (YOLO VOC on Colab):
  python eval_spk_baselines.py \\
    --activations /content/spk/data/yolo/voc/concept_head_ood/activations.csv \\
    --gt-index /content/spk/data/id/gt_voc.json \\
    --out /content/spk/data/yolo/voc/spk_baselines
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import ood_baseline

SPK4 = ["known_max", "unknown", "proxy_max", "relative_area"]
SPK_FULL = SPK4 + ["native_knn"]
SPK_METHODS = ["MDS", "BAM", "KNN", "iForest"]
DISPLAY_NAME = {"MDS": "SPK MDS", "BAM": "SPK BAM", "KNN": "SPK KNN", "iForest": "SPK IF"}
SEEDS = (42, 43, 44)

SPLIT_MAP = {
    "id_train": "train",
    "id_val": "id_val",
    "near_ood": "near_ood",
    "far_ood": "far_ood",
}


def iou(box_a, box_b) -> float:
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return overlap / max(area_a + area_b - overlap, 1e-8)


def keep_true_positives(frame: pd.DataFrame, gt_index_path: Path) -> pd.DataFrame:
    ground_truth = json.loads(gt_index_path.read_text())
    keep = []
    for class_name, file_name, x1, y1, x2, y2 in frame[
        ["class", "file_name", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    ].to_numpy():
        candidates = ground_truth.get(Path(file_name).stem, [])
        best = max(
            (iou([x1, y1, x2, y2], gt["bbox_xyxy"]) for gt in candidates if gt["class"] == class_name),
            default=0.0,
        )
        keep.append(best >= 0.5)
    return frame.loc[keep].copy()


def build_class_map(classes: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(sorted(classes))}


def frame_to_split(frame: pd.DataFrame, feature_cols: list[str], class_map: dict[str, int]) -> dict:
    x = frame[feature_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("non-finite SPK features")
    labels = frame["class"].astype(str).map(class_map).to_numpy(dtype=np.int64)
    ids = (
        frame.index.astype(str)
        + ":"
        + frame["class"].astype(str)
        + ":"
        + frame["file_name"].astype(str)
        + ":"
        + frame["bbox_x1"].map(lambda v: f"{float(v):.4f}")
        + ":"
        + frame["bbox_y1"].map(lambda v: f"{float(v):.4f}")
        + ":"
        + frame["bbox_x2"].map(lambda v: f"{float(v):.4f}")
        + ":"
        + frame["bbox_y2"].map(lambda v: f"{float(v):.4f}")
    ).to_numpy(dtype=str)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("detection_ids not unique within split")
    return dict(logits=x, x=x, labels=labels, ids=ids, names=np.array(feature_cols, dtype=str))


def method_label(method: str) -> str:
    return DISPLAY_NAME.get(method, f"SPK {method}")


def _fmt_mean_std(values: list[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).any():
        return "nan"
    return f"{float(np.mean(arr)):6.2f} ± {float(np.std(arr, ddof=1) if arr.size > 1 else 0.0):.2f}"


def run_eval(args: argparse.Namespace) -> int:
    activations = pd.read_csv(args.activations)
    missing_cols = [c for c in args.features if c not in activations.columns]
    if missing_cols:
        raise SystemExit(f"missing columns in activations.csv: {missing_cols}")

    class_names = sorted(activations["class"].astype(str).unique())
    class_map = build_class_map(class_names)

    splits: dict[str, pd.DataFrame] = {}
    for src, dst in SPLIT_MAP.items():
        part = activations[activations["data_source"] == src].copy()
        if dst == "train" and args.train_tp_only:
            if not args.gt_index.is_file():
                raise SystemExit(f"--gt-index required for TP filter: {args.gt_index}")
            before = len(part)
            part = keep_true_positives(part, args.gt_index)
            print(f"  train TP filter: {len(part):,} / {before:,}", flush=True)
        finite = np.isfinite(part[args.features].to_numpy(dtype=np.float64)).all(axis=1)
        dropped = int((~finite).sum())
        if dropped:
            print(f"  {dst}: dropped {dropped} non-finite feature rows", flush=True)
        part = part.loc[finite].copy()
        splits[dst] = part
        print(f"  {dst}: {len(part):,} rows", flush=True)

    data = {
        name: frame_to_split(frame, args.features, class_map)
        for name, frame in splits.items()
        if len(frame)
    }
    if "train" not in data or not len(data["train"]["x"]):
        raise SystemExit("empty training split after filtering")
    if "id_val" not in data or not len(data["id_val"]["x"]):
        raise SystemExit("empty id_val split")

    args.out.mkdir(parents=True, exist_ok=True)
    eval_names = [k for k in data if k != "train"]
    ood_splits = [k for k in eval_names if k != "id_val"]
    rows = []
    by_method: dict[str, dict[str, dict[str, list[float]]]] = {
        method_label(m): {split: {"fpr95_pct": [], "auroc_pct": []} for split in ood_splits}
        for m in args.methods
    }
    report = {
        "feature_set": list(args.features),
        "train_tp_only": args.train_tp_only,
        "outlier_removal": "none",
        "classes": class_names,
        "seeds": list(args.seeds),
        "methods": {},
        "errors": {},
    }

    print(f"\n=== SPK baselines ({len(args.features)}D, outlier=none) -> {args.out} ===", flush=True)
    print(
        f"train={len(data['train']['x']):,}  seeds={list(args.seeds)}  "
        f"bam_density={args.bam_density}  knn_k={args.knn_k}",
        flush=True,
    )

    for seed in args.seeds:
        baseline_args = SimpleNamespace(
            mds_labels="predicted",
            covariance=args.covariance,
            knn_mode=args.knn_mode,
            knn_k=args.knn_k,
            bam_density=args.bam_density,
            bam_max_boxes=args.bam_max_boxes,
            bam_cluster=args.bam_cluster,
            iforest_scope="classwise",
            trees=200,
            iforest_samples=512,
            seed=seed,
            jobs=args.jobs,
            batch_size=args.batch_size,
            scale_percentile=65.0,
            scale_degenerate="error",
            msp_denominator="all",
            msp_activation="softmax",
            temperature=1.0,
        )
        train_x = data["train"]["x"]
        train_y = data["train"]["labels"]
        if args.max_train_per_class:
            ix = ood_baseline.train_indices(train_y, args.max_train_per_class, seed)
            train_x, train_y = train_x[ix], train_y[ix]
        print(f"\n--- seed {seed} train={len(train_x):,} ---", flush=True)

        for method in args.methods:
            label = method_label(method)
            started = time.perf_counter()
            try:
                print(f"[{label}] fitting/scoring", flush=True)
                model = ood_baseline.fit_model(method, train_x, train_y, baseline_args)
                scores = {}
                fallbacks = {}
                for split in eval_names:
                    scores[split], fallbacks[split] = ood_baseline.score(
                        method, model, data[split], baseline_args
                    )
                wall = time.perf_counter() - started
                seed_result = dict(
                    fallback_rows=fallbacks,
                    evaluations={"full_id": {}},
                    wall_seconds=wall,
                )
                for split in ood_splits:
                    metrics = ood_baseline.metrics(scores["id_val"], scores[split])
                    seed_result["evaluations"]["full_id"][split] = metrics
                    rows.append(dict(method=label, seed=seed, protocol="full_id", split=split, **metrics))
                    if metrics.get("fpr95_pct") is not None:
                        by_method[label][split]["fpr95_pct"].append(float(metrics["fpr95_pct"]))
                    if metrics.get("auroc_pct") is not None:
                        by_method[label][split]["auroc_pct"].append(float(metrics["auroc_pct"]))
                report["methods"].setdefault(label, {})[str(seed)] = seed_result
                print(f"[{label}] done in {wall:.2f}s", flush=True)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                report["errors"][f"{label}/seed{seed}"] = str(exc)
                print(f"[{label}] FAILED: {exc}", flush=True)

    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if rows:
        with open(args.out / "summary.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print("\n=== 3-seed mean ± std (outlier removal: none) ===", flush=True)
    header = f"{'method':10s} {'near FPR95':14s} {'far FPR95':14s} {'near AUROC':14s} {'far AUROC':14s}"
    print(header, flush=True)
    pooled_rows = []
    for method in args.methods:
        label = method_label(method)
        near = by_method[label].get("near_ood", {"fpr95_pct": [], "auroc_pct": []})
        far = by_method[label].get("far_ood", {"fpr95_pct": [], "auroc_pct": []})
        line = (
            f"{label:10s} {_fmt_mean_std(near['fpr95_pct']):14s} {_fmt_mean_std(far['fpr95_pct']):14s} "
            f"{_fmt_mean_std(near['auroc_pct']):14s} {_fmt_mean_std(far['auroc_pct']):14s}"
        )
        print(line, flush=True)
        pooled_rows.append({
            "method": label,
            "outlier_removal": "none",
            "near_fpr95": _fmt_mean_std(near["fpr95_pct"]).strip(),
            "far_fpr95": _fmt_mean_std(far["fpr95_pct"]).strip(),
            "near_auroc": _fmt_mean_std(near["auroc_pct"]).strip(),
            "far_auroc": _fmt_mean_std(far["auroc_pct"]).strip(),
        })
    (args.out / "pooled_mean_std.json").write_text(json.dumps(pooled_rows, indent=2) + "\n")

    print("\n=== summary.csv ===", flush=True)
    if (args.out / "summary.csv").is_file():
        print((args.out / "summary.csv").read_text(), flush=True)

    return 2 if report["errors"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--gt-index", type=Path, default=Path("/content/spk/data/id/gt_voc.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--features", nargs="+", choices=["known_max", "unknown", "proxy_max", "relative_area", "native_knn"],
                        default=SPK_FULL, help="SPK feature columns (default: spk full 5D)")
    parser.add_argument("--methods", nargs="+", choices=["BAM", "KNN", "MDS", "iForest"],
                        default=SPK_METHODS)
    parser.add_argument("--train-tp-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bam-density", type=float, default=5.0)
    parser.add_argument("--bam-max-boxes", type=int, default=256)
    parser.add_argument("--bam-cluster", choices=["minibatch", "kmeans"], default="minibatch")
    parser.add_argument("--knn-mode", choices=["mhood", "sun"], default="mhood")
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--covariance", choices=["empirical", "ledoit-wolf"], default="empirical")
    parser.add_argument("--max-train-per-class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None, help="Deprecated: use --seeds")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS),
                        help="RNG seeds for BAM / iForest (default: 42 43 44)")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.seed is not None:
        args.seeds = [args.seed]
    with threadpool_limits(limits=args.jobs):
        return run_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
