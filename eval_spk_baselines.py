#!/usr/bin/env python3
"""SPK-feature OOD baselines (MDS / BAM / KNN / Isolation Forest).

Typical Colab flow:

  DETECTOR=yolo DATASET=bdd bash mount_data.sh
  python eval_spk_baselines.py --detector yolo --dataset bdd

Paths default to ``<root>/data/{detector}/{dataset}/``:
  concept_head_ood/seed_42/activations.csv  (or a flat activations.csv)
  -> spk_baselines/
  -> MyDrive/experiments/{detector}-{dataset}/spk_variants/  (auto backup)
GT index: ``<root>/data/id/gt_{dataset}.json`` (built by run.py stage A).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
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
SEEDS = (42,)
DEFAULT_ROOT = Path("/content/spk")
EXPERIMENTS_DRIVE_DEFAULT = Path("/content/drive/MyDrive/experiments")
DEFAULT_BAM_DENSITY_SWEEP = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0)

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
        raw = str(file_name)
        stem = Path(raw).stem
        name = Path(raw).name
        alt = name.rsplit("_", 1)[-1] if name.startswith("train_") and name.count("_") >= 2 else name
        candidates = (
            ground_truth.get(stem)
            or ground_truth.get(Path(alt).stem)
            or []
        )
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


def load_spk_splits(args: argparse.Namespace, activations_path: Path) -> tuple[dict, list[str]]:
    activations = pd.read_csv(activations_path)
    missing_cols = [c for c in args.features if c not in activations.columns]
    if missing_cols:
        raise SystemExit(f"missing columns in {activations_path}: {missing_cols}")

    class_names = sorted(activations["class"].astype(str).unique())
    class_map = build_class_map(class_names)

    splits: dict[str, pd.DataFrame] = {}
    print(f"\n  activations {activations_path}", flush=True)
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
        raise SystemExit(f"empty training split after filtering: {activations_path}")
    if "id_val" not in data or not len(data["id_val"]["x"]):
        raise SystemExit(f"empty id_val split: {activations_path}")
    return data, class_names


def discover_seed_csvs(head_dir: Path, seeds: list[int] | None = None) -> list[tuple[int, Path]]:
    """Return activations for requested seeds only (default: 42)."""
    wanted = list(seeds) if seeds is not None else list(SEEDS)
    jobs = []
    for seed in wanted:
        csv_path = head_dir / f"seed_{seed}" / "activations.csv"
        if csv_path.is_file():
            jobs.append((int(seed), csv_path))
    return jobs


def resolve_bundle(args: argparse.Namespace) -> None:
    """Fill activations / seed-root / gt / out from --detector --dataset --root."""
    if args.detector or args.dataset:
        if not args.detector or not args.dataset:
            raise SystemExit("pass both --detector and --dataset (or explicit --activations/--out)")
        root = args.root.expanduser().resolve()
        arch = root / "data" / args.detector / args.dataset
        head = arch / "concept_head_ood"
        if args.gt_index is None:
            args.gt_index = root / "data" / "id" / f"gt_{args.dataset}.json"
        if args.out is None:
            args.out = arch / "spk_baselines"
        if args.seed_root is None and args.activations is None:
            seeded = discover_seed_csvs(head, args.seeds)
            flat = head / "activations.csv"
            if seeded:
                args.seed_root = head
            elif flat.is_file():
                args.activations = flat
            else:
                need = ", ".join(f"seed_{s}/activations.csv" for s in args.seeds)
                raise SystemExit(
                    f"no activations under {head} (need {need} or activations.csv). "
                    f"Mount with DETECTOR={args.detector} DATASET={args.dataset} bash mount_data.sh "
                    "after run.py stage C, or copy concept_head_ood from Drive experiments."
                )
    if args.gt_index is None:
        args.gt_index = DEFAULT_ROOT / "data/id/gt_voc.json"
    if args.out is None:
        raise SystemExit("need --out, or --detector and --dataset")
    print(
        f"detector={args.detector or '-'}  dataset={args.dataset or '-'}  root={args.root}\n"
        f"  gt={args.gt_index}\n"
        f"  activations={args.seed_root or args.activations}\n"
        f"  out={args.out}",
        flush=True,
    )


def seed_jobs(args: argparse.Namespace) -> list[tuple[int, Path]]:
    if args.seed_root is not None:
        jobs = discover_seed_csvs(args.seed_root, args.seeds)
        if not jobs:
            need = ", ".join(f"seed_{s}/activations.csv" for s in args.seeds)
            raise SystemExit(f"no {need} under {args.seed_root}")
        return jobs
    if args.activations is None:
        raise SystemExit("need --detector/--dataset, --activations, or --seed-root")
    return [(int(seed), args.activations) for seed in args.seeds]


def resolve_experiments_root(args: argparse.Namespace) -> Path | None:
    if args.no_backup:
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


def experiment_subdir(experiments_root: Path, detector: str, dataset: str) -> Path:
    return experiments_root / f"{detector}-{dataset}"


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
    """Copy file or directory contents into dest (never dest/src.name nesting)."""
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


def backup_spk_variants(experiments_root: Path, args: argparse.Namespace) -> None:
    """Sync local spk_baselines/ into experiments/{detector}-{dataset}/spk_variants/."""
    if not args.detector or not args.dataset:
        print("  backup spk_variants: skip (need --detector and --dataset)", flush=True)
        return
    if not args.out.is_dir() or not any(args.out.rglob("*")):
        print(f"  backup spk_variants: skip (empty {args.out})", flush=True)
        return
    dest = experiment_subdir(experiments_root, args.detector, args.dataset) / "spk_variants"
    copied, nbytes = _sync_into(args.out, dest)
    if copied:
        print(
            f"  backup spk_variants: {copied} file(s), {_human_bytes(nbytes)} -> {dest}",
            flush=True,
        )
    else:
        print(f"  backup spk_variants: up to date at {dest}", flush=True)


def make_baseline_args(args: argparse.Namespace, seed: int, bam_density: float | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        mds_labels="predicted",
        covariance=args.covariance,
        knn_mode=args.knn_mode,
        knn_k=args.knn_k,
        bam_density=args.bam_density if bam_density is None else bam_density,
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


def eval_bam_seed(
    data: dict,
    train_x: np.ndarray,
    train_y: np.ndarray,
    baseline_args: SimpleNamespace,
) -> dict[str, dict[str, float | None]]:
    eval_names = [k for k in data if k != "train"]
    ood_splits = [k for k in eval_names if k != "id_val"]
    model = ood_baseline.fit_model("BAM", train_x, train_y, baseline_args)
    scores = {}
    for split in eval_names:
        scores[split], _ = ood_baseline.score("BAM", model, data[split], baseline_args)
    metrics = {}
    for split in ood_splits:
        metrics[split] = ood_baseline.metrics(scores["id_val"], scores[split])
    return metrics


def run_bam_density_sweep(args: argparse.Namespace) -> int:
    jobs = seed_jobs(args)
    sweep_dir = args.out / "bam_density_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    densities = list(args.sweep_bam_density)
    rows: list[dict] = []

    print(
        f"\n=== BAM density sweep ({len(args.features)}D) -> {sweep_dir} ===",
        flush=True,
    )
    print(f"densities={densities}  jobs={len(jobs)}", flush=True)

    for seed, activations_path in jobs:
        data, _ = load_spk_splits(args, activations_path)
        train_x = data["train"]["x"]
        train_y = data["train"]["labels"]
        if args.max_train_per_class:
            ix = ood_baseline.train_indices(train_y, args.max_train_per_class, seed)
            train_x, train_y = train_x[ix], train_y[ix]
        print(f"\n--- seed {seed} train={len(train_x):,} ---", flush=True)

        for density in densities:
            baseline_args = make_baseline_args(args, seed, bam_density=float(density))
            started = time.perf_counter()
            try:
                metrics = eval_bam_seed(data, train_x, train_y, baseline_args)
                wall = time.perf_counter() - started
                near = metrics["near_ood"]
                far = metrics["far_ood"]
                mean_fpr = float(np.mean([near["fpr95_pct"], far["fpr95_pct"]]))
                row = {
                    "seed": seed,
                    "bam_density": float(density),
                    "near_fpr95_pct": near["fpr95_pct"],
                    "far_fpr95_pct": far["fpr95_pct"],
                    "mean_fpr95_pct": mean_fpr,
                    "near_auroc_pct": near["auroc_pct"],
                    "far_auroc_pct": far["auroc_pct"],
                    "wall_seconds": wall,
                }
                rows.append(row)
                print(
                    f"  density={density:5.1f}  near={near['fpr95_pct']:6.2f}  "
                    f"far={far['fpr95_pct']:6.2f}  mean={mean_fpr:6.2f}  "
                    f"({wall:.2f}s)",
                    flush=True,
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                print(f"  density={density:5.1f}  FAILED: {exc}", flush=True)

    if not rows:
        raise SystemExit("BAM density sweep produced no successful runs")

    sweep_csv = sweep_dir / "sweep.csv"
    with open(sweep_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_density: dict[float, list[dict]] = {}
    for row in rows:
        by_density.setdefault(float(row["bam_density"]), []).append(row)

    pooled = []
    print(f"\n=== pooled over {len(jobs)} seed(s) ===", flush=True)
    print(f"{'density':8s} {'near FPR95':14s} {'far FPR95':14s} {'mean FPR95':14s}", flush=True)
    for density in densities:
        part = by_density.get(float(density), [])
        if not part:
            continue
        near_vals = [float(r["near_fpr95_pct"]) for r in part if r["near_fpr95_pct"] is not None]
        far_vals = [float(r["far_fpr95_pct"]) for r in part if r["far_fpr95_pct"] is not None]
        mean_vals = [float(r["mean_fpr95_pct"]) for r in part if r["mean_fpr95_pct"] is not None]
        pooled.append({
            "bam_density": float(density),
            "near_fpr95": _fmt_mean_std(near_vals).strip(),
            "far_fpr95": _fmt_mean_std(far_vals).strip(),
            "mean_fpr95": _fmt_mean_std(mean_vals).strip(),
        })
        print(
            f"{density:8.1f} {_fmt_mean_std(near_vals):14s} "
            f"{_fmt_mean_std(far_vals):14s} {_fmt_mean_std(mean_vals):14s}",
            flush=True,
        )

    best = min(
        pooled,
        key=lambda row: float(row["mean_fpr95"].split()[0])
        if row["mean_fpr95"] != "nan"
        else float("inf"),
    )
    (sweep_dir / "pooled_mean_std.json").write_text(json.dumps(pooled, indent=2) + "\n")
    print(
        f"\nbest mean FPR95: density={best['bam_density']}  "
        f"near={best['near_fpr95']}  far={best['far_fpr95']}  mean={best['mean_fpr95']}",
        flush=True,
    )
    print(f"wrote {sweep_csv}", flush=True)
    return 0


def run_eval(args: argparse.Namespace) -> int:
    jobs = seed_jobs(args)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    ood_splits_seen: list[str] = []
    by_method: dict[str, dict[str, dict[str, list[float]]]] = {
        method_label(m): {} for m in args.methods
    }
    report = {
        "feature_set": list(args.features),
        "train_tp_only": args.train_tp_only,
        "outlier_removal": "none",
        "jobs": [{"seed": s, "activations": str(p)} for s, p in jobs],
        "methods": {},
        "errors": {},
    }

    print(f"\n=== SPK baselines ({len(args.features)}D, outlier=none) -> {args.out} ===", flush=True)
    print(f"jobs={len(jobs)}  bam_density={args.bam_density}  knn_k={args.knn_k}", flush=True)

    for seed, activations_path in jobs:
        data, class_names = load_spk_splits(args, activations_path)
        report.setdefault("classes", class_names)
        eval_names = [k for k in data if k != "train"]
        ood_splits = [k for k in eval_names if k != "id_val"]
        for split in ood_splits:
            if split not in ood_splits_seen:
                ood_splits_seen.append(split)
            for method in args.methods:
                by_method[method_label(method)].setdefault(split, {"fpr95_pct": [], "auroc_pct": []})
        baseline_args = make_baseline_args(args, seed)
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

    print(f"\n=== {len(jobs)}-seed mean ± std (outlier removal: none) ===", flush=True)
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
    parser.add_argument("--detector", choices=("yolo", "frcnn", "rtdetr"), default=None)
    parser.add_argument("--dataset", choices=("voc", "bdd"), default=None)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Bundle root (default: /content/spk)")
    parser.add_argument("--activations", type=Path, default=None)
    parser.add_argument("--seed-root", type=Path, default=None,
                        help="concept_head_ood dir; uses seed_42/activations.csv only")
    parser.add_argument("--gt-index", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--features", nargs="+", choices=["known_max", "unknown", "proxy_max", "relative_area", "native_knn"],
                        default=SPK_FULL, help="SPK feature columns (default: spk full 5D)")
    parser.add_argument("--methods", nargs="+", choices=["BAM", "KNN", "MDS", "iForest"],
                        default=SPK_METHODS)
    parser.add_argument("--train-tp-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bam-density", type=float, default=5.0,
                        help="BAM boxes per class ~ train_rows / density (m-hood: VOC=5, BDD=50)")
    parser.add_argument(
        "--sweep-bam-density",
        nargs="*",
        type=float,
        default=None,
        metavar="D",
        help="Sweep BAM only over these densities (default grid: 1 2 3 5 10 20 50)",
    )
    parser.add_argument("--bam-max-boxes", type=int, default=256)
    parser.add_argument("--bam-cluster", choices=["minibatch", "kmeans"], default="minibatch")
    parser.add_argument("--knn-mode", choices=["mhood", "sun"], default="mhood")
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--covariance", choices=["empirical", "ledoit-wolf"], default="empirical")
    parser.add_argument("--max-train-per-class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None, help="Single seed when using one activations.csv")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS),
                        help="Which seed_* dirs / RNG seeds to use (default: 42 only)")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=None,
        help="Drive experiments root for spk_variants backup "
             "(default: /content/drive/MyDrive/experiments when mounted)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not copy results to experiments/{detector}-{dataset}/spk_variants/",
    )
    args = parser.parse_args()
    if args.seed is not None:
        args.seeds = [args.seed]
    resolve_bundle(args)
    experiments_root = resolve_experiments_root(args)
    if experiments_root is not None and args.detector and args.dataset:
        dest = experiment_subdir(experiments_root, args.detector, args.dataset) / "spk_variants"
        print(f"  experiments backup -> {dest}/", flush=True)
    elif not args.no_backup:
        print(
            "  experiments backup disabled (mount Drive or pass --experiments-dir; --no-backup to silence)",
            flush=True,
        )
    with threadpool_limits(limits=args.jobs):
        if args.sweep_bam_density is not None:
            args.sweep_bam_density = args.sweep_bam_density or list(DEFAULT_BAM_DENSITY_SWEEP)
            code = run_bam_density_sweep(args)
        else:
            code = run_eval(args)
    if experiments_root is not None:
        backup_spk_variants(experiments_root, args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
