#!/usr/bin/env python3
"""Run ood_baseline.py on detector logits from logits_extraction.py (VOC or BDD).

Expects under --logits-dir:
  BDD: bdd-train.pt  bdd-val.pt  near-ood.pt  far-ood.pt   (--dataset bdd)
  VOC: voc-train.pt  voc-val.pt  near-ood.pt  far-ood.pt   (--dataset voc)

Converts each .pt to .npz (cached), then runs all eight logit-space OOD baselines
from ood_baseline.py (MSP, EBO, MLS, SCALE, MDS, BAM, KNN, iForest).

FRCNN logits include a background column (index 20 for VOC, 10 for BDD). Pass
--detector frcnn or rely on background_index stored in the .pt files.

Examples
--------
    # YOLO VOC
    python eval_voc_logits_baselines.py --dataset voc

    # FRCNN VOC (logits already at /content/spk/data/frcnn/voc/logits)
    python eval_voc_logits_baselines.py --detector frcnn --dataset voc

    python eval_voc_logits_baselines.py --dataset bdd --methods MSP EBO MLS
    python eval_voc_logits_baselines.py --dataset bdd --outlier mhood-iqr --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import ood_baseline

DATASET_SPLITS = {
    "voc": {
        "train": "voc-train.pt",
        "id_val": "voc-val.pt",
        "near_ood": "near-ood.pt",
        "far_ood": "far-ood.pt",
        "bam_density": 5.0,
        "background_index": None,
    },
    "bdd": {
        "train": "bdd-train.pt",
        "id_val": "bdd-val.pt",
        "near_ood": "near-ood.pt",
        "far_ood": "far-ood.pt",
        "bam_density": 50.0,
        "background_index": None,
    },
}

DETECTOR_DEFAULTS = {
    "yolo": {
        "background_index": None,
        "msp_activation": "sigmoid",
    },
    "frcnn": {
        "background_index": {"voc": 20, "bdd": 10},
        "msp_activation": "softmax",
    },
}


def load_logits_pt(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    logits = np.asarray(payload["logits"], dtype=np.float64)
    pred_labels = np.asarray(payload["pred_labels"], dtype=np.int64)
    detection_ids = np.asarray(payload["detection_ids"], dtype=str)
    class_names = np.asarray(payload["class_names"], dtype=str)
    background_index = payload.get("background_index")
    if background_index is not None:
        background_index = int(background_index)
    if logits.ndim != 2:
        raise ValueError(f"{path}: logits must be [N, C], got {logits.shape}")
    if pred_labels.shape != (logits.shape[0],):
        raise ValueError(f"{path}: pred_labels length mismatch")
    if detection_ids.shape != (logits.shape[0],):
        raise ValueError(f"{path}: detection_ids length mismatch")
    if class_names.shape != (logits.shape[1],):
        raise ValueError(f"{path}: class_names length mismatch")
    if len(np.unique(detection_ids)) != len(detection_ids):
        raise ValueError(f"{path}: detection_ids must be unique")
    return {
        "logits": logits,
        "pred_labels": pred_labels,
        "detection_ids": detection_ids,
        "class_names": class_names,
        "background_index": background_index,
        "split": payload.get("split", path.stem),
        "num_detections": int(logits.shape[0]),
    }


def write_npz(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        logits=payload["logits"],
        pred_labels=payload["pred_labels"],
        detection_ids=payload["detection_ids"],
        class_names=payload["class_names"],
    )


def read_background_index(pt_path: Path) -> int | None:
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    value = payload.get("background_index")
    return int(value) if value is not None else None


def convert_splits(
    logits_dir: Path, cache_dir: Path, split_files: dict[str, str], force: bool,
) -> tuple[dict[str, Path], int | None]:
    npz_paths: dict[str, Path] = {}
    background_index: int | None = None
    print(f"logits dir: {logits_dir}", flush=True)
    print(f"npz cache:  {cache_dir}", flush=True)
    for key, filename in split_files.items():
        pt_path = logits_dir / filename
        if not pt_path.is_file():
            raise FileNotFoundError(f"missing {pt_path}")
        npz_path = cache_dir / f"{Path(filename).stem}.npz"
        pt_bg = read_background_index(pt_path)
        if pt_bg is not None:
            if background_index is None:
                background_index = pt_bg
            elif background_index != pt_bg:
                raise ValueError(
                    f"{pt_path}: background_index={pt_bg} != {background_index} from earlier split"
                )
        if npz_path.is_file() and not force:
            print(f"  reuse {npz_path.name}", flush=True)
        else:
            payload = load_logits_pt(pt_path)
            write_npz(npz_path, payload)
            print(
                f"  wrote {npz_path.name}  N={payload['num_detections']}  C={payload['logits'].shape[1]}",
                flush=True,
            )
        npz_paths[key] = npz_path
    return npz_paths, background_index


def build_baseline_args(args: argparse.Namespace, npz_paths: dict[str, Path]) -> SimpleNamespace:
    return SimpleNamespace(
        train=npz_paths["train"],
        id_val=npz_paths["id_val"],
        near_ood=npz_paths["near_ood"],
        far_ood=npz_paths["far_ood"],
        out=args.out,
        methods=list(args.methods),
        background_index=args.background_index,
        msp_denominator=args.msp_denominator,
        msp_activation=args.msp_activation,
        temperature=args.temperature,
        scale_percentile=args.scale_percentile,
        scale_degenerate=args.scale_degenerate,
        mds_labels=args.mds_labels,
        covariance=args.covariance,
        knn_mode=args.knn_mode,
        knn_k=args.knn_k,
        bam_density=args.bam_density,
        bam_max_boxes=args.bam_max_boxes,
        bam_cluster=args.bam_cluster,
        iforest_scope=args.iforest_scope,
        trees=args.trees,
        iforest_samples=args.iforest_samples,
        max_train_per_class=args.max_train_per_class,
        seed=args.seed,
        jobs=args.jobs,
        batch_size=args.batch_size,
        outlier=args.outlier,
        outlier_k=args.outlier_k,
        iqr_factor=args.iqr_factor,
        tail_fraction=args.tail_fraction,
        tail_scope=args.tail_scope,
        force=args.force,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", choices=tuple(DATASET_SPLITS), default="voc")
    parser.add_argument(
        "--detector",
        choices=tuple(DETECTOR_DEFAULTS),
        default="yolo",
        help="yolo or frcnn (default paths + MSP activation + background column)",
    )
    parser.add_argument(
        "--logits-dir",
        type=Path,
        default=None,
        help="Directory with split .pt files (default: /content/spk/data/{detector}/{dataset}/logits)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Where to write .npz caches (default: <logits-dir>/npz)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="ood_baseline output dir (default: .../data/{detector}/{dataset}/logits_baselines)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ood_baseline.METHODS,
        default=list(ood_baseline.METHODS),
        help="Baselines to run (default: all eight)",
    )
    parser.add_argument("--background-index", type=int, default=None,
                        help="Background logit column (default: from .pt or detector preset)")
    parser.add_argument("--msp-denominator", choices=["all", "foreground"], default="all")
    parser.add_argument("--msp-activation", choices=["softmax", "sigmoid"], default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--scale-percentile", type=float, default=65.0)
    parser.add_argument("--scale-degenerate", choices=["error", "identity"], default="error")
    parser.add_argument("--mds-labels", choices=["argmax", "predicted"], default="argmax")
    parser.add_argument("--covariance", choices=["empirical", "ledoit-wolf"], default="empirical")
    parser.add_argument("--knn-mode", choices=["mhood", "sun"], default="mhood")
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument(
        "--bam-density", type=float, default=None,
        help="BAM cluster density (default: 5 for VOC, 50 for BDD)",
    )
    parser.add_argument("--bam-max-boxes", type=int, default=256)
    parser.add_argument("--bam-cluster", choices=["minibatch", "kmeans"], default="minibatch")
    parser.add_argument("--iforest-scope", choices=["classwise", "global"], default="classwise")
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--iforest-samples", type=int, default=512)
    parser.add_argument("--max-train-per-class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--outlier",
        choices=["none", "mhood-iqr", "score-tail"],
        default="none",
        help="ID-val outlier removal protocol",
    )
    parser.add_argument("--outlier-k", type=int, default=5)
    parser.add_argument("--iqr-factor", type=float, default=1.5)
    parser.add_argument("--tail-fraction", type=float, default=0.05)
    parser.add_argument("--tail-scope", choices=["global", "classwise"], default="classwise")
    parser.add_argument("--force", action="store_true", help="Reconvert .pt and recompute scores")
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only write .npz caches; skip baseline evaluation",
    )
    args = parser.parse_args()

    profile = DATASET_SPLITS[args.dataset]
    detector = DETECTOR_DEFAULTS[args.detector]
    split_files = {
        k: v for k, v in profile.items() if k not in ("bam_density", "background_index")
    }
    if args.logits_dir is None:
        args.logits_dir = Path(f"/content/spk/data/{args.detector}/{args.dataset}/logits")
    if args.out is None:
        args.out = Path(f"/content/spk/data/{args.detector}/{args.dataset}/logits_baselines")
    if args.bam_density is None:
        args.bam_density = float(profile["bam_density"])
    if args.msp_activation is None:
        args.msp_activation = detector["msp_activation"]

    cache_dir = args.cache_dir or (args.logits_dir / "npz")
    npz_paths, pt_background = convert_splits(
        args.logits_dir, cache_dir, split_files, force=args.force
    )
    if args.background_index is None:
        preset = detector["background_index"]
        if isinstance(preset, dict):
            preset = preset.get(args.dataset)
        args.background_index = pt_background if pt_background is not None else preset
    if args.convert_only:
        print("convert-only: done", flush=True)
        return 0

    baseline_args = build_baseline_args(args, npz_paths)
    print(
        f"\n=== ood_baseline ({args.detector}/{args.dataset}) -> {args.out} ===",
        flush=True,
    )
    print(
        f"bam_density={args.bam_density}  background_index={args.background_index}  "
        f"msp_activation={args.msp_activation}",
        flush=True,
    )
    print(f"methods: {', '.join(baseline_args.methods)}", flush=True)
    with threadpool_limits(limits=baseline_args.jobs):
        code = ood_baseline.run(baseline_args)

    report_path = args.out / "report.json"
    summary_path = args.out / "summary.csv"
    print(f"\nreport:  {report_path}", flush=True)
    print(f"summary: {summary_path}", flush=True)
    if summary_path.is_file():
        print("\n=== summary.csv ===", flush=True)
        print(summary_path.read_text(), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
