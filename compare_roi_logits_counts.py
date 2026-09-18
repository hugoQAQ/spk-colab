#!/usr/bin/env python3
"""Compare ROI cache vs logits extraction detection counts (Colab / Drive, no download).

Reads only torch.save headers (drops tensor payloads) so large ROI files stay on disk.

Examples
--------
    python compare_roi_logits_counts.py
    python compare_roi_logits_counts.py --splits near_ood far_ood
    python compare_roi_logits_counts.py --detector frcnn --dataset voc
"""
from __future__ import annotations

import argparse
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DRIVE = Path("/content/drive/MyDrive/experiments")
SPK = Path("/content/spk/data")


@dataclass(frozen=True)
class SplitPair:
    roi_key: str          # near_ood, far_ood, id_train, id_val
    logits_name: str      # near-ood.pt, voc-train.pt, ...
    label: str            # display label


@dataclass(frozen=True)
class Experiment:
    detector: str
    dataset: str
    roi_dir_candidates: tuple[str, ...]
    logits_dir_candidates: tuple[str, ...]
    splits: tuple[SplitPair, ...]


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        "yolo", "voc",
        roi_dir_candidates=(
            str(DRIVE / "yolo-voc/roi"),
            str(SPK / "yolo/voc/roi"),
        ),
        logits_dir_candidates=(
            str(DRIVE / "yolo-voc/logits"),
            str(SPK / "yolo/voc/logits"),
        ),
        splits=(
            SplitPair("id_train", "voc-train.pt", "train"),
            SplitPair("id_val", "voc-val.pt", "id_val"),
            SplitPair("near_ood", "near-ood.pt", "near_ood"),
            SplitPair("far_ood", "far-ood.pt", "far_ood"),
        ),
    ),
    Experiment(
        "yolo", "bdd",
        roi_dir_candidates=(
            str(DRIVE / "yolo-bdd/roi"),
            str(SPK / "yolo/bdd/roi"),
        ),
        logits_dir_candidates=(
            str(DRIVE / "yolo-bdd/logits"),
            str(SPK / "yolo/bdd/logits"),
        ),
        splits=(
            SplitPair("id_train", "bdd-train.pt", "train"),
            SplitPair("id_val", "bdd-val.pt", "id_val"),
            SplitPair("near_ood", "near-ood.pt", "near_ood"),
            SplitPair("far_ood", "far-ood.pt", "far_ood"),
        ),
    ),
    Experiment(
        "frcnn", "voc",
        roi_dir_candidates=(
            str(DRIVE / "frcnn-voc/roi/roi"),
            str(DRIVE / "frcnn-voc/roi"),
            str(SPK / "frcnn/voc/roi"),
        ),
        logits_dir_candidates=(
            str(DRIVE / "frcnn-voc/logits/logits"),
            str(DRIVE / "frcnn-voc/logits"),
            str(SPK / "frcnn/voc/logits"),
        ),
        splits=(
            SplitPair("id_train", "voc-train.pt", "train"),
            SplitPair("id_val", "voc-val.pt", "id_val"),
            SplitPair("near_ood", "near-ood.pt", "near_ood"),
            SplitPair("far_ood", "far-ood.pt", "far_ood"),
        ),
    ),
    Experiment(
        "frcnn", "bdd",
        roi_dir_candidates=(
            str(DRIVE / "frcnn-bdd/roi"),
            str(SPK / "frcnn/bdd/roi"),
        ),
        logits_dir_candidates=(
            str(DRIVE / "frcnn-bdd/logits"),
            str(SPK / "frcnn/bdd/logits"),
        ),
        splits=(
            SplitPair("id_train", "bdd-train.pt", "train"),
            SplitPair("id_val", "bdd-val.pt", "id_val"),
            SplitPair("near_ood", "near-ood.pt", "near_ood"),
            SplitPair("far_ood", "far-ood.pt", "far_ood"),
        ),
    ),
)


class _Drop:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    def __call__(self, *args, **kwargs):
        return _Drop()


class _HeaderUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        return _Drop()

    def find_class(self, module, name):
        if module.startswith(("torch", "numpy")):
            return _Drop
        return super().find_class(module, name)


def torch_header(path: Path) -> dict[str, Any] | None:
    try:
        with zipfile.ZipFile(path) as zf:
            name = next((n for n in zf.namelist() if n.endswith("data.pkl")), None)
            if name is None:
                return None
            with zf.open(name) as fh:
                obj = _HeaderUnpickler(fh).load()
        return obj if isinstance(obj, dict) else None
    except (zipfile.BadZipFile, pickle.UnpicklingError, OSError, EOFError, StopIteration):
        return None


def first_file(candidates: tuple[str, ...], *parts: str) -> Path | None:
    for base in candidates:
        path = Path(base, *parts)
        if path.is_file():
            return path
    return None


def roi_counts(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    header = torch_header(path)
    if header is None:
        return {"status": "unreadable", "path": str(path)}
    info: dict[str, Any] = {"status": "ok", "path": str(path)}
    if header.get("num_detections") is not None:
        info["num_detections"] = int(header["num_detections"])
    if header.get("num_images") is not None:
        info["num_images"] = int(header["num_images"])
    if "num_detections" not in info:
        meta = header.get("metadata")
        if isinstance(meta, list):
            info["num_detections"] = len(meta)
    if "num_detections" not in info and "logits" in header:
        logits = header["logits"]
        if hasattr(logits, "shape"):
            info["num_detections"] = int(logits.shape[0])
    return info


def logits_counts(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    header = torch_header(path)
    if header is None:
        return {"status": "unreadable", "path": str(path)}
    info: dict[str, Any] = {"status": "ok", "path": str(path)}
    if header.get("num_detections") is not None:
        info["num_detections"] = int(header["num_detections"])
    elif "logits" in header:
        logits = header["logits"]
        if hasattr(logits, "shape"):
            info["num_detections"] = int(logits.shape[0])
    if header.get("num_images") is not None:
        info["num_images"] = int(header["num_images"])
    if header.get("num_classes") is not None:
        info["num_classes"] = int(header["num_classes"])
    return info


def fmt_count(info: dict[str, Any]) -> str:
    if info.get("status") != "ok":
        return info.get("status", "?")
    parts = [str(info.get("num_detections", "?"))]
    if info.get("num_images") is not None:
        parts.append(f"img={info['num_images']}")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", choices=("yolo", "frcnn"), default=None)
    parser.add_argument("--dataset", choices=("voc", "bdd"), default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="ROI split keys: id_train id_val near_ood far_ood (default: all)",
    )
    args = parser.parse_args()

    exps = [
        e for e in EXPERIMENTS
        if (args.detector is None or e.detector == args.detector)
        and (args.dataset is None or e.dataset == args.dataset)
    ]
    split_filter = set(args.splits) if args.splits else None

    print("compare_roi_logits_counts  (header-only, no tensor download)\n")
    mismatches = 0
    compared = 0

    for exp in exps:
        print(f"=== {exp.detector}/{exp.dataset} ===")
        print(f"{'split':10} {'ROI dets':>12} {'logits dets':>12} {'match':>8}  notes")
        print("-" * 72)
        for sp in exp.splits:
            if split_filter and sp.roi_key not in split_filter:
                continue
            roi_path = first_file(exp.roi_dir_candidates, f"{sp.roi_key}.pt")
            logits_path = first_file(exp.logits_dir_candidates, sp.logits_name)
            roi = roi_counts(roi_path) if roi_path else {"status": "missing"}
            logits = logits_counts(logits_path) if logits_path else {"status": "missing"}

            if roi.get("status") == "ok" and logits.get("status") == "ok":
                compared += 1
                rd = int(roi["num_detections"])
                ld = int(logits["num_detections"])
                match = rd == ld
                if not match:
                    mismatches += 1
                flag = "yes" if match else f"NO Δ{ld - rd:+d}"
                notes = []
                if roi.get("num_images") is not None:
                    notes.append(f"roi_img={roi['num_images']}")
                if logits.get("num_images") is not None:
                    notes.append(f"logits_img={logits['num_images']}")
                print(f"{sp.label:10} {rd:12d} {ld:12d} {flag:>8}  {' '.join(notes)}")
            else:
                roi_s = fmt_count(roi) if roi.get("status") == "ok" else roi.get("status", "?")
                log_s = fmt_count(logits) if logits.get("status") == "ok" else logits.get("status", "?")
                print(f"{sp.label:10} {roi_s:>12} {log_s:>12} {'—':>8}")
                if roi.get("status") == "missing":
                    print(f"           roi:    (not found under {exp.roi_dir_candidates[0]})")
                if logits.get("status") == "missing":
                    print(f"           logits: (not found under {exp.logits_dir_candidates[0]})")
        print()

    print(f"compared={compared}  mismatches={mismatches}")
    if mismatches:
        print(
            "Note: ROI (run.py) and logits (logits_extraction.py) use the same detector "
            "forward + max_det=30 but are separate runs; small diffs are unexpected, "
            "large diffs usually mean different checkpoints, splits, or script versions."
        )


if __name__ == "__main__":
    main()
