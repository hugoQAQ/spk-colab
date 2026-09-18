#!/usr/bin/env python3
"""Check which of the 8 OOD baseline score caches exist."""
from pathlib import Path

EXPECTED = [
    "MaxSigmoid_scores.npz",  # MSP + sigmoid
    "EBO_scores.npz",
    "MLS_scores.npz",
    "MDS-logits_scores.npz",
    "BAM-logits_scores.npz",
    "KNN-logits_scores.npz",
    "iForest-logits_scores.npz",
]
SCALE_ANY = ("SCALE-logits_scores.npz", "SCALE-logits-guarded_scores.npz")

CASES = [
    ("rtdetr/voc", Path("/content/spk/data/rtdetr/voc/logits_baselines")),
    ("rtdetr/voc", Path("/content/drive/MyDrive/experiments/rtdetr-voc/eval_baselines")),
    ("rtdetr/bdd", Path("/content/spk/data/rtdetr/bdd/logits_baselines")),
    ("rtdetr/bdd", Path("/content/drive/MyDrive/experiments/rtdetr-bdd/logits_baselines")),
]

for tag, out in CASES:
    print(f"\n=== {tag}  {out} ===")
    if not out.is_dir():
        print("  MISSING dir")
        continue
    present = {p.name for p in out.glob("*_scores.npz")}
    for name in EXPECTED:
        st = "ok" if name in present else "MISSING"
        print(f"  {st:7} {name}")
    scale = next((n for n in SCALE_ANY if n in present), None)
    print(f"  {'ok' if scale else 'MISSING':7} SCALE ({scale or 'none'})")
    print(f"  logits dir: ", end="")
    ld = Path("/content/spk/data") / tag.replace("/", "/") / "logits"
    print(ld, "exists" if ld.is_dir() else "MISSING")
    if ld.is_dir():
        for f in ("voc-train.pt", "voc-val.pt", "bdd-train.pt", "bdd-val.pt", "near-ood.pt", "far-ood.pt"):
            p = ld / f
            if p.is_file():
                print(f"    logits: {f}")
