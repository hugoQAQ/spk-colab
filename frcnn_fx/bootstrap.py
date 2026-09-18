"""Register bundled Detectron2 FX Faster R-CNN extensions for inference."""
from __future__ import annotations

import sys
from pathlib import Path

_FX_DIR: Path | None = None


def frcnn_fx_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_frcnn_fx_dir(root: Path | None = None) -> Path:
    """Locate bundled ``frcnn_fx`` (git clone or mounted layout)."""
    root = Path(root).resolve() if root is not None else frcnn_fx_dir().parent
    here = frcnn_fx_dir()
    for candidate in (
        here,
        root / "model" / "frcnn" / "frcnn_fx",
        root / "frcnn_fx",
    ):
        if (candidate / "FX_vanilla_voc.yaml").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "frcnn_fx not found. Expected bundled copy next to frcnn_logits_extraction.py "
        f"or under {root / 'model/frcnn/frcnn_fx'}"
    )


def config_path(root: Path | None, dataset: str) -> Path:
    fx_dir = resolve_frcnn_fx_dir(root)
    name = f"FX_vanilla_{dataset}.yaml"
    path = fx_dir / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def setup_frcnn_fx(root: Path | None = None) -> Path:
    """Add ``frcnn_fx`` to ``sys.path`` and register ``FXGeneralizedRCNN``."""
    global _FX_DIR
    fx_dir = resolve_frcnn_fx_dir(root)
    fx_str = str(fx_dir)
    if fx_str not in sys.path:
        sys.path.insert(0, fx_str)
    if _FX_DIR != fx_dir:
        import utils.fxrcnn  # noqa: F401

        _FX_DIR = fx_dir
    return fx_dir
