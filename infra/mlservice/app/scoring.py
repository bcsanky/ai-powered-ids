from __future__ import annotations

from ml.src.scoring_runtime import AEScorer

from app.config import get_config


def build_scorer() -> AEScorer:
    cfg = get_config()
    return AEScorer(
        model_root=cfg.model_root,
        preprocess_path=cfg.preprocess_path,
        model_version=cfg.model_version,
    )
