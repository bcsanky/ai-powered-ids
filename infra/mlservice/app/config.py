from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServiceConfig:
    model_root: Path
    preprocess_path: Path
    model_version: str


def get_config() -> ServiceConfig:
    return ServiceConfig(
        model_root=Path(os.getenv("AE_MODEL_ROOT", "artifacts/final/final-ae-minimal-v1")),
        preprocess_path=Path(
            os.getenv("AE_PREPROCESS_PATH", "data/processed/final/ae_minimal/preprocess.pkl")
        ),
        model_version=os.getenv("AE_MODEL_VERSION", "final-ae-minimal-v1"),
    )
