from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.src.autoencoder import reconstruct, sample_reconstruction_scores


REQUIRED_FEATURES = [
    "destination_port",
    "flow_duration",
    "total_fwd_packets",
    "total_backward_packets",
    "flow_bytes_per_sec",
    "flow_packets_per_sec",
    "protocol",
]


@dataclass
class ThresholdChoice:
    name: str
    value: float


@dataclass
class ScoringPaths:
    model_path: Path
    preprocess_path: Path
    thresholds_path: Path
    run_dir: Path


@dataclass
class ScoringResult:
    event_id: str
    model_loaded: bool
    model_version: str
    anomaly_score: float
    threshold_name: str
    threshold_value: float
    ml_alert: bool
    rule_flag: bool
    rule_level: int
    risk_level: str
    reason: str


class ScoringModelUnavailable(RuntimeError):
    pass


def find_latest_run_dir(model_root: Path) -> Path:
    if not model_root.exists():
        raise FileNotFoundError(f"Hiányzó modellgyökér: {model_root}")

    candidates = [
        path
        for path in model_root.iterdir()
        if path.is_dir() and (path / "model.joblib").exists() and (path / "thresholds.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"Nem található használható AE futtatási könyvtár itt: {model_root}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def resolve_scoring_paths(
    model_root: Path,
    preprocess_path: Path,
    thresholds_path: Path | None = None,
) -> ScoringPaths:
    run_dir = find_latest_run_dir(model_root)
    resolved_thresholds = thresholds_path or (run_dir / "thresholds.json")
    return ScoringPaths(
        model_path=run_dir / "model.joblib",
        preprocess_path=preprocess_path,
        thresholds_path=resolved_thresholds,
        run_dir=run_dir,
    )


def select_threshold(thresholds: dict[str, Any]) -> ThresholdChoice:
    if "f1_optimum" in thresholds:
        value = thresholds["f1_optimum"]
        if isinstance(value, dict):
            value = value.get("threshold")
        if value is not None:
            return ThresholdChoice("f1_optimum", float(value))

    if "percentile_95" in thresholds:
        return ThresholdChoice("percentile_95", float(thresholds["percentile_95"]))

    if "fixed" in thresholds:
        return ThresholdChoice("fixed", float(thresholds["fixed"]))

    raise ValueError("Nem található használható küszöb: f1_optimum, percentile_95 vagy fixed szükséges.")


def validate_features(features: dict[str, Any]) -> None:
    missing = [name for name in REQUIRED_FEATURES if name not in features]
    if missing:
        raise ValueError(f"Hiányzó bemeneti feature mezők: {missing}")


def event_to_dataframe(features: dict[str, Any]) -> pd.DataFrame:
    validate_features(features)
    row = {name: features[name] for name in REQUIRED_FEATURES}
    return pd.DataFrame([row])


def determine_risk_level(
    *,
    anomaly_score: float,
    threshold_value: float,
    ml_alert: bool,
    rule_flag: bool,
    rule_level: int,
) -> tuple[str, str]:
    high_score = anomaly_score >= 1.5 * threshold_value

    if rule_flag and ml_alert:
        return "critical", "Szabályalapú jelzés és ML riasztás egyszerre jelentkezett."
    if rule_level >= 10:
        return "high", "Magas szabályszintű jelzés érkezett."
    if ml_alert and high_score:
        return "high", "Az anomáliapontszám jelentősen meghaladja a kiválasztott küszöböt."
    if ml_alert:
        return "medium", "Az ML modell anomáliát jelzett a kiválasztott küszöb alapján."
    if rule_flag:
        return "medium", "Szabályalapú jelzés érkezett ML riasztás nélkül."
    return "normal", "Nem jelentkezett ML vagy szabályalapú riasztás."


class AEScorer:
    def __init__(
        self,
        *,
        model_root: Path,
        preprocess_path: Path,
        thresholds_path: Path | None = None,
        model_version: str = "final-ae-minimal-v1",
    ) -> None:
        self.model_root = model_root
        self.preprocess_path = preprocess_path
        self.thresholds_path = thresholds_path
        self.model_version = model_version
        self.paths: ScoringPaths | None = None
        self.model = None
        self.preprocess = None
        self.threshold: ThresholdChoice | None = None

    @property
    def model_loaded(self) -> bool:
        return self.model is not None and self.preprocess is not None and self.threshold is not None

    def load(self) -> None:
        paths = resolve_scoring_paths(self.model_root, self.preprocess_path, self.thresholds_path)
        if not paths.preprocess_path.exists():
            raise FileNotFoundError(f"Hiányzó preprocess fájl: {paths.preprocess_path}")
        if not paths.thresholds_path.exists():
            raise FileNotFoundError(f"Hiányzó küszöbfájl: {paths.thresholds_path}")

        with paths.thresholds_path.open("r", encoding="utf-8") as f:
            thresholds = json.load(f)

        self.model = joblib.load(paths.model_path)
        self.preprocess = joblib.load(paths.preprocess_path)
        self.threshold = select_threshold(thresholds)
        self.paths = paths

    def ensure_loaded(self) -> None:
        if not self.model_loaded:
            self.load()

    def score_event(
        self,
        *,
        event_id: str,
        features: dict[str, Any],
        rule_flag: bool = False,
        rule_level: int = 0,
    ) -> ScoringResult:
        self.ensure_loaded()
        if not self.model_loaded or self.threshold is None or self.model is None or self.preprocess is None:
            raise ScoringModelUnavailable("A scoring modell nem érhető el.")

        x_raw = event_to_dataframe(features)
        x = self.preprocess.transform(x_raw)
        x_dense = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        x_recon = reconstruct(self.model, x_dense)
        score = float(sample_reconstruction_scores(x_dense, x_recon)[0])
        ml_alert = bool(score >= self.threshold.value)
        normalized_rule_level = int(rule_level or 0)
        normalized_rule_flag = bool(rule_flag)
        risk_level, reason = determine_risk_level(
            anomaly_score=score,
            threshold_value=self.threshold.value,
            ml_alert=ml_alert,
            rule_flag=normalized_rule_flag,
            rule_level=normalized_rule_level,
        )
        return ScoringResult(
            event_id=event_id,
            model_loaded=True,
            model_version=self.model_version,
            anomaly_score=score,
            threshold_name=self.threshold.name,
            threshold_value=self.threshold.value,
            ml_alert=ml_alert,
            rule_flag=normalized_rule_flag,
            rule_level=normalized_rule_level,
            risk_level=risk_level,
            reason=reason,
        )


def result_to_dict(result: ScoringResult) -> dict[str, Any]:
    return {
        "event_id": result.event_id,
        "model_loaded": result.model_loaded,
        "model_version": result.model_version,
        "anomaly_score": result.anomaly_score,
        "threshold_name": result.threshold_name,
        "threshold_value": result.threshold_value,
        "ml_alert": result.ml_alert,
        "rule_flag": result.rule_flag,
        "rule_level": result.rule_level,
        "risk_level": result.risk_level,
        "reason": result.reason,
    }
