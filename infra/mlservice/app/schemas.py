from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    event_id: str
    features: dict[str, Any]
    rule_flag: bool = False
    rule_level: int = 0


class ScoreResponse(BaseModel):
    event_id: str
    model_loaded: bool
    model_version: str
    anomaly_score: float
    threshold_name: str
    threshold_value: float
    ml_alert: bool
    rule_flag: bool
    rule_level: int
    risk_level: Literal["normal", "medium", "high", "critical"]
    reason: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    model_root: str
    preprocess_path: str
    detail: str | None = Field(default=None)
