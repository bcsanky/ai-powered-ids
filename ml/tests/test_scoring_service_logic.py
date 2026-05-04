from __future__ import annotations

from pathlib import Path

import pytest

from ml.src.scoring_runtime import AEScorer, determine_risk_level, select_threshold


def test_threshold_priority_order():
    choice = select_threshold(
        {
            "fixed": 0.3,
            "percentile_95": 0.2,
            "f1_optimum": {"threshold": 0.1},
        }
    )
    assert choice.name == "f1_optimum"
    assert choice.value == 0.1

    choice = select_threshold({"fixed": 0.3, "percentile_95": 0.2})
    assert choice.name == "percentile_95"
    assert choice.value == 0.2

    choice = select_threshold({"fixed": 0.3})
    assert choice.name == "fixed"
    assert choice.value == 0.3


def test_risk_level_logic():
    assert determine_risk_level(
        anomaly_score=0.1,
        threshold_value=1.0,
        ml_alert=False,
        rule_flag=False,
        rule_level=0,
    )[0] == "normal"

    assert determine_risk_level(
        anomaly_score=1.1,
        threshold_value=1.0,
        ml_alert=True,
        rule_flag=False,
        rule_level=0,
    )[0] == "medium"

    assert determine_risk_level(
        anomaly_score=0.1,
        threshold_value=1.0,
        ml_alert=False,
        rule_flag=True,
        rule_level=5,
    )[0] == "medium"

    assert determine_risk_level(
        anomaly_score=1.1,
        threshold_value=1.0,
        ml_alert=True,
        rule_flag=True,
        rule_level=5,
    )[0] == "critical"

    assert determine_risk_level(
        anomaly_score=0.1,
        threshold_value=1.0,
        ml_alert=False,
        rule_flag=False,
        rule_level=10,
    )[0] == "high"


def test_scorer_does_not_return_dummy_score_when_model_is_missing(tmp_path):
    scorer = AEScorer(
        model_root=tmp_path / "missing_model_root",
        preprocess_path=tmp_path / "missing_preprocess.pkl",
    )

    with pytest.raises(FileNotFoundError):
        scorer.score_event(
            event_id="demo",
            features={
                "destination_port": 80,
                "flow_duration": 1,
                "total_fwd_packets": 1,
                "total_backward_packets": 1,
                "flow_bytes_per_sec": 1.0,
                "flow_packets_per_sec": 1.0,
                "protocol": "6",
            },
        )
