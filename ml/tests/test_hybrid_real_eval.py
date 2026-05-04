from __future__ import annotations

import pandas as pd
import pytest

from ml.src.hybrid_real.evaluate_hybrid_real import (
    build_hybrid_predictions,
    evaluate_hybrid_real,
)


def make_wazuh_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "e1",
                "label": "benign",
                "y_true": 0,
                "wazuh_pred": 0,
                "max_rule_level": 0,
                "time_to_detection_sec": "",
            },
            {
                "event_id": "e2",
                "label": "attack",
                "y_true": 1,
                "wazuh_pred": 1,
                "max_rule_level": 15,
                "time_to_detection_sec": 4.0,
            },
            {
                "event_id": "e3",
                "label": "attack",
                "y_true": 1,
                "wazuh_pred": 0,
                "max_rule_level": 0,
                "time_to_detection_sec": "",
            },
            {
                "event_id": "e4",
                "label": "benign",
                "y_true": 0,
                "wazuh_pred": 1,
                "max_rule_level": 10,
                "time_to_detection_sec": 2.0,
            },
        ]
    )


def make_ae_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"event_id": "e1", "label": "benign", "y_true": 0, "ae_pred": 0, "anomaly_score": 0.0},
            {"event_id": "e2", "label": "attack", "y_true": 1, "ae_pred": 0, "anomaly_score": 10.0},
            {"event_id": "e3", "label": "attack", "y_true": 1, "ae_pred": 1, "anomaly_score": 5.0},
            {"event_id": "e4", "label": "benign", "y_true": 0, "ae_pred": 0, "anomaly_score": 2.0},
        ]
    )


def test_hybrid_real_strategies_are_deterministic():
    out = build_hybrid_predictions(
        make_wazuh_predictions(),
        make_ae_predictions(),
        weighted_threshold=0.5,
    )

    assert out["hybrid_or_pred"].tolist() == [0, 1, 1, 1]
    assert out["hybrid_weighted_pred"].tolist() == [0, 1, 0, 0]
    assert out["hybrid_priority_level"].tolist() == ["normal", "high", "medium", "high"]
    assert out["hybrid_priority_pred"].tolist() == [0, 1, 1, 1]
    assert out.loc[out["event_id"] == "e2", "hybrid_weighted_score"].iloc[0] == pytest.approx(1.0)


def test_hybrid_real_rejects_label_mismatch():
    ae = make_ae_predictions()
    ae.loc[ae["event_id"] == "e2", "y_true"] = 0

    with pytest.raises(ValueError, match="y_true"):
        build_hybrid_predictions(make_wazuh_predictions(), ae)


def test_evaluate_hybrid_real_writes_strategy_metrics(tmp_path):
    wazuh_path = tmp_path / "wazuh_predictions.csv"
    ae_path = tmp_path / "ae_predictions.csv"
    results_dir = tmp_path / "results" / "hybrid_real"
    make_wazuh_predictions().to_csv(wazuh_path, index=False)
    make_ae_predictions().to_csv(ae_path, index=False)

    outputs = evaluate_hybrid_real(
        wazuh_predictions_path=wazuh_path,
        ae_predictions_path=ae_path,
        results_dir=results_dir,
        weighted_threshold=0.5,
    )

    assert outputs["metrics"].exists()
    assert outputs["predictions"].exists()
    metrics = pd.read_csv(outputs["metrics"])
    assert set(metrics["strategy"]) == {"hybrid_or", "hybrid_weighted", "hybrid_priority"}
    assert {"precision", "recall", "f1", "alert_count", "mean_ttd"}.issubset(metrics.columns)
