from __future__ import annotations

import pandas as pd
import pytest

from ml.src.lab_ae_eval.evaluate_ae_lab import compute_binary_metrics, evaluate_ae_lab


def make_ae_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"event_id": "e1", "label": "benign", "y_true": 0, "ae_pred": 0},
            {"event_id": "e2", "label": "benign", "y_true": 0, "ae_pred": 1},
            {"event_id": "e3", "label": "attack", "y_true": 1, "ae_pred": 1},
            {"event_id": "e4", "label": "attack", "y_true": 1, "ae_pred": 0},
        ]
    )


def test_compute_ae_lab_metrics():
    df = make_ae_predictions()

    metrics = compute_binary_metrics(df["y_true"], df["ae_pred"])

    assert metrics["TP"] == 1
    assert metrics["FP"] == 1
    assert metrics["TN"] == 1
    assert metrics["FN"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["false_positive_rate"] == pytest.approx(0.5)
    assert metrics["false_negative_rate"] == pytest.approx(0.5)
    assert metrics["alert_count"] == 2


def test_evaluate_ae_lab_writes_outputs(tmp_path):
    input_path = tmp_path / "ae_lab_predictions.csv"
    results_dir = tmp_path / "results" / "ae_lab"
    make_ae_predictions().to_csv(input_path, index=False)

    outputs = evaluate_ae_lab(input_path, results_dir)

    assert outputs["metrics"].exists()
    assert outputs["predictions"].exists()
    assert outputs["confusion_matrix"].exists()
    assert outputs["metadata"].exists()
    metrics = pd.read_csv(outputs["metrics"]).iloc[0]
    assert metrics["TP"] == 1
    assert metrics["FP"] == 1
