from __future__ import annotations

import pandas as pd
import pytest

from ml.src.wazuh_baseline.evaluate_wazuh_baseline import compute_wazuh_metrics, save_outputs


def make_correlated_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "tp",
                "label": "attack",
                "y_true": 1,
                "wazuh_pred": 1,
                "time_to_detection_sec": 5.0,
                "alert_count": 2,
            },
            {
                "event_id": "fn",
                "label": "attack",
                "y_true": 1,
                "wazuh_pred": 0,
                "time_to_detection_sec": "",
                "alert_count": 0,
            },
            {
                "event_id": "fp",
                "label": "benign",
                "y_true": 0,
                "wazuh_pred": 1,
                "time_to_detection_sec": 3.0,
                "alert_count": 1,
            },
            {
                "event_id": "tn",
                "label": "benign",
                "y_true": 0,
                "wazuh_pred": 0,
                "time_to_detection_sec": "",
                "alert_count": 0,
            },
        ]
    )


def test_compute_wazuh_metrics():
    metrics = compute_wazuh_metrics(make_correlated_predictions())

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
    assert metrics["raw_alert_count"] == 3
    assert metrics["mean_ttd"] == pytest.approx(5.0)
    assert metrics["median_ttd"] == pytest.approx(5.0)


def test_save_outputs_writes_wazuh_result_files(tmp_path):
    input_path = tmp_path / "wazuh_correlated.csv"
    results_dir = tmp_path / "results" / "wazuh_real"
    make_correlated_predictions().to_csv(input_path, index=False)

    outputs = save_outputs(input_path, results_dir)

    assert outputs["metrics"].exists()
    assert outputs["predictions"].exists()
    assert outputs["confusion_matrix"].exists()
    assert outputs["metadata"].exists()
    metrics = pd.read_csv(outputs["metrics"]).iloc[0]
    assert metrics["TP"] == 1
    assert metrics["FP"] == 1
    assert metrics["TN"] == 1
    assert metrics["FN"] == 1
