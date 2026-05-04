from __future__ import annotations

import pandas as pd

from ml.src.measurement_quality.check_ttd_quality import run_check


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_ttd_negative_value_fails(tmp_path):
    wazuh = tmp_path / "results/wazuh_real/predictions.csv"
    hybrid = tmp_path / "results/hybrid_real/predictions.csv"
    gt = tmp_path / "data/lab/lab_ground_truth.csv"
    write_csv(wazuh, [{"event_id": "e1", "y_true": 1, "wazuh_pred": 1, "time_to_detection_sec": -1}])
    write_csv(hybrid, [{"event_id": "e1", "time_to_detection_sec": -1}])
    write_csv(gt, [{"event_id": "e1", "label": "attack"}])

    result = run_check(wazuh_predictions=wazuh, hybrid_predictions=hybrid, ground_truth=gt, large_ttd_threshold_sec=300, output_dir=tmp_path / "reports/measurement_quality")

    assert any(row["check_id"] == "ttd_negative" and row["status"] == "FAIL" for row in result["rows"])


def test_ttd_nan_reported_as_no_data(tmp_path):
    wazuh = tmp_path / "results/wazuh_real/predictions.csv"
    hybrid = tmp_path / "results/hybrid_real/predictions.csv"
    gt = tmp_path / "data/lab/lab_ground_truth.csv"
    write_csv(wazuh, [{"event_id": "e1", "y_true": 1, "wazuh_pred": 0, "time_to_detection_sec": ""}])
    write_csv(hybrid, [{"event_id": "e1", "time_to_detection_sec": ""}])
    write_csv(gt, [{"event_id": "e1", "label": "attack"}])

    result = run_check(wazuh_predictions=wazuh, hybrid_predictions=hybrid, ground_truth=gt, large_ttd_threshold_sec=300, output_dir=tmp_path / "reports/measurement_quality")
    text = result["markdown"].read_text(encoding="utf-8")

    assert "nincs adat" in text
