from __future__ import annotations

import pandas as pd

from ml.src.measurement_quality.check_metric_consistency import run_check


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def metric_row(**kwargs):
    row = {
        "TP": 1,
        "FP": 0,
        "TN": 1,
        "FN": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "false_positive_rate": 0.0,
        "false_negative_rate": 0.0,
        "alert_count": 1,
        "n_samples": 2,
        "n_attack": 1,
        "n_benign": 1,
        "mean_ttd": 3.0,
        "median_ttd": 3.0,
    }
    row.update(kwargs)
    return row


def test_metric_consistency_valid_hybrid_or_logic(tmp_path):
    wazuh = tmp_path / "results/wazuh_real/metrics_summary.csv"
    ae = tmp_path / "results/ae_lab/metrics_summary.csv"
    hybrid = tmp_path / "results/hybrid_real/metrics_summary.csv"
    comparison = tmp_path / "results/real_comparison/metrics_comparison.csv"
    wazuh_pred = tmp_path / "results/wazuh_real/predictions.csv"
    ae_pred = tmp_path / "results/ae_lab/predictions.csv"
    hybrid_pred = tmp_path / "results/hybrid_real/predictions.csv"
    write_csv(wazuh, [metric_row()])
    write_csv(ae, [metric_row()])
    write_csv(hybrid, [metric_row(strategy="hybrid_or"), metric_row(strategy="hybrid_weighted"), metric_row(strategy="hybrid_priority")])
    write_csv(
        comparison,
        [
            {"configuration": "Wazuh-only", **metric_row()},
            {"configuration": "AE-Minimal lab", **metric_row()},
            {"configuration": "Hybrid OR", **metric_row()},
            {"configuration": "Hybrid weighted", **metric_row()},
            {"configuration": "Hybrid priority", **metric_row()},
        ],
    )
    write_csv(wazuh_pred, [{"event_id": "e1", "y_true": 1, "wazuh_pred": 1}])
    write_csv(ae_pred, [{"event_id": "e1", "y_true": 1, "ae_pred": 0, "anomaly_score": 0.2}])
    write_csv(hybrid_pred, [{"event_id": "e1", "wazuh_pred": 1, "ae_pred": 0, "hybrid_or_pred": 1}])

    result = run_check(
        wazuh_metrics=wazuh,
        ae_metrics=ae,
        hybrid_metrics=hybrid,
        comparison_path=comparison,
        wazuh_predictions=wazuh_pred,
        ae_predictions=ae_pred,
        hybrid_predictions=hybrid_pred,
        output_dir=tmp_path / "reports/measurement_quality",
    )

    assert any(row["check_id"] == "hybrid_or_logic" and row["status"] == "PASS" for row in result["rows"])

