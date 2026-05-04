from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.src.real_measurement.validate_measurement_bundle import validate_measurement_bundle


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def create_bundle(root: Path, *, missing_comparison_config: bool = False):
    write_csv(
        root / "data/lab/lab_ground_truth.csv",
        [
            {"event_id": "e1", "label": "benign"},
            {"event_id": "e2", "label": "attack"},
        ],
    )
    write_csv(root / "data/lab/lab_features.csv", [{"event_id": "e1"}, {"event_id": "e2"}])
    alerts = root / "data/wazuh/alerts.jsonl"
    alerts.parent.mkdir(parents=True, exist_ok=True)
    alerts.write_text(json.dumps({"timestamp": "2026-05-04T10:00:00Z"}) + "\n", encoding="utf-8")
    for rel in [
        "data/processed/wazuh_real/alerts_parsed.csv",
        "data/processed/wazuh_real/wazuh_correlated.csv",
        "data/processed/lab_ae/ae_lab_predictions.csv",
    ]:
        write_csv(root / rel, [{"event_id": "e1"}])
    for rel in [
        "results/wazuh_real/metrics_summary.csv",
        "results/ae_lab/metrics_summary.csv",
    ]:
        write_csv(root / rel, [{"n_samples": 2}])
    write_csv(
        root / "results/hybrid_real/metrics_summary.csv",
        [{"strategy": "hybrid_or", "n_samples": 2}],
    )
    configs = ["Wazuh-only", "AE-Minimal lab", "Hybrid OR", "Hybrid weighted", "Hybrid priority"]
    if missing_comparison_config:
        configs = configs[:-1]
    write_csv(
        root / "results/real_comparison/metrics_comparison.csv",
        [{"configuration": config, "n_samples": 2} for config in configs],
    )
    (root / "results/real_comparison/metrics_comparison.md").parent.mkdir(parents=True, exist_ok=True)
    (root / "results/real_comparison/metrics_comparison.md").write_text("| ok |\n", encoding="utf-8")
    for name in ["fig_precision_recall_f1.png", "fig_false_positive_rate.png", "fig_alert_count.png"]:
        path = root / "results/real_comparison" / name
        path.write_bytes(b"png")


def test_bundle_validation_pass(tmp_path):
    create_bundle(tmp_path)

    result = validate_measurement_bundle(tmp_path, tmp_path / "reports/real_measurement")

    assert result["overall_status"] == "PASS"
    assert result["summary"].exists()
    assert result["report"].exists()


def test_bundle_validation_reports_fail(tmp_path):
    create_bundle(tmp_path, missing_comparison_config=True)

    result = validate_measurement_bundle(tmp_path, tmp_path / "reports/real_measurement")

    assert result["overall_status"] == "FAIL"
    summary = pd.read_csv(result["summary"])
    assert "FAIL" in summary["status"].tolist()
