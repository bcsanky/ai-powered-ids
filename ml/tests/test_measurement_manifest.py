from __future__ import annotations

import hashlib

import pandas as pd

from ml.src.real_measurement.create_measurement_manifest import create_manifest


def test_measurement_manifest_hashes_and_submission_flags(tmp_path):
    alerts = tmp_path / "data/wazuh/alerts.jsonl"
    metrics = tmp_path / "results/real_comparison/metrics_comparison.csv"
    report = tmp_path / "reports/real_measurement/real_lab_results_report.md"
    figure = tmp_path / "results/real_comparison/fig_alert_count.png"
    for path, content in [
        (alerts, "alert\n"),
        (metrics, "configuration,f1\nWazuh-only,0.5\n"),
        (report, "# Report\n"),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    figure.parent.mkdir(parents=True, exist_ok=True)
    figure.write_bytes(b"png")

    outputs = create_manifest(tmp_path, tmp_path / "reports/real_measurement")

    manifest = pd.read_csv(outputs["csv"])
    metric_row = manifest[manifest["relative_path"] == "results/real_comparison/metrics_comparison.csv"].iloc[0]
    alert_row = manifest[manifest["relative_path"] == "data/wazuh/alerts.jsonl"].iloc[0]
    expected_hash = hashlib.sha256(metrics.read_bytes()).hexdigest()
    assert metric_row["sha256"] == expected_hash
    assert metric_row["provenance_status"] == "missing_provenance"
    assert bool(metric_row["include_in_submission"]) is False
    assert bool(alert_row["include_in_submission"]) is False
    assert outputs["markdown"].exists()
    assert outputs["json"].exists()
