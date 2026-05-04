from __future__ import annotations

import json

import pandas as pd

from ml.src.wazuh_export.summarize_wazuh_export import summarize_wazuh_export


def test_wazuh_export_summary_counts_top_rules(tmp_path):
    alerts = tmp_path / "alerts.jsonl"
    output_dir = tmp_path / "reports" / "wazuh_export"
    rows = [
        {"timestamp": "2026-05-04T10:00:00Z", "rule": {"id": "5710", "level": 10, "description": "SSH"}},
        {"timestamp": "2026-05-04T10:01:00Z", "rule": {"id": "5710", "level": 10, "description": "SSH"}},
        {"timestamp": "2026-05-04T10:02:00Z", "rule": {"id": "1002", "level": 5, "description": "Scan"}},
    ]
    alerts.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    outputs = summarize_wazuh_export(alerts, output_dir)

    assert outputs["summary"].exists()
    assert outputs["rule_summary"].exists()
    rule_summary = pd.read_csv(outputs["rule_summary"])
    assert rule_summary.iloc[0]["rule_id"] == 5710
    assert rule_summary.iloc[0]["count"] == 2
    assert outputs["timeline"].exists()
