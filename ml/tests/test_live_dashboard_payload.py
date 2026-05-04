from __future__ import annotations

import json

import pandas as pd

from ml.src.live_integration.generate_dashboard_payload import generate_dashboard_payload


def test_dashboard_payload_uses_enriched_csv_values(tmp_path):
    output_dir = tmp_path / "reports/live_integration"
    output_dir.mkdir(parents=True)
    input_path = output_dir / "enriched_alerts.csv"
    pd.DataFrame(
        [
            {
                "top_level_status": "scored",
                "ml_alert": True,
                "wazuh_pred": 1,
                "hybrid_or_pred": 1,
                "risk_level": "critical",
                "rule_id": "5710",
                "source_ip": "10.0.0.2",
                "scenario": "port_scan",
            },
            {
                "top_level_status": "unmatched",
                "ml_alert": False,
                "wazuh_pred": 1,
                "hybrid_or_pred": 1,
                "risk_level": "high",
                "rule_id": "5710",
                "source_ip": "10.0.0.3",
                "scenario": "",
            },
        ]
    ).to_csv(input_path, index=False)

    outputs = generate_dashboard_payload(input_path, output_dir)
    payload = json.loads(outputs["payload"].read_text(encoding="utf-8"))
    cards = {card["metric"]: card["value"] for card in payload["cards"]}

    assert cards["total_alerts"] == 2
    assert cards["scored_alerts"] == 1
    assert cards["unmatched_alerts"] == 1
    assert cards["ml_positive"] == 1
    assert cards["wazuh_positive"] == 2
    assert payload["top_rule_ids"][0] == {"value": "5710", "count": 2}

