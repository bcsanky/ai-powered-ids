from __future__ import annotations

import pandas as pd
import pytest

from ml.src.wazuh_baseline.correlate_alerts import correlate_alerts


def write_ground_truth(path):
    pd.DataFrame(
        [
            {
                "event_id": "evt-attack-1",
                "timestamp_start": "2026-05-11T10:00:00Z",
                "timestamp_end": "2026-05-11T10:01:00Z",
                "scenario": "ssh_bruteforce",
                "label": "attack",
                "attack_type": "ssh_bruteforce",
                "source_ip": "10.0.0.5",
                "target_ip": "10.0.0.10",
            },
            {
                "event_id": "evt-benign-1",
                "timestamp_start": "2026-05-11T10:02:00Z",
                "timestamp_end": "2026-05-11T10:03:00Z",
                "scenario": "benign_activity",
                "label": "benign",
                "attack_type": "",
                "source_ip": "10.0.0.6",
                "target_ip": "10.0.0.20",
            },
            {
                "event_id": "evt-attack-2",
                "timestamp_start": "2026-05-11T10:04:00Z",
                "timestamp_end": "2026-05-11T10:05:00Z",
                "scenario": "port_scan",
                "label": "attack",
                "attack_type": "port_scan",
                "source_ip": "10.0.0.7",
                "target_ip": "10.0.0.30",
            },
        ]
    ).to_csv(path, index=False)


def write_alerts(path):
    pd.DataFrame(
        [
            {
                "timestamp": "2026-05-11T10:01:30Z",
                "rule_id": "5710",
                "rule_level": 10,
                "rule_description": "SSH brute force",
                "agent_name": "agent-1",
                "source_ip": "10.0.0.5",
                "target_ip": "10.0.0.10",
                "full_log": "matched ssh attack",
            },
            {
                "timestamp": "2026-05-11T10:02:30Z",
                "rule_id": "1002",
                "rule_level": 5,
                "rule_description": "Benign event alert",
                "agent_name": "agent-1",
                "source_ip": "10.0.0.6",
                "target_ip": "10.0.0.20",
                "full_log": "benign alert",
            },
            {
                "timestamp": "2026-05-11T10:04:30Z",
                "rule_id": "9999",
                "rule_level": 12,
                "rule_description": "Mismatched IP",
                "agent_name": "agent-1",
                "source_ip": "10.0.0.99",
                "target_ip": "10.0.0.30",
                "full_log": "different source",
            },
        ]
    ).to_csv(path, index=False)


def test_correlate_alerts_uses_time_window_and_ip_relevance(tmp_path):
    ground_truth = tmp_path / "lab_ground_truth.csv"
    alerts = tmp_path / "alerts_parsed.csv"
    output = tmp_path / "wazuh_correlated.csv"
    write_ground_truth(ground_truth)
    write_alerts(alerts)

    correlated = correlate_alerts(
        ground_truth_path=ground_truth,
        alerts_path=alerts,
        output_path=output,
        window_seconds=60,
    )

    assert output.exists()
    attack_1 = correlated[correlated["event_id"] == "evt-attack-1"].iloc[0]
    assert attack_1["y_true"] == 1
    assert attack_1["wazuh_pred"] == 1
    assert attack_1["first_alert_time"] == "2026-05-11T10:01:30Z"
    assert attack_1["time_to_detection_sec"] == pytest.approx(90.0)
    assert attack_1["matched_rule_ids"] == "5710"
    assert attack_1["max_rule_level"] == 10
    assert attack_1["alert_count"] == 1

    benign = correlated[correlated["event_id"] == "evt-benign-1"].iloc[0]
    assert benign["y_true"] == 0
    assert benign["wazuh_pred"] == 1

    attack_2 = correlated[correlated["event_id"] == "evt-attack-2"].iloc[0]
    assert attack_2["y_true"] == 1
    assert attack_2["wazuh_pred"] == 0
    assert attack_2["alert_count"] == 0


def test_correlate_alerts_rejects_negative_window(tmp_path):
    ground_truth = tmp_path / "lab_ground_truth.csv"
    alerts = tmp_path / "alerts_parsed.csv"
    output = tmp_path / "wazuh_correlated.csv"
    write_ground_truth(ground_truth)
    write_alerts(alerts)

    with pytest.raises(ValueError, match="window_seconds"):
        correlate_alerts(
            ground_truth_path=ground_truth,
            alerts_path=alerts,
            output_path=output,
            window_seconds=-1,
        )
