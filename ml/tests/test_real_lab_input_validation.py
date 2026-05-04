from __future__ import annotations

import json

import pandas as pd
import pytest

from ml.src.lab_features.validate_real_lab_inputs import validate_real_lab_inputs


def write_ground_truth(path):
    pd.DataFrame(
        [
            {
                "event_id": "lab-001",
                "timestamp_start": "2026-05-11T10:00:00Z",
                "timestamp_end": "2026-05-11T10:01:00Z",
                "scenario": "benign_ssh_login",
                "label": "benign",
                "attack_type": "",
                "source_ip": "192.168.56.20",
                "target_ip": "192.168.56.10",
            },
            {
                "event_id": "lab-002",
                "timestamp_start": "2026-05-11T10:02:00Z",
                "timestamp_end": "2026-05-11T10:03:00Z",
                "scenario": "port_scan",
                "label": "attack",
                "attack_type": "port_scan",
                "source_ip": "192.168.56.20",
                "target_ip": "192.168.56.10",
            },
        ]
    ).to_csv(path, index=False)


def write_features(path, ids=None):
    ids = ids or ["lab-001", "lab-002"]
    rows = []
    for event_id in ids:
        rows.append(
            {
                "event_id": event_id,
                "timestamp": "2026-05-11T10:00:00Z",
                "destination_port": 22,
                "flow_duration": 1.0,
                "total_fwd_packets": 2,
                "total_backward_packets": 2,
                "flow_bytes_per_sec": 100.0,
                "flow_packets_per_sec": 4.0,
                "protocol": "tcp",
                "source_ip": "192.168.56.20",
                "target_ip": "192.168.56.10",
                "scenario": "scenario",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_alerts(path):
    alert = {
        "timestamp": "2026-05-11T10:02:10Z",
        "rule": {"id": "5710", "level": 10, "description": "SSH event"},
        "agent": {"name": "target"},
        "data": {"srcip": "192.168.56.20", "dstip": "192.168.56.10"},
        "full_log": "lab alert",
    }
    path.write_text(json.dumps(alert) + "\n", encoding="utf-8")


def test_real_lab_input_validation_writes_report(tmp_path):
    ground_truth = tmp_path / "lab_ground_truth.csv"
    features = tmp_path / "lab_features.csv"
    alerts = tmp_path / "alerts.jsonl"
    output_dir = tmp_path / "validation"
    write_ground_truth(ground_truth)
    write_features(features)
    write_alerts(alerts)

    outputs = validate_real_lab_inputs(
        ground_truth_path=ground_truth,
        features_path=features,
        wazuh_alerts_path=alerts,
        output_dir=output_dir,
    )

    assert outputs["summary"].exists()
    assert outputs["report"].exists()
    assert outputs["metadata"].exists()
    summary = pd.read_csv(outputs["summary"])
    assert "event_id_consistency" in summary["check"].tolist()


def test_real_lab_input_validation_rejects_event_id_mismatch(tmp_path):
    ground_truth = tmp_path / "lab_ground_truth.csv"
    features = tmp_path / "lab_features.csv"
    alerts = tmp_path / "alerts.jsonl"
    write_ground_truth(ground_truth)
    write_features(features, ids=["lab-001", "lab-other"])
    write_alerts(alerts)

    with pytest.raises(ValueError, match="event_id"):
        validate_real_lab_inputs(
            ground_truth_path=ground_truth,
            features_path=features,
            wazuh_alerts_path=alerts,
            output_dir=tmp_path / "validation",
        )
