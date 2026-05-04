from __future__ import annotations

import json

import pandas as pd

from ml.src.lab_session.post_session_input_check import run_post_session_input_check


def write_ground_truth(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "event_id": "lab-001",
                "timestamp_start": "2026-05-04T10:00:00Z",
                "timestamp_end": "2026-05-04T10:05:00Z",
                "scenario": "benign_ssh_login",
                "label": "benign",
                "attack_type": "none",
                "source_ip": "10.0.0.2",
                "target_ip": "10.0.0.10",
            },
            {
                "event_id": "lab-002",
                "timestamp_start": "2026-05-04T10:10:00Z",
                "timestamp_end": "2026-05-04T10:15:00Z",
                "scenario": "port_scan",
                "label": "attack",
                "attack_type": "port_scan",
                "source_ip": "10.0.0.2",
                "target_ip": "10.0.0.10",
            },
        ]
    ).to_csv(path, index=False)


def write_alerts(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"timestamp": "2026-05-04T10:11:00Z", "rule": {"id": "5710", "level": 7}}) + "\n",
        encoding="utf-8",
    )


def test_post_session_input_check_fails_when_wazuh_alerts_missing(tmp_path):
    ground_truth = tmp_path / "data/lab/lab_ground_truth.csv"
    write_ground_truth(ground_truth)
    zeek = tmp_path / "data/lab/zeek/conn.log"
    zeek.parent.mkdir(parents=True)
    zeek.write_text("#fields\tts\n", encoding="utf-8")

    result = run_post_session_input_check(
        ground_truth_path=ground_truth,
        wazuh_alerts_path=tmp_path / "data/wazuh/alerts.jsonl",
        zeek_conn_path=zeek,
        flow_csv_path=tmp_path / "data/lab/flows.csv",
        output_dir=tmp_path / "reports/lab_session",
    )

    assert result["status"] == "FAIL"


def test_post_session_input_check_fails_for_examples_path(tmp_path):
    result = run_post_session_input_check(
        ground_truth_path=tmp_path / "examples/lab/lab_ground_truth.csv",
        wazuh_alerts_path=tmp_path / "data/wazuh/alerts.jsonl",
        zeek_conn_path=tmp_path / "data/lab/zeek/conn.log",
        flow_csv_path=tmp_path / "data/lab/flows.csv",
        output_dir=tmp_path / "reports/lab_session",
    )

    assert result["status"] == "FAIL"
    assert any(row["check_id"] == "ground_truth_path" and row["status"] == "FAIL" for row in result["rows"])


def test_post_session_input_check_passes_for_consistent_tmp_inputs(tmp_path):
    ground_truth = tmp_path / "data/lab/lab_ground_truth.csv"
    alerts = tmp_path / "data/wazuh/alerts.jsonl"
    zeek = tmp_path / "data/lab/zeek/conn.log"
    write_ground_truth(ground_truth)
    write_alerts(alerts)
    zeek.parent.mkdir(parents=True)
    zeek.write_text("#fields\tts\n", encoding="utf-8")

    result = run_post_session_input_check(
        ground_truth_path=ground_truth,
        wazuh_alerts_path=alerts,
        zeek_conn_path=zeek,
        flow_csv_path=tmp_path / "data/lab/flows.csv",
        output_dir=tmp_path / "reports/lab_session",
    )

    assert result["status"] == "PASS"
    assert (tmp_path / "reports/lab_session/post_session_input_check.md").exists()

