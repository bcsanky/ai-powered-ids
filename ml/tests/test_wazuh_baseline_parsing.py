from __future__ import annotations

import json

from ml.src.wazuh_baseline.parse_wazuh_alerts import OUTPUT_COLUMNS, normalize_alert, parse_wazuh_alerts


def test_parse_wazuh_jsonl_to_normalized_csv(tmp_path):
    input_path = tmp_path / "alerts.jsonl"
    output_path = tmp_path / "alerts_parsed.csv"
    rows = [
        {
            "timestamp": "2026-05-11T10:00:01Z",
            "rule": {"id": "5710", "level": 10, "description": "SSH login failure"},
            "agent": {"name": "wazuh-agent-1"},
            "data": {"srcip": "10.0.0.5", "dstip": "10.0.0.10"},
            "full_log": "sshd failed login",
        },
        {
            "@timestamp": "2026-05-11T10:00:03Z",
            "rule_id": "1002",
            "rule_level": "5",
            "rule_description": "Port scan indicator",
            "agent_name": "wazuh-agent-2",
            "source_ip": "10.0.0.6",
            "target_ip": "10.0.0.20",
            "message": "scan pattern",
        },
    ]
    input_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    parsed = parse_wazuh_alerts(input_path, output_path)

    assert list(parsed.columns) == OUTPUT_COLUMNS
    assert len(parsed) == 2
    assert output_path.exists()
    assert parsed.iloc[0]["timestamp"] == "2026-05-11T10:00:01Z"
    assert parsed.iloc[0]["rule_id"] == "5710"
    assert parsed.iloc[0]["rule_level"] == 10
    assert parsed.iloc[0]["rule_description"] == "SSH login failure"
    assert parsed.iloc[0]["agent_name"] == "wazuh-agent-1"
    assert parsed.iloc[0]["source_ip"] == "10.0.0.5"
    assert parsed.iloc[0]["target_ip"] == "10.0.0.10"


def test_parse_wazuh_json_hits_export(tmp_path):
    input_path = tmp_path / "alerts.json"
    output_path = tmp_path / "alerts_parsed.csv"
    payload = {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "timestamp": "2026-05-11T11:00:00Z",
                        "rule": {"id": "5503", "level": 7, "description": "Authentication failure"},
                        "data": {"src_ip": "192.168.1.5", "dst_ip": "192.168.1.10"},
                    }
                }
            ]
        }
    }
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    parsed = parse_wazuh_alerts(input_path, output_path)

    assert len(parsed) == 1
    assert parsed.iloc[0]["rule_id"] == "5503"
    assert parsed.iloc[0]["source_ip"] == "192.168.1.5"
    assert parsed.iloc[0]["target_ip"] == "192.168.1.10"


def test_explicit_destination_ip_has_priority_over_agent_ip():
    normalized = normalize_alert(
        {
            "timestamp": "2026-05-11T12:00:00Z",
            "rule": {"id": "1001", "level": 3},
            "agent": {"name": "target-ubuntu", "ip": "192.168.56.101"},
            "data": {"dstip": "10.10.10.20"},
        }
    )

    assert normalized["target_ip"] == "10.10.10.20"


def test_destination_ip_field_has_priority_over_agent_ip():
    normalized = normalize_alert(
        {
            "timestamp": "2026-05-11T12:00:00Z",
            "rule": {"id": "1001", "level": 3},
            "agent": {"name": "target-ubuntu", "ip": "192.168.56.101"},
            "destination": {"ip": "10.10.10.30"},
        }
    )

    assert normalized["target_ip"] == "10.10.10.30"


def test_agent_ip_is_target_ip_fallback_for_host_based_alert():
    normalized = normalize_alert(
        {
            "timestamp": "2026-05-11T12:00:00Z",
            "rule": {"id": "550", "level": 7, "description": "Integrity checksum changed"},
            "agent": {"name": "target-ubuntu", "ip": "192.168.56.101"},
            "full_log": "host-based Wazuh alert",
        }
    )

    assert normalized["agent_name"] == "target-ubuntu"
    assert normalized["source_ip"] == ""
    assert normalized["target_ip"] == "192.168.56.101"
