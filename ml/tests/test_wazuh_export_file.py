from __future__ import annotations

import json

from ml.src.wazuh_export.export_alerts_from_file import export_alerts_from_file


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_export_alerts_from_opensearch_hits_json(tmp_path):
    raw = tmp_path / "wazuh_export.json"
    output = tmp_path / "alerts.jsonl"
    metadata = tmp_path / "metadata.json"
    payload = {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "@timestamp": "2026-05-04T10:15:00Z",
                        "rule": {"id": "5710", "level": 10},
                    }
                },
                {
                    "_source": {
                        "@timestamp": "2026-05-04T12:00:00Z",
                        "rule": {"id": "9999", "level": 3},
                    }
                },
            ]
        }
    }
    raw.write_text(json.dumps(payload), encoding="utf-8")

    result = export_alerts_from_file(
        input_path=raw,
        output_path=output,
        time_start="2026-05-04T10:00:00Z",
        time_end="2026-05-04T11:00:00Z",
        metadata_output_path=metadata,
    )

    rows = read_jsonl(output)
    assert result["event_count"] == 1
    assert rows[0]["rule"]["id"] == "5710"
    assert "password" not in metadata.read_text(encoding="utf-8").lower()


def test_export_alerts_from_jsonl_filters_time_window(tmp_path):
    raw = tmp_path / "alerts_raw.jsonl"
    output = tmp_path / "alerts.jsonl"
    raw.write_text(
        "\n".join(
            [
                json.dumps({"timestamp": "2026-05-04T10:01:00Z", "rule": {"id": "1"}}),
                json.dumps({"timestamp": "2026-05-04T09:59:00Z", "rule": {"id": "2"}}),
            ]
        ),
        encoding="utf-8",
    )

    export_alerts_from_file(
        input_path=raw,
        output_path=output,
        time_start="2026-05-04T10:00:00Z",
        time_end="2026-05-04T10:30:00Z",
    )

    rows = read_jsonl(output)
    assert len(rows) == 1
    assert rows[0]["rule"]["id"] == "1"


def test_export_alerts_empty_window_writes_empty_jsonl(tmp_path):
    raw = tmp_path / "alerts_raw.jsonl"
    output = tmp_path / "alerts.jsonl"
    metadata = tmp_path / "metadata.json"
    raw.write_text(json.dumps({"timestamp": "2026-05-04T09:00:00Z", "rule": {"id": "2"}}), encoding="utf-8")

    result = export_alerts_from_file(
        input_path=raw,
        output_path=output,
        time_start="2026-05-04T10:00:00Z",
        time_end="2026-05-04T10:30:00Z",
        metadata_output_path=metadata,
    )

    assert output.exists()
    assert output.read_text(encoding="utf-8") == ""
    assert result["event_count"] == 0
    assert "0 találat" in metadata.read_text(encoding="utf-8")
