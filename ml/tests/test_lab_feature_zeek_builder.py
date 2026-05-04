from __future__ import annotations

import json

import pandas as pd
import pytest

from ml.src.lab_features.build_features_from_zeek_conn import build_features_from_zeek


def epoch(value: str) -> float:
    return pd.Timestamp(value, tz="UTC").timestamp()


def write_ground_truth(path):
    pd.DataFrame(
        [
            {
                "event_id": "lab-001",
                "timestamp_start": "2026-05-11T10:00:00Z",
                "timestamp_end": "2026-05-11T10:01:00Z",
                "scenario": "port_scan",
                "label": "attack",
                "attack_type": "port_scan",
                "source_ip": "192.168.56.20",
                "target_ip": "192.168.56.10",
            },
            {
                "event_id": "lab-002",
                "timestamp_start": "2026-05-11T10:02:00Z",
                "timestamp_end": "2026-05-11T10:03:00Z",
                "scenario": "ssh_failed_logins",
                "label": "attack",
                "attack_type": "ssh_failed_logins",
                "source_ip": "192.168.56.20",
                "target_ip": "192.168.56.10",
            },
        ]
    ).to_csv(path, index=False)


def write_zeek_conn(path):
    fields = [
        "ts",
        "uid",
        "id.orig_h",
        "id.orig_p",
        "id.resp_h",
        "id.resp_p",
        "proto",
        "duration",
        "orig_bytes",
        "resp_bytes",
        "orig_pkts",
        "resp_pkts",
    ]
    rows = [
        [epoch("2026-05-11T10:00:10Z"), "C1", "192.168.56.20", 51000, "192.168.56.10", 80, "tcp", 2.0, 100, 50, 5, 3],
        [epoch("2026-05-11T10:00:20Z"), "C2", "192.168.56.20", 51001, "192.168.56.10", 443, "udp", 3.0, 200, 100, 7, 1],
        [epoch("2026-05-11T10:02:30Z"), "C3", "192.168.56.20", 51002, "192.168.56.10", 22, "tcp", 0.0, 10, 10, 1, 1],
    ]
    lines = [
        "#separator \\x09",
        "#set_separator\t,",
        "#empty_field\t(empty)",
        "#unset_field\t-",
        "#path\tconn",
        "#fields\t" + "\t".join(fields),
        "#types\t" + "\t".join(["string"] * len(fields)),
    ]
    lines.extend("\t".join(str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_zeek_builder_aggregates_event_window_features(tmp_path):
    ground_truth = tmp_path / "lab_ground_truth.csv"
    conn_log = tmp_path / "conn.log"
    output = tmp_path / "lab_features.csv"
    metadata_output = tmp_path / "metadata.json"
    write_ground_truth(ground_truth)
    write_zeek_conn(conn_log)

    features = build_features_from_zeek(
        conn_log_path=conn_log,
        ground_truth_path=ground_truth,
        output_path=output,
        metadata_output_path=metadata_output,
    )

    assert output.exists()
    assert len(features) == 2
    first = features[features["event_id"] == "lab-001"].iloc[0]
    assert first["destination_port"] == 80
    assert first["protocol"] == "tcp"
    assert first["flow_duration"] == pytest.approx(5.0)
    assert first["total_fwd_packets"] == pytest.approx(12.0)
    assert first["total_backward_packets"] == pytest.approx(4.0)
    assert first["flow_bytes_per_sec"] == pytest.approx(90.0)
    assert first["flow_packets_per_sec"] == pytest.approx(16 / 5)

    second = features[features["event_id"] == "lab-002"].iloc[0]
    assert second["flow_duration"] == pytest.approx(0.0)
    assert second["flow_packets_per_sec"] > 0
    metadata = json.loads(metadata_output.read_text(encoding="utf-8"))
    assert metadata["zero_duration_events"] == ["lab-002"]


def test_zeek_builder_rejects_missing_flow_by_default(tmp_path):
    ground_truth = tmp_path / "lab_ground_truth.csv"
    conn_log = tmp_path / "conn.log"
    write_ground_truth(ground_truth)
    conn_log.write_text(
        "#separator \\x09\n#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tduration\torig_bytes\tresp_bytes\torig_pkts\tresp_pkts\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Nincs illeszkedő flow"):
        build_features_from_zeek(
            conn_log_path=conn_log,
            ground_truth_path=ground_truth,
            output_path=tmp_path / "lab_features.csv",
            metadata_output_path=tmp_path / "metadata.json",
        )
