from __future__ import annotations

import argparse

import pandas as pd
import pytest

from ml.src.lab_features.build_features_from_flow_csv import build_features_from_flow_csv


def make_args() -> argparse.Namespace:
    return argparse.Namespace(
        timestamp_col="ts",
        src_ip_col="src",
        dst_ip_col="dst",
        dst_port_col="dport",
        protocol_col="proto_name",
        duration_col="dur",
        fwd_packets_col="fwd",
        bwd_packets_col="bwd",
        bytes_per_sec_col="bps",
        packets_per_sec_col="pps",
    )


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
            }
        ]
    ).to_csv(path, index=False)


def test_flow_csv_builder_uses_parameterized_columns(tmp_path):
    ground_truth = tmp_path / "lab_ground_truth.csv"
    flow_csv = tmp_path / "flows.csv"
    output = tmp_path / "lab_features.csv"
    write_ground_truth(ground_truth)
    pd.DataFrame(
        [
            {
                "ts": "2026-05-11T10:00:10Z",
                "src": "192.168.56.20",
                "dst": "192.168.56.10",
                "dport": 22,
                "proto_name": "tcp",
                "dur": 5.0,
                "fwd": 8,
                "bwd": 6,
                "bps": 100.0,
                "pps": 2.8,
            }
        ]
    ).to_csv(flow_csv, index=False)

    features = build_features_from_flow_csv(
        flow_csv_path=flow_csv,
        ground_truth_path=ground_truth,
        output_path=output,
        args=make_args(),
        metadata_output_path=tmp_path / "metadata.json",
    )

    assert output.exists()
    row = features.iloc[0]
    assert row["destination_port"] == 22
    assert row["protocol"] == "tcp"
    assert row["flow_bytes_per_sec"] == pytest.approx(100.0)
    assert row["flow_packets_per_sec"] == pytest.approx(14 / 5)


def test_flow_csv_builder_rejects_missing_required_column(tmp_path):
    ground_truth = tmp_path / "lab_ground_truth.csv"
    flow_csv = tmp_path / "flows.csv"
    write_ground_truth(ground_truth)
    pd.DataFrame({"ts": ["2026-05-11T10:00:10Z"]}).to_csv(flow_csv, index=False)

    with pytest.raises(ValueError, match="Hiányzó flow CSV oszlopok"):
        build_features_from_flow_csv(
            flow_csv_path=flow_csv,
            ground_truth_path=ground_truth,
            output_path=tmp_path / "lab_features.csv",
            args=make_args(),
            metadata_output_path=tmp_path / "metadata.json",
        )
