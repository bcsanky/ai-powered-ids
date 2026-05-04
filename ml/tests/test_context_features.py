from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ML_SRC = Path(__file__).resolve().parents[1] / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from build_dataset import apply_context_features, apply_dev_sample, fit_context_statistics


def make_flow_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "destination_port": [80, 80, 443],
            "protocol": ["tcp", "tcp", "udp"],
            "total_fwd_packets": [10, 20, 30],
            "total_backward_packets": [5, 0, 10],
            "flow_bytes_per_sec": [100.0, 200.0, 300.0],
            "flow_packets_per_sec": [10.0, 20.0, 0.0],
        }
    )


def test_context_frequencies_are_fit_on_train_and_unknowns_are_zero():
    train_df = make_flow_df()
    stats = fit_context_statistics(
        train_df,
        rare_destination_port_threshold=0.4,
    )

    eval_df = pd.DataFrame(
        {
            "destination_port": [80, 22],
            "protocol": ["tcp", "icmp"],
            "total_fwd_packets": [6, 9],
            "total_backward_packets": [3, 0],
            "flow_bytes_per_sec": [60.0, 90.0],
            "flow_packets_per_sec": [6.0, 0.0],
        }
    )

    out = apply_context_features(
        eval_df,
        context_statistics=stats,
        bytes_packets_epsilon=1e-9,
        unknown_frequency=0.0,
    )

    assert out.loc[0, "destination_port_frequency"] == 2 / 3
    assert out.loc[0, "protocol_frequency"] == 2 / 3
    assert out.loc[0, "is_rare_destination_port"] == 0
    assert out.loc[0, "packet_ratio"] == 2.0
    assert out.loc[0, "bytes_packets_ratio"] == 10.0

    assert out.loc[1, "destination_port_frequency"] == 0.0
    assert out.loc[1, "protocol_frequency"] == 0.0
    assert out.loc[1, "is_rare_destination_port"] == 1
    assert out.loc[1, "packet_ratio"] == 9.0
    assert out.loc[1, "bytes_packets_ratio"] == 90.0 / 1e-9


def test_dev_sample_is_deterministic_and_preserves_classes():
    df = pd.DataFrame(
        {
            "is_benign": [1] * 90 + [0] * 10,
            "label": ["benign"] * 90 + ["attack"] * 10,
        }
    )

    sampled_a, metadata_a = apply_dev_sample(
        df,
        {"enabled": True, "max_rows_total": 20},
        seed=42,
    )
    sampled_b, metadata_b = apply_dev_sample(
        df,
        {"enabled": True, "max_rows_total": 20},
        seed=42,
    )

    assert sampled_a.equals(sampled_b)
    assert metadata_a == metadata_b
    assert len(sampled_a) == 20
    assert set(sampled_a["is_benign"].unique()) == {0, 1}
    assert metadata_a["used"] is True
    assert metadata_a["sample_stage"] == "after_cleaning_before_split"
