from __future__ import annotations

import pandas as pd
import pytest

from ml.src.lab_ae_eval.score_lab_features import merge_features_with_ground_truth
from ml.src.lab_ae_eval.validate_lab_features import validate_lab_features


def write_lab_features(path, rows=None):
    if rows is None:
        rows = [
            {
                "event_id": "evt-1",
                "destination_port": 80,
                "flow_duration": 1000,
                "total_fwd_packets": 10,
                "total_backward_packets": 8,
                "flow_bytes_per_sec": 100.0,
                "flow_packets_per_sec": 2.0,
                "protocol": "6",
                "timestamp": "2026-05-11T10:00:00Z",
                "source_ip": "10.0.0.1",
                "target_ip": "10.0.0.2",
                "scenario": "benign_activity",
            }
        ]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_ground_truth(path, label="benign", event_id="evt-1"):
    pd.DataFrame(
        [
            {
                "event_id": event_id,
                "timestamp_start": "2026-05-11T10:00:00Z",
                "timestamp_end": "2026-05-11T10:01:00Z",
                "scenario": "benign_activity",
                "label": label,
                "attack_type": "",
                "source_ip": "10.0.0.1",
                "target_ip": "10.0.0.2",
            }
        ]
    ).to_csv(path, index=False)


def test_validate_lab_features_writes_validated_csv(tmp_path):
    input_path = tmp_path / "lab_features.csv"
    output_path = tmp_path / "lab_features_validated.csv"
    write_lab_features(input_path)

    validated = validate_lab_features(input_path, output_path)

    assert output_path.exists()
    assert len(validated) == 1
    assert validated.loc[0, "event_id"] == "evt-1"
    assert validated.loc[0, "destination_port"] == 80
    assert validated.loc[0, "timestamp"] == "2026-05-11T10:00:00Z"


def test_validate_lab_features_rejects_missing_required_column(tmp_path):
    input_path = tmp_path / "lab_features.csv"
    rows = [
        {
            "event_id": "evt-1",
            "destination_port": 80,
            "flow_duration": 1000,
            "total_fwd_packets": 10,
            "total_backward_packets": 8,
            "flow_bytes_per_sec": 100.0,
            "protocol": "6",
        }
    ]
    write_lab_features(input_path, rows=rows)

    with pytest.raises(ValueError, match="flow_packets_per_sec"):
        validate_lab_features(input_path)


def test_validate_lab_features_rejects_non_numeric_value(tmp_path):
    input_path = tmp_path / "lab_features.csv"
    rows = [
        {
            "event_id": "evt-1",
            "destination_port": "not-a-number",
            "flow_duration": 1000,
            "total_fwd_packets": 10,
            "total_backward_packets": 8,
            "flow_bytes_per_sec": 100.0,
            "flow_packets_per_sec": 2.0,
            "protocol": "6",
        }
    ]
    write_lab_features(input_path, rows=rows)

    with pytest.raises(ValueError, match="destination_port"):
        validate_lab_features(input_path)


def test_merge_features_rejects_bad_ground_truth_label(tmp_path):
    features_path = tmp_path / "lab_features.csv"
    truth_path = tmp_path / "lab_ground_truth.csv"
    write_lab_features(features_path)
    write_ground_truth(truth_path, label="unknown")

    with pytest.raises(ValueError, match="label"):
        merge_features_with_ground_truth(features_path, truth_path)


def test_merge_features_rejects_event_id_mismatch(tmp_path):
    features_path = tmp_path / "lab_features.csv"
    truth_path = tmp_path / "lab_ground_truth.csv"
    write_lab_features(features_path)
    write_ground_truth(truth_path, event_id="evt-other")

    with pytest.raises(ValueError, match="event_id"):
        merge_features_with_ground_truth(features_path, truth_path)
