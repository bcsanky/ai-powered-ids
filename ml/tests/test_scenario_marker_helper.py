from __future__ import annotations

import pytest

from ml.src.lab_session.scenario_marker_helper import scenario_defaults, start_scenario
from ml.src.lab_capture.event_marker import end_event, export_events


def test_scenario_mapping_sets_label_and_attack_type():
    assert scenario_defaults("port_scan") == {"label": "attack", "attack_type": "port_scan"}
    assert scenario_defaults("benign_ssh_login") == {"label": "benign", "attack_type": "none"}


def test_invalid_scenario_raises_error():
    with pytest.raises(ValueError, match="Ismeretlen scenario"):
        scenario_defaults("unknown")


def test_scenario_helper_start_end_export(tmp_path):
    state_file = tmp_path / "data/lab/session_events.json"
    output = tmp_path / "data/lab/lab_ground_truth.csv"

    event = start_scenario(
        state_file=state_file,
        scenario="ssh_bruteforce",
        event_id="lab-001",
        source_ip="10.0.0.2",
        target_ip="10.0.0.10",
        timestamp="2026-05-04T10:00:00Z",
    )
    end_event(state_file=state_file, event_id="lab-001", timestamp="2026-05-04T10:05:00Z")
    df = export_events(state_file=state_file, output_path=output)

    assert event["label"] == "attack"
    assert event["attack_type"] == "brute_force"
    assert df.loc[0, "event_id"] == "lab-001"

