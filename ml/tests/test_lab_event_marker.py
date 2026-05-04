from __future__ import annotations

import pandas as pd
import pytest

from ml.src.lab_capture.event_marker import end_event, export_events, read_state, start_event


def test_event_marker_start_end_export(tmp_path):
    state_file = tmp_path / "session_events.json"
    output = tmp_path / "lab_ground_truth.csv"

    start_event(
        state_file=state_file,
        event_id="lab-001",
        scenario="port_scan",
        label="attack",
        attack_type="port_scan",
        source_ip="192.168.56.20",
        target_ip="192.168.56.10",
        timestamp="2026-05-11T10:00:00Z",
    )
    end_event(
        state_file=state_file,
        event_id="lab-001",
        timestamp="2026-05-11T10:01:00Z",
    )
    exported = export_events(state_file=state_file, output_path=output)

    assert len(read_state(state_file)) == 1
    assert output.exists()
    assert exported.loc[0, "event_id"] == "lab-001"
    assert exported.loc[0, "label"] == "attack"
    written = pd.read_csv(output)
    assert written.loc[0, "timestamp_start"] == "2026-05-11T10:00:00Z"


def test_event_marker_rejects_open_event_export(tmp_path):
    state_file = tmp_path / "session_events.json"
    start_event(
        state_file=state_file,
        event_id="lab-open",
        scenario="ssh_failed_logins",
        label="attack",
        attack_type="ssh_failed_logins",
        source_ip="192.168.56.20",
        target_ip="192.168.56.10",
        timestamp="2026-05-11T10:00:00Z",
    )

    with pytest.raises(ValueError, match="Lezáratlan"):
        export_events(state_file=state_file, output_path=tmp_path / "lab_ground_truth.csv")


def test_event_marker_rejects_invalid_label(tmp_path):
    with pytest.raises(ValueError, match="label"):
        start_event(
            state_file=tmp_path / "session_events.json",
            event_id="lab-bad",
            scenario="bad",
            label="unknown",
            attack_type="",
            source_ip="192.168.56.20",
            target_ip="192.168.56.10",
        )
