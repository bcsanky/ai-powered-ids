from __future__ import annotations

import pandas as pd

from ml.src.live_smoke.check_real_input_paths import run_check


def test_input_path_examples_lab_fails(tmp_path):
    result = run_check(
        tmp_path / "reports/live_smoke",
        ground_truth=tmp_path / "examples/lab/lab_events.csv",
        lab_features=tmp_path / "data/lab/lab_features.csv",
        wazuh_alerts=tmp_path / "data/wazuh/alerts.jsonl",
    )
    summary = pd.read_csv(result["csv"])

    assert "FAIL" in set(summary["status"])


def test_missing_real_input_files_warn_not_fail(tmp_path):
    result = run_check(
        tmp_path / "reports/live_smoke",
        ground_truth=tmp_path / "data/lab/lab_ground_truth.csv",
        lab_features=tmp_path / "data/lab/lab_features.csv",
        wazuh_alerts=tmp_path / "data/wazuh/alerts.jsonl",
    )
    summary = pd.read_csv(result["csv"])

    assert set(summary["status"]) == {"WARN"}


def test_sample_demo_fixture_paths_fail(tmp_path):
    result = run_check(
        tmp_path / "reports/live_smoke",
        ground_truth=tmp_path / "data/lab/sample_ground_truth.csv",
        lab_features=tmp_path / "data/lab/demo_features.csv",
        wazuh_alerts=tmp_path / "data/wazuh/fixture_alerts.jsonl",
    )
    summary = pd.read_csv(result["csv"])

    assert set(summary["status"]) == {"FAIL"}

