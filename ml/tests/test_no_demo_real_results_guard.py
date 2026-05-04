from __future__ import annotations

import json

import pandas as pd

from ml.src.repo_hygiene.check_no_demo_real_results import check_no_demo_real_results


def test_no_demo_guard_passes_when_no_real_lab_result(tmp_path):
    result = check_no_demo_real_results(tmp_path, tmp_path / "reports/repo_hygiene")

    assert result["status"] == "PASS"


def test_no_demo_guard_fails_without_provenance(tmp_path):
    path = tmp_path / "results/real_comparison/metrics_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"configuration": "Wazuh-only", "f1": 0.5}]).to_csv(path, index=False)

    result = check_no_demo_real_results(tmp_path, tmp_path / "reports/repo_hygiene")

    assert result["status"] == "FAIL"


def test_no_demo_guard_fails_for_demo_input_in_provenance(tmp_path):
    path = tmp_path / "results/real_comparison/metrics_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"configuration": "Wazuh-only", "f1": 0.5}]).to_csv(path, index=False)
    provenance = tmp_path / "reports/real_measurement/measurement_provenance.json"
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(
        json.dumps(
            {
                "measurement_source": "real_lab",
                "ground_truth_path": "examples/lab/lab_ground_truth.csv",
                "lab_features_path": "data/lab/lab_features.csv",
                "wazuh_alerts_path": "data/wazuh/alerts.jsonl",
                "ground_truth_sha256": "a",
                "lab_features_sha256": "b",
                "wazuh_alerts_sha256": "c",
            }
        ),
        encoding="utf-8",
    )

    result = check_no_demo_real_results(tmp_path, tmp_path / "reports/repo_hygiene")

    assert result["status"] == "FAIL"
