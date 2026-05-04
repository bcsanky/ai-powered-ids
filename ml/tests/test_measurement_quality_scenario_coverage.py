from __future__ import annotations

import json

import pandas as pd

from ml.src.measurement_quality.check_scenario_coverage import run_check


def write_thresholds(path):
    path.write_text(
        """
minimums:
  min_total_events: 20
  min_benign_events: 5
  min_attack_events: 5
  min_scenarios: 4
  min_attack_scenarios: 2
expected_scenarios:
  benign: [benign_ssh_login]
  attack: [port_scan, ssh_bruteforce]
claim_thresholds:
  min_f1_delta_for_improvement: 0.01
  max_fpr_increase_without_warning: 0.10
  max_alert_count_multiplier_without_warning: 2.0
""",
        encoding="utf-8",
    )


def write_gt(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_provenance(path, gt_path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "measurement_source": "real_lab",
                "ground_truth_path": gt_path.as_posix(),
                "lab_features_path": "data/lab/lab_features.csv",
                "wazuh_alerts_path": "data/wazuh/alerts.jsonl",
                "ground_truth_sha256": "a",
                "lab_features_sha256": "b",
                "wazuh_alerts_sha256": "c",
            }
        ),
        encoding="utf-8",
    )


def base_rows():
    return [
        {
            "event_id": "e1",
            "timestamp_start": "2026-05-04T10:00:00Z",
            "timestamp_end": "2026-05-04T10:00:10Z",
            "scenario": "benign_ssh_login",
            "label": "benign",
            "attack_type": "none",
            "source_ip": "10.0.0.1",
            "target_ip": "10.0.0.2",
        },
        {
            "event_id": "e2",
            "timestamp_start": "2026-05-04T10:01:00Z",
            "timestamp_end": "2026-05-04T10:01:10Z",
            "scenario": "port_scan",
            "label": "attack",
            "attack_type": "port_scan",
            "source_ip": "10.0.0.1",
            "target_ip": "10.0.0.2",
        },
    ]


def test_scenario_coverage_not_ready_without_provenance(tmp_path):
    thresholds = tmp_path / "thresholds.yaml"
    gt = tmp_path / "data/lab/lab_ground_truth.csv"
    write_thresholds(thresholds)
    write_gt(gt, base_rows())

    result = run_check(
        ground_truth=gt,
        thresholds_path=thresholds,
        provenance_path=tmp_path / "reports/real_measurement/measurement_provenance.json",
        output_dir=tmp_path / "reports/measurement_quality",
    )

    assert any(row["status"] == "NOT_READY" for row in result["rows"])


def test_scenario_coverage_demo_path_fails(tmp_path):
    thresholds = tmp_path / "thresholds.yaml"
    gt = tmp_path / "examples/lab/lab_ground_truth.csv"
    provenance = tmp_path / "reports/real_measurement/measurement_provenance.json"
    write_thresholds(thresholds)
    write_gt(gt, base_rows())
    write_provenance(provenance, gt)

    result = run_check(ground_truth=gt, thresholds_path=thresholds, provenance_path=provenance, output_dir=tmp_path / "reports/measurement_quality")

    assert any(row["status"] == "FAIL" and row["check_id"] == "ground_truth_path_guard" for row in result["rows"])


def test_scenario_coverage_few_events_warns(tmp_path):
    thresholds = tmp_path / "thresholds.yaml"
    gt = tmp_path / "data/lab/lab_ground_truth.csv"
    provenance = tmp_path / "reports/real_measurement/measurement_provenance.json"
    write_thresholds(thresholds)
    write_gt(gt, base_rows())
    write_provenance(provenance, gt)

    result = run_check(ground_truth=gt, thresholds_path=thresholds, provenance_path=provenance, output_dir=tmp_path / "reports/measurement_quality")

    assert any(row["status"] == "WARN" and row["check_id"] == "min_total_events" for row in result["rows"])


def test_scenario_coverage_missing_benign_fails(tmp_path):
    thresholds = tmp_path / "thresholds.yaml"
    gt = tmp_path / "data/lab/lab_ground_truth.csv"
    provenance = tmp_path / "reports/real_measurement/measurement_provenance.json"
    rows = base_rows()
    rows[0]["label"] = "attack"
    rows[0]["scenario"] = "ssh_bruteforce"
    write_thresholds(thresholds)
    write_gt(gt, rows)
    write_provenance(provenance, gt)

    result = run_check(ground_truth=gt, thresholds_path=thresholds, provenance_path=provenance, output_dir=tmp_path / "reports/measurement_quality")

    assert any(row["status"] == "FAIL" and row["check_id"] == "min_benign_events" for row in result["rows"])

