from __future__ import annotations

import pandas as pd

from ml.src.real_measurement_qa.preflight_check import run_preflight


def create_required_preflight_files(root):
    for rel_path in [
        "Makefile",
        "experiments/final/ae_minimal.yaml",
        "templates/lab/lab_ground_truth_template.csv",
        "templates/lab/lab_features_template.csv",
        "templates/lab/lab_scenarios_template.yaml",
        "docs/lab_attack_scenarios_runbook.md",
        "docs/final_real_measurement_checklist.md",
        "docs/real_hybrid_lab_evaluation_runbook.md",
        "docs/real_measurement_export_and_packaging_runbook.md",
    ]:
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n", encoding="utf-8")


def test_preflight_warns_for_missing_optional_inputs(tmp_path):
    create_required_preflight_files(tmp_path)

    result = run_preflight(tmp_path, tmp_path / "reports/real_measurement_qa")

    summary = pd.read_csv(result["summary"])
    assert result["overall_status"] == "PASS"
    assert "WARN" in set(summary["status"])
    optional = summary[summary["category"] == "Opcionális inputok"]
    assert not optional.empty
    assert set(optional["status"]) == {"WARN"}


def test_preflight_fails_for_missing_required_project_file(tmp_path):
    create_required_preflight_files(tmp_path)
    (tmp_path / "Makefile").unlink()

    result = run_preflight(tmp_path, tmp_path / "reports/real_measurement_qa")

    summary = pd.read_csv(result["summary"])
    assert result["overall_status"] == "FAIL"
    assert "FAIL" in set(summary["status"])
