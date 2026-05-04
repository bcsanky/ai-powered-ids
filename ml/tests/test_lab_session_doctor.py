from __future__ import annotations

import pandas as pd

from ml.src.lab_session.session_doctor import run_session_doctor


def write(path, content="x\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def prepare_minimal_project(root):
    makefile_targets = "\n".join(
        [
            "lab-templates:",
            "real-measurement-preflight:",
            "final-real-measurement-package-with-provenance:",
            "final-live-integration:",
            "repo-hygiene-check:",
        ]
    )
    write(root / "Makefile", makefile_targets)
    write(root / "experiments/final/ae_minimal.yaml")
    for path in [
        "templates/lab/lab_ground_truth_template.csv",
        "templates/lab/lab_features_template.csv",
        "templates/lab/lab_scenarios_template.yaml",
        "docs/lab_attack_scenarios_runbook.md",
        "docs/final_real_measurement_checklist.md",
        "docs/real_measurement_export_and_packaging_runbook.md",
        "docs/data_provenance_and_no_fake_measurements.md",
        "docs/live_integration_runbook.md",
    ]:
        write(root / path)


def test_session_doctor_warns_for_missing_model_files_but_no_fail(tmp_path):
    prepare_minimal_project(tmp_path)

    result = run_session_doctor(root=tmp_path, output_dir=tmp_path / "reports/lab_session")
    summary = pd.read_csv(result["summary"])

    assert result["status"] == "WARN"
    assert "FAIL" not in set(summary["status"])
    assert (tmp_path / "data/lab").exists()
    assert (tmp_path / "reports/lab_session/session_doctor_report.md").exists()


def test_session_doctor_fails_for_missing_required_project_file(tmp_path):
    prepare_minimal_project(tmp_path)
    (tmp_path / "Makefile").unlink()

    result = run_session_doctor(root=tmp_path, output_dir=tmp_path / "reports/lab_session")

    assert result["status"] == "FAIL"

