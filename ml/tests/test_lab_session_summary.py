from __future__ import annotations

import json

import yaml

from ml.src.lab_session.session_summary import generate_session_summary


def test_session_summary_does_not_invent_metrics_when_missing(tmp_path):
    plan = tmp_path / "reports/lab_session/session_plan.yaml"
    plan.parent.mkdir(parents=True)
    plan.write_text(yaml.safe_dump({"session_id": "real-lab-test"}), encoding="utf-8")

    outputs = generate_session_summary(
        output_dir=tmp_path / "reports/lab_session",
        session_plan_path=plan,
        post_input_check_path=tmp_path / "reports/lab_session/post_session_input_check.csv",
        provenance_path=tmp_path / "reports/real_measurement/measurement_provenance.json",
        thesis_readiness_path=tmp_path / "reports/real_measurement_qa/thesis_readiness.md",
        metrics_comparison_path=tmp_path / "results/real_comparison/metrics_comparison.csv",
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    text = outputs["markdown"].read_text(encoding="utf-8")

    assert payload["metrics_comparison_exists"] is False
    assert payload["metrics_comparison_rows"] == 0
    assert "Nincs metrics_comparison.csv" in text


def test_session_summary_reports_existing_provenance(tmp_path):
    provenance = tmp_path / "reports/real_measurement/measurement_provenance.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text("{}", encoding="utf-8")

    outputs = generate_session_summary(
        output_dir=tmp_path / "reports/lab_session",
        session_plan_path=tmp_path / "reports/lab_session/session_plan.yaml",
        post_input_check_path=tmp_path / "reports/lab_session/post_session_input_check.csv",
        provenance_path=provenance,
        thesis_readiness_path=tmp_path / "reports/real_measurement_qa/thesis_readiness.md",
        metrics_comparison_path=tmp_path / "results/real_comparison/metrics_comparison.csv",
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))

    assert payload["provenance_exists"] is True

