from __future__ import annotations

from ml.src.final_submission_check.check_requirement_coverage import run_check


def test_requirement_coverage_missing_required_evidence_fails(tmp_path):
    req = tmp_path / "requirements.yaml"
    req.write_text(
        "requirements:\n"
        "  - id: architecture\n"
        "    title: Architecture\n"
        "    evidence:\n"
        "      - docs/missing.md\n",
        encoding="utf-8",
    )

    result = run_check(req, tmp_path / "reports/final_submission_check", tmp_path)

    assert result["status"] == "FAIL"


def test_requirement_coverage_missing_runtime_output_warns(tmp_path):
    req = tmp_path / "requirements.yaml"
    req.write_text(
        "requirements:\n"
        "  - id: summary\n"
        "    title: Summary\n"
        "    evidence:\n"
        "      - reports/thesis_integration/chapter7_osszegzes_generated.md\n",
        encoding="utf-8",
    )

    result = run_check(req, tmp_path / "reports/final_submission_check", tmp_path)

    assert result["status"] == "WARN"

