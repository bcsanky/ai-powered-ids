from __future__ import annotations

from pathlib import Path

from ml.src.final_submission_check.generate_final_submission_readiness import run_check


def write_status(path: Path, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("check_id,category,status,message,recommendation,path\nx,test," + status + ",ok,,\n", encoding="utf-8")


def test_submission_readiness_not_submission_review_without_provenance(tmp_path):
    out = tmp_path / "reports/final_submission_check"
    for name in [
        "requirement_coverage.csv",
        "thesis_structure_check.csv",
        "no_overclaiming_check.csv",
        "submission_artifact_plan.csv",
    ]:
        write_status(out / name, "PASS")
    final_acceptance = tmp_path / "reports/final_acceptance/release_candidate_readiness.json"
    final_acceptance.parent.mkdir(parents=True)
    final_acceptance.write_text('{"readiness":"READY_FOR_REAL_LAB_RUN"}\n', encoding="utf-8")

    result = run_check(out, tmp_path)

    assert result["readiness"] == "READY_FOR_REAL_MEASUREMENT"

