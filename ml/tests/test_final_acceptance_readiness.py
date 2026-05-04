from __future__ import annotations

from pathlib import Path

from ml.src.final_acceptance.check_release_candidate_readiness import run_check


def write_check(path: Path, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("check_id,category,status,message,recommendation,path\nx,test," + status + ",ok,,\n", encoding="utf-8")


def test_readiness_not_ready_on_any_fail(tmp_path):
    out = tmp_path / "reports/final_acceptance"
    write_check(out / "make_targets_check.csv", "PASS")
    write_check(out / "failure_modes_check.csv", "FAIL")
    write_check(out / "provenance_policy_check.csv", "PASS")
    write_check(out / "documentation_consistency_check.csv", "PASS")

    result = run_check(out)

    assert result["readiness"] == "NOT_READY"


def test_readiness_ready_when_all_critical_checks_pass(tmp_path):
    out = tmp_path / "reports/final_acceptance"
    for name in [
        "make_targets_check.csv",
        "failure_modes_check.csv",
        "provenance_policy_check.csv",
        "documentation_consistency_check.csv",
    ]:
        write_check(out / name, "PASS")

    result = run_check(out)

    assert result["readiness"] == "READY_FOR_REAL_LAB_RUN"

