from __future__ import annotations

from ml.src.final_acceptance.check_expected_failure_modes import run_check


def test_failure_modes_check_passes_expected_failures(tmp_path):
    result = run_check(tmp_path / "reports/final_acceptance")

    assert result["status"] == "PASS"
    statuses = {row["check_id"]: row["status"] for row in result["rows"]}
    assert statuses["demo_path_rejected"] == "PASS"
    assert statuses["missing_provenance_thesis_inputs"] == "PASS"

