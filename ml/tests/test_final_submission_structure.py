from __future__ import annotations

from ml.src.final_submission_check.check_thesis_structure_expectations import run_check


def test_thesis_structure_warns_for_results_without_provenance(tmp_path):
    doc = tmp_path / "docs/thesis_chapter_5_implementation_final.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("architektúra Python Wazuh autoencoder hibrid korlát összegzés Summary\n", encoding="utf-8")

    result = run_check(tmp_path / "reports/final_submission_check", tmp_path)

    statuses = {row["check_id"]: row["status"] for row in result["rows"]}
    assert statuses["eredmények"] == "WARN"

