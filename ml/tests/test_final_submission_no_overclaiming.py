from __future__ import annotations

from ml.src.final_submission_check.check_no_overclaiming import run_check


def test_no_overclaiming_fails_on_production_ready(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "x.md").write_text("A rendszer production-ready megoldás.\n", encoding="utf-8")

    result = run_check(tmp_path / "reports/final_submission_check", tmp_path)

    assert result["status"] == "FAIL"


def test_no_overclaiming_accepts_laboratory_prototype(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "x.md").write_text("A rendszer laboratóriumi prototípus.\n", encoding="utf-8")

    result = run_check(tmp_path / "reports/final_submission_check", tmp_path)

    assert result["status"] == "PASS"

