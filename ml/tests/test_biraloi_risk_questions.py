from __future__ import annotations

from ml.src.final_submission_check.generate_biraloi_risk_questions import generate_questions


def test_biraloi_risk_questions_contains_at_least_25_questions(tmp_path):
    output = generate_questions(tmp_path / "reports/final_submission_check")

    text = output.read_text(encoding="utf-8")
    assert text.count("## ") >= 25
    assert "Mi bizonyítja a provenance-t?" in text

