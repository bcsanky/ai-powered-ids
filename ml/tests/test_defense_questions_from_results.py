from __future__ import annotations

from ml.src.thesis_integration.generate_defense_questions_from_results import generate_defense_questions
from ml.tests.thesis_integration_fixture import create_thesis_measurement


def test_defense_questions_include_provenance_question(tmp_path):
    create_thesis_measurement(tmp_path)

    output = generate_defense_questions(
        tmp_path / "results/real_comparison/metrics_comparison.csv",
        tmp_path / "reports/thesis_integration",
    )

    text = output.read_text(encoding="utf-8")
    assert "Mi a provenance szerepe?" in text
    assert text.count("## ") >= 20

