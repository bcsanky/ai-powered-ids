from __future__ import annotations

from ml.src.thesis_integration.generate_chapter7_summary_hu import generate_summary_hu
from ml.src.thesis_integration.generate_chapter8_summary_en import generate_summary_en
from ml.tests.thesis_integration_fixture import create_thesis_measurement


def test_summary_sections_avoid_production_soc_claim(tmp_path):
    create_thesis_measurement(tmp_path)

    hu = generate_summary_hu(
        tmp_path / "results/real_comparison/metrics_comparison.csv",
        tmp_path / "reports/thesis_integration/chapter6_research_question_answer.md",
        tmp_path / "reports/real_measurement/measurement_provenance.json",
        tmp_path / "reports/thesis_integration",
    )
    en = generate_summary_en(
        tmp_path / "results/real_comparison/metrics_comparison.csv",
        tmp_path / "reports/real_measurement/measurement_provenance.json",
        tmp_path / "reports/thesis_integration",
    )

    combined = hu.read_text(encoding="utf-8") + en.read_text(encoding="utf-8")
    assert "éles SOC rendszer" not in combined
    assert "production-ready SOC system" not in combined

