from __future__ import annotations

from ml.src.thesis_integration.generate_chapter6_results_section import generate_chapter6
from ml.tests.thesis_integration_fixture import create_thesis_measurement


def call_generator(root):
    return generate_chapter6(
        comparison_path=root / "results/real_comparison/metrics_comparison.csv",
        wazuh_metrics_path=root / "results/wazuh_real/metrics_summary.csv",
        ae_metrics_path=root / "results/ae_lab/metrics_summary.csv",
        hybrid_metrics_path=root / "results/hybrid_real/metrics_summary.csv",
        research_answer_path=root / "reports/real_measurement_qa/research_question_answer.json",
        provenance_path=root / "reports/real_measurement/measurement_provenance.json",
        output_dir=root / "reports/thesis_integration",
    )


def test_chapter6_does_not_claim_improvement_when_hybrid_not_better(tmp_path):
    create_thesis_measurement(tmp_path, hybrid_f1=0.4)

    outputs = call_generator(tmp_path)

    text = outputs["chapter6"].read_text(encoding="utf-8")
    assert "nem igazolható egyértelmű javulás" in text


def test_chapter6_uses_cautious_improvement_when_hybrid_better(tmp_path):
    create_thesis_measurement(tmp_path, hybrid_f1=0.7)

    outputs = call_generator(tmp_path)

    text = outputs["chapter6"].read_text(encoding="utf-8")
    assert "a vizsgált lab mérésben F1 alapján javulás figyelhető meg" in text

