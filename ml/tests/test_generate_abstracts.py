from __future__ import annotations

from ml.src.thesis_integration.generate_abstracts import generate_abstracts
from ml.tests.thesis_integration_fixture import create_thesis_measurement


def test_abstract_does_not_claim_fake_improvement(tmp_path):
    create_thesis_measurement(tmp_path, hybrid_f1=0.4)

    outputs = generate_abstracts(
        tmp_path / "results/real_comparison/metrics_comparison.csv",
        tmp_path / "reports/real_measurement/measurement_provenance.json",
        tmp_path / "reports/thesis_integration",
    )

    hu = outputs["hu"].read_text(encoding="utf-8")
    en = outputs["en"].read_text(encoding="utf-8")
    assert "nem igazolható egyértelmű javulás" in hu
    assert "do not show clear F1 improvement" in en

