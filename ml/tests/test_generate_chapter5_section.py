from __future__ import annotations

from ml.src.thesis_integration.generate_chapter5_implementation_section import generate_chapter5
from ml.tests.thesis_integration_fixture import create_thesis_measurement


def test_generate_chapter5_section_writes_implementation_text(tmp_path):
    create_thesis_measurement(tmp_path)

    outputs = generate_chapter5(
        tmp_path / "reports/thesis_integration",
        tmp_path / "reports/real_measurement/measurement_provenance.json",
        tmp_path / "reports/live_integration/enrichment_summary.csv",
    )

    text = outputs["chapter5"].read_text(encoding="utf-8")
    assert "5.5 Wazuh-only baseline" in text
    assert "nem benchmark" in text

