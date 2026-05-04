from __future__ import annotations

from ml.src.final_acceptance.generate_real_lab_execution_brief import generate_brief


def test_execution_brief_contains_measurement_package_command(tmp_path):
    output = generate_brief(tmp_path / "reports/final_acceptance")

    text = output.read_text(encoding="utf-8")
    assert "make final-real-measurement-package-with-provenance" in text
    assert "dummy" not in text.lower()

