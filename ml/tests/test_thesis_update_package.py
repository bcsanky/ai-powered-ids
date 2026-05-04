from __future__ import annotations

from ml.src.thesis_integration.generate_thesis_update_package import generate_package


def test_update_package_lists_generated_files(tmp_path):
    output_dir = tmp_path / "reports/thesis_integration"
    (output_dir / "chapter5_implementation_generated.md").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "chapter5_implementation_generated.md").write_text("ok\n", encoding="utf-8")

    outputs = generate_package(output_dir)

    text = outputs["package"].read_text(encoding="utf-8")
    assert "chapter5_implementation_generated.md" in text
    assert "defense_questions_generated.md" in text

