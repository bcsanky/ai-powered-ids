from __future__ import annotations

from pathlib import Path

from ml.src.submission_bundle.generate_submission_readme import generate_readme


def test_submission_readme_demo_examples_nem_meresi_eredmenyek(tmp_path: Path) -> None:
    generate_readme(tmp_path / "reports", tmp_path / "dist")

    text = (tmp_path / "dist/SUBMISSION_README.md").read_text(encoding="utf-8")

    assert "demo examples nem mérési eredmények" in text
    assert "raw Wazuh alert" in text

