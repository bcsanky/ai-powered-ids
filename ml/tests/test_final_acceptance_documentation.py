from __future__ import annotations

from pathlib import Path

from ml.src.final_acceptance.check_documentation_consistency import DOCS, run_check


def prepare_docs(root: Path, text: str) -> None:
    for rel_path in DOCS:
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_documentation_checker_finds_forbidden_claim(tmp_path):
    prepare_docs(tmp_path, "Ez production-ready állítás.\n")

    result = run_check(tmp_path, tmp_path / "reports/final_acceptance")

    assert result["status"] == "FAIL"


def test_documentation_checker_accepts_laboratory_prototype_wording(tmp_path):
    prepare_docs(tmp_path, "Ez laboratóriumi prototípus, verified provenance esetén használható.\n")

    result = run_check(tmp_path, tmp_path / "reports/final_acceptance")

    assert result["status"] == "PASS"

