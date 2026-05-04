from __future__ import annotations

from pathlib import Path

from ml.src.final_acceptance.check_real_lab_runbook_docs import REQUIRED_DOCS, REQUIRED_KEYWORDS, check_docs


def write_doc(root: Path, relative_path: str, include_safety: bool = True) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    keywords = REQUIRED_KEYWORDS.copy()
    if not include_safety:
        keywords = [keyword for keyword in keywords if keyword != "tilos idegen IP"]
    path.write_text(" ".join(keywords) + "\n", encoding="utf-8")


def test_hianyzo_runbook_fail(tmp_path: Path) -> None:
    result = check_docs(tmp_path, tmp_path / "reports", docs=["docs/missing.md"])

    assert result["status"] == "FAIL"


def test_minden_kotelezo_dokumentum_pass(tmp_path: Path) -> None:
    for doc in REQUIRED_DOCS:
        write_doc(tmp_path, doc)

    result = check_docs(tmp_path, tmp_path / "reports")

    assert result["status"] == "PASS"


def test_hianyzo_tilos_idegen_ip_fail(tmp_path: Path) -> None:
    write_doc(tmp_path, "docs/real_lab_execution_day_runbook.md", include_safety=False)

    result = check_docs(tmp_path, tmp_path / "reports", docs=["docs/real_lab_execution_day_runbook.md"])

    assert result["status"] == "FAIL"


def test_output_report_letrejon(tmp_path: Path) -> None:
    write_doc(tmp_path, "docs/real_lab_execution_day_runbook.md")

    check_docs(tmp_path, tmp_path / "reports", docs=["docs/real_lab_execution_day_runbook.md"])

    assert (tmp_path / "reports/real_lab_runbook_docs_check.md").exists()
    assert (tmp_path / "reports/real_lab_runbook_docs_check.csv").exists()
    assert (tmp_path / "reports/real_lab_runbook_docs_check.json").exists()

