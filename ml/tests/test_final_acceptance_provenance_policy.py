from __future__ import annotations

from pathlib import Path

from ml.src.final_acceptance.check_provenance_policy import GITIGNORE_PATTERNS, REQUIRED_FILES, run_check


def prepare_policy_root(root: Path, *, include_thesis_ignore: bool = True) -> None:
    for rel_path in REQUIRED_FILES:
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n", encoding="utf-8")
    (root / "examples/lab").mkdir(parents=True)
    (root / "examples/lab/README.md").write_text("demo input\n", encoding="utf-8")
    (root / "examples/scoring").mkdir(parents=True)
    (root / "examples/scoring/README.md").write_text("demo input\n", encoding="utf-8")
    (root / "reports").mkdir(parents=True)
    (root / "reports/README.md").write_text("futási kimenet\n", encoding="utf-8")
    (root / "results").mkdir(parents=True)
    (root / "results/README.md").write_text("futási kimenet\n", encoding="utf-8")
    patterns = [pattern for pattern in GITIGNORE_PATTERNS if include_thesis_ignore or pattern != "reports/thesis_integration/"]
    (root / ".gitignore").write_text("\n".join(patterns) + "\n", encoding="utf-8")


def test_provenance_policy_fails_when_thesis_integration_not_ignored(tmp_path):
    prepare_policy_root(tmp_path, include_thesis_ignore=False)

    result = run_check(tmp_path, tmp_path / "reports/final_acceptance")

    assert result["status"] == "FAIL"


def test_provenance_policy_passes_complete_policy(tmp_path):
    prepare_policy_root(tmp_path)

    result = run_check(tmp_path, tmp_path / "reports/final_acceptance")

    assert result["status"] == "PASS"

