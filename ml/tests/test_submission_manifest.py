from __future__ import annotations

import hashlib
from pathlib import Path

from ml.src.submission_bundle.common import write_csv
from ml.src.submission_bundle.create_submission_manifest import create_manifest_rows


FIELDS = ["relative_path", "group", "exists", "include_candidate", "requires_provenance", "reason", "warning"]


def test_manifest_sha256_helyes_tmp_fajlon(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    content = "hello\n"
    (tmp_path / "docs/a.md").write_text(content, encoding="utf-8")
    write_csv(
        tmp_path / "submission_candidates.csv",
        [
            {
                "relative_path": "docs/a.md",
                "group": "documentation",
                "exists": True,
                "include_candidate": True,
                "requires_provenance": False,
                "reason": "",
                "warning": "",
            }
        ],
        FIELDS,
    )
    write_csv(
        tmp_path / "submission_candidate_validation.csv",
        [{"check_id": "x", "relative_path": "docs/a.md", "category": "documentation", "status": "PASS", "message": "", "recommendation": ""}],
        ["check_id", "relative_path", "category", "status", "message", "recommendation"],
    )

    rows = create_manifest_rows(tmp_path, tmp_path)

    assert rows[0]["include_in_zip"] is True
    assert rows[0]["sha256"] == hashlib.sha256(content.encode()).hexdigest()


def test_runtime_result_missing_provenance_nem_include(tmp_path: Path) -> None:
    (tmp_path / "reports/real_measurement").mkdir(parents=True)
    (tmp_path / "reports/real_measurement/a.md").write_text("x\n", encoding="utf-8")
    write_csv(
        tmp_path / "submission_candidates.csv",
        [
            {
                "relative_path": "reports/real_measurement/a.md",
                "group": "runtime_output",
                "exists": True,
                "include_candidate": True,
                "requires_provenance": True,
                "reason": "",
                "warning": "",
            }
        ],
        FIELDS,
    )
    write_csv(
        tmp_path / "submission_candidate_validation.csv",
        [{"check_id": "x", "relative_path": "reports/real_measurement/a.md", "category": "runtime", "status": "PASS", "message": "", "recommendation": ""}],
        ["check_id", "relative_path", "category", "status", "message", "recommendation"],
    )

    rows = create_manifest_rows(tmp_path, tmp_path)

    assert rows[0]["include_in_zip"] is False
    assert rows[0]["provenance_status"] == "missing_provenance"

