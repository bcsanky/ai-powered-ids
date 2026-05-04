from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from ml.src.submission_bundle.common import write_csv
from ml.src.submission_bundle.create_submission_zip import create_submission_zip


def test_zip_nem_tartalmaz_tiltott_pathot(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/a.md").write_text("doc\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    write_csv(
        manifest,
        [
            {
                "relative_path": "docs/a.md",
                "group": "documentation",
                "file_size_bytes": 4,
                "sha256": "",
                "include_in_zip": True,
                "provenance_status": "not_required",
                "note": "",
            }
        ],
        ["relative_path", "group", "file_size_bytes", "sha256", "include_in_zip", "provenance_status", "note"],
    )

    metadata = create_submission_zip(tmp_path, manifest, tmp_path / "bundle.zip")

    with zipfile.ZipFile(tmp_path / "bundle.zip") as zf:
        assert "docs/a.md" in zf.namelist()
        assert not any(name.startswith("data/") for name in zf.namelist())
    assert metadata["file_count"] == 1


def test_zip_fail_tiltott_alert_manifest_eseten(tmp_path: Path) -> None:
    (tmp_path / "data/wazuh").mkdir(parents=True)
    (tmp_path / "data/wazuh/alerts.jsonl").write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    write_csv(
        manifest,
        [
            {
                "relative_path": "data/wazuh/alerts.jsonl",
                "group": "raw",
                "file_size_bytes": 3,
                "sha256": "",
                "include_in_zip": True,
                "provenance_status": "not_required",
                "note": "",
            }
        ],
        ["relative_path", "group", "file_size_bytes", "sha256", "include_in_zip", "provenance_status", "note"],
    )

    with pytest.raises(ValueError):
        create_submission_zip(tmp_path, manifest, tmp_path / "bundle.zip")

