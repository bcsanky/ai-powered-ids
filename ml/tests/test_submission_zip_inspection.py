from __future__ import annotations

import zipfile
from pathlib import Path

from ml.src.submission_bundle.common import write_csv, write_json
from ml.src.submission_bundle.inspect_submission_zip import inspect_zip


def test_zip_inspection_fail_alerts_jsonl_eseten(tmp_path: Path) -> None:
    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("data/wazuh/alerts.jsonl", "{}\n")
    write_json(tmp_path / "zip_metadata.json", {"zip_sha256": ""})
    manifest = tmp_path / "manifest.csv"
    write_csv(
        manifest,
        [],
        ["relative_path", "group", "file_size_bytes", "sha256", "include_in_zip", "provenance_status", "note"],
    )

    rows = inspect_zip(tmp_path, zip_path, manifest)

    assert any(row["status"] == "FAIL" and row["check_id"] == "forbidden_paths" for row in rows)


def test_zip_inspection_manifest_tagok_ellenorzese(tmp_path: Path) -> None:
    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("README.md", "readme\n")
        zf.writestr("Makefile", "all:\n")
        zf.writestr("SUBMISSION_README.md", "readme\n")
        zf.writestr("docs/final_submission_check_runbook.md", "doc\n")
        zf.writestr("docs/data_provenance_and_no_fake_measurements.md", "doc\n")
        zf.writestr("ml/src/app.py", "x\n")
    write_json(tmp_path / "zip_metadata.json", {"zip_sha256": ""})
    manifest = tmp_path / "manifest.csv"
    write_csv(
        manifest,
        [
            {
                "relative_path": "README.md",
                "group": "documentation",
                "file_size_bytes": 7,
                "sha256": "",
                "include_in_zip": True,
                "provenance_status": "not_required",
                "note": "",
            }
        ],
        ["relative_path", "group", "file_size_bytes", "sha256", "include_in_zip", "provenance_status", "note"],
    )

    rows = inspect_zip(tmp_path, zip_path, manifest)

    assert not any(row["status"] == "FAIL" for row in rows)

