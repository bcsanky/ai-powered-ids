from __future__ import annotations

from pathlib import Path

from ml.src.submission_bundle.common import write_csv, write_json
from ml.src.submission_bundle.generate_submission_bundle_report import report_status


def test_bundle_report_ready_to_attach_csak_fail_nelkul(tmp_path: Path) -> None:
    output_dir = tmp_path / "reports/submission_bundle"
    output_dir.mkdir(parents=True)
    write_csv(
        output_dir / "submission_manifest.csv",
        [
            {
                "relative_path": "README.md",
                "group": "documentation",
                "file_size_bytes": 1,
                "sha256": "x",
                "include_in_zip": True,
                "provenance_status": "not_required",
                "note": "",
            }
        ],
        ["relative_path", "group", "file_size_bytes", "sha256", "include_in_zip", "provenance_status", "note"],
    )
    write_json(output_dir / "submission_zip_inspection.json", {"status": "PASS"})
    write_json(output_dir / "submission_zip_report.json", {"status": "PASS"})

    status, _rows, payload = report_status(output_dir, tmp_path)

    assert status == "READY_TO_ATTACH"
    assert payload["file_count"] == 1


def test_bundle_report_not_ready_fail_eseten(tmp_path: Path) -> None:
    output_dir = tmp_path / "reports/submission_bundle"
    output_dir.mkdir(parents=True)
    write_csv(
        output_dir / "submission_manifest.csv",
        [],
        ["relative_path", "group", "file_size_bytes", "sha256", "include_in_zip", "provenance_status", "note"],
    )
    write_json(output_dir / "submission_zip_inspection.json", {"status": "FAIL"})
    write_json(output_dir / "submission_zip_report.json", {"status": "PASS"})

    status, _rows, _payload = report_status(output_dir, tmp_path)

    assert status == "NOT_READY"

