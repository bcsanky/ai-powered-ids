from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from ml.src.submission_bundle.common import (
    bool_from_csv,
    ensure_dir,
    read_csv_rows,
    read_json_optional,
    write_csv,
    write_json,
    write_markdown_report,
)


FIELDS = ["item", "status", "message"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/submission_bundle")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def report_status(output_dir: Path, root: Path) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    manifest = read_csv_rows(output_dir / "submission_manifest.csv")
    inspection = read_json_optional(output_dir / "submission_zip_inspection.json") or {}
    zip_report = read_json_optional(output_dir / "submission_zip_report.json") or {}
    final_submission = read_json_optional(root / "reports/final_submission_check/final_submission_readiness.json")
    final_acceptance = read_json_optional(root / "reports/final_acceptance/release_candidate_readiness.json")
    include_rows = [row for row in manifest if bool_from_csv(row.get("include_in_zip", "false"))]
    group_counts = Counter(row.get("group", "unknown") for row in include_rows)
    failures = []
    if inspection.get("status") == "FAIL":
        failures.append("ZIP inspection FAIL")
    if zip_report.get("status") == "FAIL":
        failures.append("ZIP report FAIL")
    if not include_rows:
        failures.append("Nincs ZIP-be kerülő fájl")
    warnings = []
    if any(row.get("provenance_status") == "missing_provenance" for row in manifest):
        warnings.append("Runtime output provenance nélkül nem került csomagolásra")
    status = "NOT_READY" if failures else ("READY_WITH_WARNINGS" if warnings else "READY_TO_ATTACH")
    rows = [
        {"item": "bundle_status", "status": status, "message": "Beadási csomag státusza."},
        {"item": "file_count", "status": "PASS", "message": str(len(include_rows))},
        {"item": "groups", "status": "PASS", "message": ", ".join(f"{key}: {value}" for key, value in sorted(group_counts.items()))},
        {"item": "zip_inspection", "status": inspection.get("status", "WARN"), "message": "ZIP inspection státusz."},
        {"item": "final_submission", "status": final_submission.get("status", "WARN") if isinstance(final_submission, dict) else "WARN", "message": "Final submission readiness, ha rendelkezésre áll."},
        {"item": "final_acceptance", "status": final_acceptance.get("status", "WARN") if isinstance(final_acceptance, dict) else "WARN", "message": "Final acceptance readiness, ha rendelkezésre áll."},
    ]
    for warning in warnings:
        rows.append({"item": "warning", "status": "WARN", "message": warning})
    for failure in failures:
        rows.append({"item": "failure", "status": "FAIL", "message": failure})
    payload = {
        "status": status,
        "file_count": len(include_rows),
        "group_counts": dict(group_counts),
        "failures": failures,
        "warnings": warnings,
    }
    return status, rows, payload


def write_outputs(output_dir: Path, status: str, rows: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    ensure_dir(output_dir)
    write_csv(output_dir / "submission_bundle_report.csv", rows, FIELDS)
    intro = (
        "A riport a beadási ZIP csomag állapotát foglalja össze. "
        "A státusz nem mérési eredmény, hanem csomagolási és archiválási QA."
    )
    write_markdown_report(
        output_dir / "submission_bundle_report.md",
        "Submission bundle riport",
        rows,
        FIELDS,
        intro,
        status,
    )
    write_json(output_dir / "submission_bundle_report.json", payload)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    status, rows, payload = report_status(output_dir, Path(args.root))
    write_outputs(output_dir, status, rows, payload)
    if status == "NOT_READY":
        raise SystemExit(1)
    print(f"[OK] Submission bundle report: {output_dir / 'submission_bundle_report.md'}")


if __name__ == "__main__":
    main()

