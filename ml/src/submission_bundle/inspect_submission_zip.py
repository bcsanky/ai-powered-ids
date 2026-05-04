from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Any

from ml.src.submission_bundle.common import (
    bool_from_csv,
    ensure_dir,
    is_sensitive_path,
    read_csv_rows,
    read_json_optional,
    sha256_file,
    status_from_rows,
    write_csv,
    write_json,
    write_markdown_report,
)


FIELDS = ["check_id", "category", "status", "message", "recommendation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", dest="zip_path", default="dist/submission/ai_powered_ids_submission_bundle.zip")
    parser.add_argument("--manifest", default="reports/submission_bundle/submission_manifest.csv")
    parser.add_argument("--output-dir", default="reports/submission_bundle")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def check_row(check_id: str, category: str, status: str, message: str, recommendation: str = "") -> dict[str, Any]:
    return {
        "check_id": check_id,
        "category": category,
        "status": status,
        "message": message,
        "recommendation": recommendation,
    }


def inspect_zip(root: Path, zip_path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not zip_path.exists():
        return [check_row("zip_exists", "zip", "FAIL", "A ZIP nem létezik.", "Futtasd: make submission-zip")]
    if zip_path.stat().st_size == 0:
        rows.append(check_row("zip_not_empty", "zip", "FAIL", "A ZIP üres.", "Hozz létre nem üres bundle-t."))
        return rows
    rows.append(check_row("zip_exists", "zip", "PASS", "A ZIP létezik és nem üres."))

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = sorted(zf.namelist())
    except zipfile.BadZipFile:
        return [check_row("zip_valid", "zip", "FAIL", "A fájl nem érvényes ZIP.", "Készítsd újra a ZIP-et.")]

    forbidden = [name for name in names if is_sensitive_path(name) or name.startswith(("data/", "raw/", "artifacts/"))]
    if forbidden:
        rows.append(check_row("forbidden_paths", "security", "FAIL", "Tiltott fájl van a ZIP-ben: " + ", ".join(forbidden[:10]), "Távolítsd el a tiltott fájlokat."))
    else:
        rows.append(check_row("forbidden_paths", "security", "PASS", "Nem található raw, secret vagy érzékeny path a ZIP-ben."))

    manifest_rows = read_csv_rows(manifest_path)
    expected = sorted(row.get("relative_path", "") for row in manifest_rows if bool_from_csv(row.get("include_in_zip", "false")))
    missing = [path for path in expected if path not in names]
    if missing:
        rows.append(check_row("manifest_members", "manifest", "FAIL", "Manifestben szereplő fájl hiányzik a ZIP-ből: " + ", ".join(missing[:10]), "Készítsd újra a ZIP-et."))
    else:
        rows.append(check_row("manifest_members", "manifest", "PASS", "Minden include_in_zip fájl szerepel a ZIP-ben."))

    metadata = read_json_optional(zip_path.parent / "zip_metadata.json") or {}
    actual_hash = sha256_file(zip_path)
    if metadata.get("zip_sha256") and metadata.get("zip_sha256") != actual_hash:
        rows.append(check_row("zip_hash", "integrity", "FAIL", "A ZIP hash nem egyezik a metadata hash-sel.", "Készítsd újra a ZIP-et."))
    else:
        rows.append(check_row("zip_hash", "integrity", "PASS", "A ZIP hash egyezik vagy metadata nem tartalmaz eltérő hash-t."))

    required_entries = [
        "README.md",
        "Makefile",
        "SUBMISSION_README.md",
        "docs/final_submission_check_runbook.md",
        "docs/data_provenance_and_no_fake_measurements.md",
    ]
    for entry in required_entries:
        rows.append(
            check_row(
                f"required_{entry.replace('/', '_')}",
                "content",
                "PASS" if entry in names else "FAIL",
                f"{entry} {'szerepel' if entry in names else 'hiányzik'} a ZIP-ben.",
                "Ellenőrizd a policyt." if entry not in names else "",
            )
        )
    has_source = any(name.startswith("ml/src/") for name in names)
    rows.append(check_row("source_tree", "content", "PASS" if has_source else "FAIL", "Az ml/src fa szerepel a ZIP-ben." if has_source else "Az ml/src fa hiányzik.", "Ellenőrizd a policyt."))
    return rows


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    ensure_dir(output_dir)
    status = status_from_rows(rows)
    write_csv(output_dir / "submission_zip_inspection.csv", rows, FIELDS)
    write_markdown_report(
        output_dir / "submission_zip_inspection.md",
        "Submission ZIP inspection",
        rows,
        FIELDS,
        "A ZIP tartalmi és biztonsági ellenőrzése.",
        status,
    )
    write_json(output_dir / "submission_zip_inspection.json", {"status": status, "checks": rows})


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows = inspect_zip(root, root / args.zip_path, root / args.manifest)
    write_outputs(rows, Path(args.output_dir))
    if status_from_rows(rows) == "FAIL":
        raise SystemExit(1)
    print(f"[OK] ZIP inspection: {Path(args.output_dir) / 'submission_zip_inspection.md'}")


if __name__ == "__main__":
    main()

