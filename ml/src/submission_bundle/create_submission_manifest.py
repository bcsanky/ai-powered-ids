from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ml.src.submission_bundle.common import (
    bool_from_csv,
    ensure_dir,
    is_runtime_output_path,
    load_verified_provenance,
    read_csv_rows,
    sha256_file,
    write_csv,
    write_json,
    write_markdown_report,
)


FIELDS = [
    "relative_path",
    "group",
    "file_size_bytes",
    "sha256",
    "include_in_zip",
    "provenance_status",
    "note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/submission_bundle")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def validation_failures(validation_rows: list[dict[str, str]]) -> set[str]:
    return {row.get("relative_path", "") for row in validation_rows if row.get("status") == "FAIL"}


def create_manifest_rows(root: Path, output_dir: Path) -> list[dict[str, Any]]:
    candidates = read_csv_rows(output_dir / "submission_candidates.csv")
    validation = read_csv_rows(output_dir / "submission_candidate_validation.csv")
    failed_paths = validation_failures(validation)
    provenance_status, _payload, _errors = load_verified_provenance(root)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        rel = candidate.get("relative_path", "")
        include_candidate = bool_from_csv(candidate.get("include_candidate", "false"))
        path = root / rel
        runtime = is_runtime_output_path(rel)
        row_provenance = provenance_status if runtime else "not_required"
        include = include_candidate and rel not in failed_paths and path.exists()
        if runtime and row_provenance != "verified_real_lab":
            include = False
        note = "Validált csomagolási jelölt." if include else "Nem kerül a ZIP-be."
        rows.append(
            {
                "relative_path": rel,
                "group": candidate.get("group", ""),
                "file_size_bytes": path.stat().st_size if path.exists() else "",
                "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
                "include_in_zip": include,
                "provenance_status": row_provenance,
                "note": note,
            }
        )
    return rows


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    ensure_dir(output_dir)
    write_csv(output_dir / "submission_manifest.csv", rows, FIELDS)
    write_markdown_report(
        output_dir / "submission_manifest.md",
        "Submission bundle manifest",
        rows,
        FIELDS,
        "A manifest a ténylegesen ZIP-be kerülő fájlokat és SHA256 hash-eket rögzíti.",
    )
    write_json(
        output_dir / "submission_manifest.json",
        {
            "file_count": len(rows),
            "include_in_zip_count": sum(1 for row in rows if row["include_in_zip"]),
            "rows": rows,
        },
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = create_manifest_rows(Path(args.root), output_dir)
    write_outputs(rows, output_dir)
    print(f"[OK] Submission manifest: {output_dir / 'submission_manifest.csv'}")


if __name__ == "__main__":
    main()

