from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ml.src.submission_bundle.common import (
    bool_from_csv,
    create_zip,
    ensure_dir,
    git_value,
    is_sensitive_path,
    load_verified_provenance,
    now_utc,
    read_csv_rows,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="reports/submission_bundle/submission_manifest.csv")
    parser.add_argument("--output", default="dist/submission/ai_powered_ids_submission_bundle.zip")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def selected_manifest_files(root: Path, manifest_path: Path) -> list[tuple[Path, str]]:
    rows = read_csv_rows(manifest_path)
    files: list[tuple[Path, str]] = []
    for row in rows:
        if not bool_from_csv(row.get("include_in_zip", "false")):
            continue
        rel = row.get("relative_path", "")
        if is_sensitive_path(rel):
            raise ValueError(f"Tiltott fájl kerülne ZIP-be: {rel}")
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(f"Manifestben szereplő fájl nem létezik: {rel}")
        files.append((path, rel))
    readme = root / "dist/submission/SUBMISSION_README.md"
    if readme.exists():
        files.append((readme, "SUBMISSION_README.md"))
    return files


def create_submission_zip(root: Path, manifest_path: Path, output_path: Path) -> dict[str, Any]:
    files = selected_manifest_files(root, manifest_path)
    if not files:
        raise ValueError("Nincs csomagolható fájl; üres ZIP nem készülhet.")
    for source, arcname in files:
        if is_sensitive_path(arcname):
            raise ValueError(f"Tiltott fájl kerülne ZIP-be: {arcname}")
        if not source.exists():
            raise FileNotFoundError(source)
    create_zip(output_path, files)
    zip_sha = sha256_file(output_path)
    provenance_status, _payload, provenance_errors = load_verified_provenance(root)
    metadata = {
        "created_at": now_utc(),
        "git_branch": git_value(["git", "branch", "--show-current"], root),
        "git_commit": git_value(["git", "rev-parse", "HEAD"], root),
        "file_count": len(files),
        "zip_path": output_path.as_posix(),
        "zip_sha256": zip_sha,
        "provenance_status": provenance_status,
        "warning_count": len(provenance_errors) if provenance_status != "verified_real_lab" else 0,
    }
    metadata_path = output_path.parent / "zip_metadata.json"
    write_json(metadata_path, metadata)
    return metadata


def write_report(metadata: dict[str, Any], output_dir: Path) -> None:
    ensure_dir(output_dir)
    status = "PASS" if metadata.get("file_count", 0) > 0 else "FAIL"
    md = output_dir / "submission_zip_report.md"
    md.write_text(
        "\n".join(
            [
                "# Submission ZIP riport",
                "",
                f"Státusz: **{status}**",
                f"ZIP: `{metadata.get('zip_path')}`",
                f"Fájlok száma: {metadata.get('file_count')}",
                f"SHA256: `{metadata.get('zip_sha256')}`",
                f"Provenance státusz: {metadata.get('provenance_status')}",
                "",
                "A ZIP forráskódot, konfigurációt, dokumentációt, sablonokat és engedélyezett demo példákat tartalmaz. "
                "Runtime mérési output csak verified real_lab provenance mellett kerülhet bele.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(output_dir / "submission_zip_report.json", {"status": status, **metadata})


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output)
    ensure_dir(output.parent)
    metadata = create_submission_zip(root, root / args.manifest, output)
    write_report(metadata, root / "reports/submission_bundle")
    print(f"[OK] Submission ZIP: {output}")


if __name__ == "__main__":
    main()

