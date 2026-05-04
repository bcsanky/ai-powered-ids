from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.repo_hygiene.common import load_provenance, provenance_status_for_path, validate_provenance_payload


DEFAULT_ROOTS = [
    "examples/lab",
    "examples/scoring",
    "data/lab",
    "data/wazuh",
    "data/processed/wazuh_real",
    "data/processed/lab_ae",
    "results/wazuh_real",
    "results/ae_lab",
    "results/hybrid_real",
    "results/real_comparison",
    "results/performance",
    "reports/lab",
    "reports/final",
    "reports/performance",
    "reports/lab_input_validation",
    "reports/wazuh_export",
    "reports/real_measurement",
    "figures/final",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/real_measurement")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def category_for(path: Path) -> str:
    text = path.as_posix()
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".svg"}:
        return "figure"
    if "metrics" in path.name or "comparison" in path.name:
        return "metric"
    if suffix in {".md", ".html"}:
        return "report"
    if path.name.endswith("metadata.json") or path.name == "run_metadata.json":
        return "metadata"
    if text.startswith("data/lab") or text.startswith("data/wazuh"):
        return "raw_input"
    if text.startswith("data/processed"):
        return "intermediate"
    return "validated_input"


def include_in_submission(path: Path, category: str, provenance_status: str) -> tuple[bool, str]:
    text = path.as_posix()
    suffix = path.suffix.lower()
    if text.startswith("examples/"):
        return False, "Demonstrációs input; nem real-lab mérési eredmény."
    if text.startswith("reports/lab/") or text.startswith("reports/final/"):
        return False, "Demonstrációs vagy offline kimenet; real-lab eredményként nem használható."
    if suffix in {".pcap", ".pcapng"}:
        return False, "Nagy nyers forgalmi állomány; külön kezelendő."
    if path.name.startswith("alerts") and suffix in {".json", ".jsonl"}:
        return False, "Wazuh alert export érzékeny adatot tartalmazhat."
    if "redaction_mapping" in path.name:
        return False, "Anonimizálási mapping érzékeny adat."
    if text.startswith("reports/real_measurement") or text.startswith("results/real_comparison"):
        if provenance_status == "verified_real_lab":
            return True, "Provenance alapján real-lab méréshez köthető eredményfájl."
        return False, "Provenance hiányában nem kerülhet real-lab beadási csomagba."
    if category in {"metric", "figure", "report", "metadata"}:
        return True, "Dolgozati melléklethez használható eredményfájl."
    return False, "Input vagy köztes állomány; szükség szerint külön mellékelhető."


def iter_files(root: Path, relative_root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.resolve() != relative_root.resolve())


def build_manifest(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    provenance = load_provenance(root / "reports/real_measurement/measurement_provenance.json")
    validate_provenance_payload(provenance)
    output_manifest_names = {
        "measurement_manifest.csv",
        "measurement_manifest.md",
        "measurement_manifest.json",
    }
    for rel_root in DEFAULT_ROOTS:
        scan_root = root / rel_root
        for path in iter_files(scan_root, root):
            rel_path = path.relative_to(root)
            if rel_path.name in output_manifest_names:
                continue
            category = category_for(rel_path)
            provenance_status = provenance_status_for_path(rel_path, provenance)
            include, note = include_in_submission(rel_path, category, provenance_status)
            rows.append(
                {
                    "relative_path": rel_path.as_posix(),
                    "file_size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "category": category,
                    "provenance_status": provenance_status,
                    "include_in_submission": include,
                    "note": note,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "relative_path",
            "file_size_bytes",
            "sha256",
            "category",
            "provenance_status",
            "include_in_submission",
            "note",
        ],
    )


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Real-lab mérési manifest",
        "",
        "| relative_path | file_size_bytes | sha256 | category | provenance_status | include_in_submission | note |",
        "|---|---:|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["relative_path"]),
                    str(row["file_size_bytes"]),
                    str(row["sha256"]),
                    str(row["category"]),
                    str(row["provenance_status"]),
                    str(row["include_in_submission"]),
                    str(row["note"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_manifest(root: Path, output_dir: Path) -> dict[str, Path]:
    df = build_manifest(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "measurement_manifest.csv"
    md_path = output_dir / "measurement_manifest.md"
    json_path = output_dir / "measurement_manifest.json"
    df.to_csv(csv_path, index=False)
    write_markdown(df, md_path)
    json_path.write_text(json.dumps(df.to_dict(orient="records"), indent=2, ensure_ascii=False), encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path, "json": json_path}


def main() -> None:
    args = parse_args()
    outputs = create_manifest(Path(args.root), Path(args.output_dir))
    for path in outputs.values():
        print(f"[OK] Kimenet: {path}")


if __name__ == "__main__":
    main()
