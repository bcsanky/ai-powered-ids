from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from ml.src.live_smoke.common import ensure_output_dir, now_utc, readiness_status, read_json_optional, write_json, write_rows_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/live_smoke")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def collect_check_rows(output_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(output_dir.glob("*_check.csv")):
        for row in read_rows(path):
            rows.append({"source_file": path.name, **row})
    return rows


def write_readiness_markdown(path: Path, status: str, rows: list[dict[str, str]], extra: dict[str, Any]) -> None:
    lines = [
        "# Live környezeti readiness összefoglaló",
        "",
        f"Readiness státusz: **{status}**",
        "",
        "Ez a státusz nem jelenti azt, hogy a mérés elkészült. Csak azt mutatja, hogy a környezet technikailag készen állhat a lab mérés megkezdésére.",
        "",
    ]
    if extra:
        lines.extend(["## Kapcsolódó QA státuszok", ""])
        for key, value in extra.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.extend(
        [
            "## Ellenőrzések",
            "",
            "| Forrás | Ellenőrzés | Státusz | Üzenet |",
            "|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.get('source_file', '')} | {row.get('check_id', '')} | {row.get('status', '')} | {row.get('message', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_readiness(output_dir: Path) -> dict[str, Any]:
    ensure_output_dir(output_dir)
    rows = collect_check_rows(output_dir)
    if not rows:
        rows = [
            {
                "source_file": "",
                "check_id": "live_smoke_inputs",
                "category": "readiness",
                "status": "FAIL",
                "message": "nem található live smoke check CSV",
                "recommendation": "Futtasd a live-smoke ellenőrzéseket.",
                "path": "",
            }
        ]
    status = readiness_status(rows)
    final_acceptance = read_json_optional(Path("reports/final_acceptance/release_candidate_readiness.json")) or {}
    final_submission = read_json_optional(Path("reports/final_submission_check/final_submission_readiness.json")) or {}
    extra = {
        "final_acceptance": final_acceptance.get("readiness_status") or final_acceptance.get("status", "nincs adat"),
        "final_submission": final_submission.get("readiness_status") or final_submission.get("status", "nincs adat"),
    }
    md_path = output_dir / "live_smoke_readiness.md"
    csv_path = output_dir / "live_smoke_readiness.csv"
    json_path = output_dir / "live_smoke_readiness.json"
    write_readiness_markdown(md_path, status, rows, extra)
    write_rows_csv(
        csv_path,
        [
            {
                "check_id": "live_smoke_readiness",
                "category": "readiness",
                "status": status,
                "message": f"{len(rows)} ellenőrzési sor alapján",
                "recommendation": "",
                "path": output_dir.as_posix(),
            }
        ],
    )
    write_json(
        json_path,
        {
            "created_at": now_utc(),
            "readiness_status": status,
            "rows": rows,
            "related_statuses": extra,
        },
    )
    return {"status": status, "rows": rows, "markdown": md_path, "csv": csv_path, "json": json_path}


def main() -> None:
    args = parse_args()
    result = run_readiness(Path(args.output_dir))
    print(f"[OK] Kimenet: {result['markdown']}")


if __name__ == "__main__":
    main()

