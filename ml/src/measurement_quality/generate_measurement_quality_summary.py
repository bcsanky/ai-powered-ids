from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from ml.src.measurement_quality.common import ensure_output_dir, now_utc, read_json_optional, write_json, write_rows_csv


CHECK_FILES = [
    "scenario_coverage.csv",
    "feature_alert_alignment.csv",
    "metric_consistency.csv",
    "ttd_quality.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/measurement_quality")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return [{"check_id": path.stem, "category": "Input", "status": "FAIL", "message": f"hiányzó quality check: {path.name}", "recommendation": "", "value": ""}]
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def measurement_status(rows: list[dict[str, str]], claim_category: str | None) -> str:
    fail_count = sum(row.get("status") in {"FAIL", "NOT_READY"} for row in rows)
    warn_count = sum(row.get("status") == "WARN" for row in rows)
    if fail_count:
        return "MEASUREMENT_NOT_READY"
    if claim_category == "INSUFFICIENT_MEASUREMENT":
        return "MEASUREMENT_NOT_READY"
    if claim_category == "CLAIM_NOT_SUPPORTED" or warn_count >= 5:
        return "MEASUREMENT_WEAK"
    if warn_count or claim_category == "TRADEOFF_ONLY":
        return "MEASUREMENT_USABLE_WITH_LIMITATIONS"
    return "MEASUREMENT_STRONG"


def write_markdown(path: Path, status: str, rows: list[dict[str, str]], claim: dict[str, Any]) -> None:
    lines = [
        "# Mérési minőségi összefoglaló",
        "",
        f"Mérési minőségi státusz: **{status}**",
        "",
        f"Kutatási állítás kategória: **{claim.get('claim_category', 'nincs adat')}**",
        "",
        "Ez az összefoglaló nem új mérési eredmény, hanem a meglévő verified real-lab kimenetek minőségi ellenőrzése.",
        "",
        "| Forrás | Ellenőrzés | Státusz | Üzenet |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row.get('source_file', '')} | {row.get('check_id', '')} | {row.get('status', '')} | {row.get('message', '')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_summary(output_dir: Path) -> dict[str, Any]:
    ensure_output_dir(output_dir)
    rows: list[dict[str, str]] = []
    for filename in CHECK_FILES:
        for row in read_rows(output_dir / filename):
            rows.append({"source_file": filename, **row})
    claim = read_json_optional(output_dir / "research_claim_strength.json") or {"claim_category": "INSUFFICIENT_MEASUREMENT"}
    status = measurement_status(rows, claim.get("claim_category"))
    md_path = output_dir / "measurement_quality_summary.md"
    csv_path = output_dir / "measurement_quality_summary.csv"
    json_path = output_dir / "measurement_quality_summary.json"
    write_markdown(md_path, status, rows, claim)
    write_rows_csv(
        csv_path,
        [
            {
                "check_id": "measurement_quality_summary",
                "category": "summary",
                "status": status,
                "message": f"{len(rows)} ellenőrzési sor alapján",
                "recommendation": "",
                "value": claim.get("claim_category", "nincs adat"),
            }
        ],
    )
    payload = {
        "created_at": now_utc(),
        "measurement_quality_status": status,
        "claim_category": claim.get("claim_category"),
        "fail_count": sum(row.get("status") in {"FAIL", "NOT_READY"} for row in rows),
        "warn_count": sum(row.get("status") == "WARN" for row in rows),
        "rows": rows,
    }
    write_json(json_path, payload)
    return {"status": status, "markdown": md_path, "csv": csv_path, "json": json_path, "rows": rows}


def main() -> None:
    args = parse_args()
    result = run_summary(Path(args.output_dir))
    print(f"[OK] Kimenet: {result['markdown']}")


if __name__ == "__main__":
    main()

