from __future__ import annotations

import csv
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


STATUSES = ("PASS", "WARN", "FAIL", "NOT_APPLICABLE")
STATUS_HU = {
    "PASS": "rendben",
    "WARN": "figyelmeztetés",
    "FAIL": "hiba",
    "NOT_APPLICABLE": "nem alkalmazható",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_row(
    check_id: str,
    category: str,
    status: str,
    message: str,
    recommendation: str = "",
    path: str = "",
) -> dict[str, str]:
    if status not in STATUSES:
        raise ValueError(f"Ismeretlen státusz: {status}")
    return {
        "check_id": check_id,
        "category": category,
        "status": status,
        "status_hu": STATUS_HU[status],
        "message": message,
        "recommendation": recommendation,
        "path": path,
    }


def has_fail(rows: list[dict[str, str]]) -> bool:
    return any(row.get("status") == "FAIL" for row in rows)


def has_warn(rows: list[dict[str, str]]) -> bool:
    return any(row.get("status") == "WARN" for row in rows)


def aggregate_status(rows: list[dict[str, str]]) -> str:
    if has_fail(rows):
        return "FAIL"
    if has_warn(rows):
        return "WARN"
    return "PASS"


def file_exists(path: Path) -> bool:
    return path.exists()


def resolve_evidence(root: Path, evidence: str) -> list[Path]:
    if evidence == "uploaded thesis document":
        return []
    if any(char in evidence for char in "*?[]"):
        return [Path(match) for match in glob.glob((root / evidence).as_posix(), recursive=True)]
    path = root / evidence
    return [path] if path.exists() else []


def evidence_kind(evidence: str) -> str:
    if evidence == "uploaded thesis document":
        return "manual"
    if evidence.startswith("reports/") or evidence.startswith("results/"):
        return "runtime"
    return "repository"


def read_text_optional(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def read_csv_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = ["check_id", "category", "status", "status_hu", "message", "recommendation", "path"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown_report(path: Path, title: str, rows: list[dict[str, str]], intro: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", f"Összesített státusz: **{aggregate_status(rows)}**"]
    if intro:
        lines.extend(["", intro])
    lines.extend(
        [
            "",
            "| Ellenőrzés | Kategória | Státusz | Üzenet | Javaslat | Útvonal |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.get('check_id', '')} | {row.get('category', '')} | {row.get('status', '')} | "
            f"{row.get('message', '')} | {row.get('recommendation', '')} | {row.get('path', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    *,
    output_dir: Path,
    basename: str,
    title: str,
    rows: list[dict[str, str]],
    intro: str = "",
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    md = output_dir / f"{basename}.md"
    csv_path = output_dir / f"{basename}.csv"
    json_path = output_dir / f"{basename}.json"
    write_markdown_report(md, title, rows, intro)
    write_rows_csv(csv_path, rows)
    payload: dict[str, Any] = {"created_at": now_utc(), "status": aggregate_status(rows), "checks": rows}
    if extra_payload:
        payload.update(extra_payload)
    write_json(json_path, payload)
    return {"status": payload["status"], "rows": rows, "markdown": md, "csv": csv_path, "json": json_path}


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"

