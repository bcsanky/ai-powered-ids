from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.repo_hygiene.common import is_demo_or_fixture_path


STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def check_row(
    check_id: str,
    category: str,
    status: str,
    message: str,
    recommendation: str = "",
) -> dict[str, str]:
    return {
        "check_id": check_id,
        "category": category,
        "status": status,
        "message": message,
        "recommendation": recommendation,
    }


def has_fail(rows: list[dict[str, str]]) -> bool:
    return any(row["status"] == "FAIL" for row in rows)


def has_warn(rows: list[dict[str, str]]) -> bool:
    return any(row["status"] == "WARN" for row in rows)


def overall_status(rows: list[dict[str, str]]) -> str:
    if has_fail(rows):
        return "FAIL"
    if has_warn(rows):
        return "WARN"
    return "PASS"


def write_check_report(
    rows: list[dict[str, str]],
    output_path: Path,
    *,
    title: str,
    intro: str,
) -> None:
    lines = [
        f"# {title}",
        "",
        intro,
        "",
        f"Összesített státusz: **{overall_status(rows)}**",
        "",
        "| Ellenőrzés | Kategória | Státusz | Üzenet | Javaslat |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in sorted(rows, key=lambda item: (STATUS_ORDER[item["status"]], item["category"], item["check_id"])):
        lines.append(
            f"| {row['check_id']} | {row['category']} | {row['status']} | "
            f"{row['message']} | {row['recommendation']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    pd.DataFrame(
        rows,
        columns=["check_id", "category", "status", "message", "recommendation"],
    ).to_csv(output_path, index=False)


def assert_real_lab_path(path: Path, label: str) -> None:
    if is_demo_or_fixture_path(path):
        raise ValueError(f"{label} nem lehet demo/sample/fixture eredetű útvonal: {path}")

