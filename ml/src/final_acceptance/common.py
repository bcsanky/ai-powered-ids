from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


STATUSES = ("PASS", "WARN", "FAIL", "SKIP")


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
        "message": message,
        "recommendation": recommendation,
        "path": path,
    }


def file_exists_row(check_id: str, path: Path, category: str, required: bool = True) -> dict[str, str]:
    if path.exists():
        return check_row(check_id, category, "PASS", f"rendben: {path}", path=path.as_posix())
    return check_row(
        check_id,
        category,
        "FAIL" if required else "WARN",
        f"hiányzik: {path}",
        "Hozd létre vagy futtasd a kapcsolódó előkészítő lépést.",
        path.as_posix(),
    )


def read_text_optional(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = ["check_id", "category", "status", "message", "recommendation", "path"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


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


def write_markdown_report(path: Path, title: str, rows: list[dict[str, str]], intro: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status = aggregate_status(rows)
    lines = [f"# {title}", "", f"Összesített státusz: **{status}**"]
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


def write_check_outputs(
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
    payload: dict[str, Any] = {
        "created_at": now_utc(),
        "status": aggregate_status(rows),
        "checks": rows,
    }
    if extra_payload:
        payload.update(extra_payload)
    write_json(json_path, payload)
    return {"status": payload["status"], "rows": rows, "markdown": md, "csv": csv_path, "json": json_path}


def makefile_targets(makefile_path: Path) -> set[str]:
    if not makefile_path.exists():
        return set()
    targets: set[str] = set()
    pattern = re.compile(r"^([A-Za-z0-9_.-]+):(?:\s|$)")
    for line in makefile_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("\t") or line.startswith("."):
            continue
        match = pattern.match(line)
        if match and not match.group(1).isupper():
            targets.add(match.group(1))
    return targets


def makefile_target_body(makefile_path: Path, target: str) -> list[str]:
    if not makefile_path.exists():
        return []
    lines = makefile_path.read_text(encoding="utf-8").splitlines()
    body: list[str] = []
    in_target = False
    header = f"{target}:"
    for line in lines:
        if line.startswith(header):
            in_target = True
            continue
        if in_target:
            if line and not line.startswith("\t") and not line.startswith(" "):
                break
            if line.startswith("\t"):
                body.append(line.strip())
    return body


def command_sequence_contains(body: list[str], expected: list[str]) -> bool:
    index = 0
    for line in body:
        if index < len(expected) and expected[index] in line:
            index += 1
    return index == len(expected)


def run_command(args: list[str], cwd: Path, timeout_seconds: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=timeout_seconds, check=False)
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
        }

