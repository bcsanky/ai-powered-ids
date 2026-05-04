from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATUSES = ("PASS", "WARN", "FAIL", "SKIP")
FORBIDDEN_PATH_PARTS = {"examples", "templates", "tests"}
FORBIDDEN_NAME_TOKENS = ("sample", "demo", "fixture", "dummy", "fake")


@dataclass(frozen=True)
class CommandResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    missing_executable: bool = False


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "igen"}:
        return True
    if text in {"0", "false", "no", "n", "nem"}:
        return False
    return default


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


def readiness_status(rows: list[dict[str, str]]) -> str:
    if has_fail(rows):
        return "NOT_READY"
    if has_warn(rows):
        return "READY_WITH_WARNINGS"
    return "READY_FOR_LAB_EXECUTION"


def redact_secret(text: str, secret: str | None) -> str:
    if not secret:
        return text
    return text.replace(secret, "[REDACTED]")


def run_command(args: list[str], timeout_seconds: int = 20, cwd: Path | None = None) -> CommandResult:
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)
    except FileNotFoundError as exc:
        return CommandResult(None, "", str(exc), missing_executable=True)
    except subprocess.TimeoutExpired as exc:
        return CommandResult(None, exc.stdout or "", exc.stderr or "", timed_out=True)


def read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


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
            f"| {escape_md(row.get('check_id', ''))} | {escape_md(row.get('category', ''))} | "
            f"{escape_md(row.get('status', ''))} | {escape_md(row.get('message', ''))} | "
            f"{escape_md(row.get('recommendation', ''))} | {escape_md(row.get('path', ''))} |"
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
    markdown_path = output_dir / f"{basename}.md"
    csv_path = output_dir / f"{basename}.csv"
    json_path = output_dir / f"{basename}.json"
    write_markdown_report(markdown_path, title, rows, intro)
    write_rows_csv(csv_path, rows)
    payload: dict[str, Any] = {
        "created_at": now_utc(),
        "status": aggregate_status(rows),
        "checks": rows,
    }
    if extra_payload:
        payload.update(extra_payload)
    write_json(json_path, payload)
    return {"status": payload["status"], "rows": rows, "markdown": markdown_path, "csv": csv_path, "json": json_path}


def path_guard_status(path: Path) -> tuple[bool, str]:
    text = path.as_posix().lower()
    parts = set(Path(text).parts)
    if FORBIDDEN_PATH_PARTS & parts:
        return False, "az útvonal demo/sablon/teszt könyvtár alatt van"
    for token in FORBIDDEN_NAME_TOKENS:
        if token in text:
            return False, f"az útvonal tiltott névrészletet tartalmaz: {token}"
    return True, ""


def summarize_command_result(result: CommandResult, secret: str | None = None, max_chars: int = 240) -> str:
    if result.timed_out:
        return "időtúllépés"
    if result.missing_executable:
        return "parancs nem található"
    text = (result.stdout or result.stderr or "").strip()
    text = redact_secret(text, secret)
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text

