from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ml.src.repo_hygiene.common import load_provenance, validate_provenance_payload


STATUSES = ("PASS", "WARN", "FAIL", "NOT_READY")
FORBIDDEN_PATH_PARTS = {"examples", "templates", "tests"}
FORBIDDEN_NAME_TOKENS = ("sample", "demo", "fixture", "dummy", "fake")
PROVENANCE_PATH = Path("reports/real_measurement/measurement_provenance.json")


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó konfiguráció: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def check_row(
    check_id: str,
    category: str,
    status: str,
    message: str,
    recommendation: str = "",
    value: Any = "",
) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(f"Ismeretlen státusz: {status}")
    return {
        "check_id": check_id,
        "category": category,
        "status": status,
        "message": message,
        "recommendation": recommendation,
        "value": format_value(value),
    }


def has_fail(rows: list[dict[str, Any]]) -> bool:
    return any(row.get("status") in {"FAIL", "NOT_READY"} for row in rows)


def has_warn(rows: list[dict[str, Any]]) -> bool:
    return any(row.get("status") == "WARN" for row in rows)


def aggregate_status(rows: list[dict[str, Any]]) -> str:
    if any(row.get("status") == "NOT_READY" for row in rows):
        return "NOT_READY"
    if any(row.get("status") == "FAIL" for row in rows):
        return "FAIL"
    if has_warn(rows):
        return "WARN"
    return "PASS"


def is_forbidden_real_path(path: str | Path) -> bool:
    text = Path(path).as_posix().lower()
    parts = set(Path(text).parts)
    if FORBIDDEN_PATH_PARTS & parts:
        return True
    return any(token in text for token in FORBIDDEN_NAME_TOKENS)


def validate_verified_provenance(
    provenance_path: Path = PROVENANCE_PATH,
) -> tuple[bool, list[str], dict[str, Any] | None]:
    payload = load_provenance(provenance_path)
    valid, errors = validate_provenance_payload(payload)
    if payload:
        for key in ["ground_truth_path", "lab_features_path", "wazuh_alerts_path"]:
            value = payload.get(key)
            if value and is_forbidden_real_path(value):
                errors.append(f"tiltott real-lab input útvonal: {value}")
    return valid and not errors, errors, payload


def provenance_row(provenance_path: Path = PROVENANCE_PATH) -> dict[str, Any]:
    valid, errors, _ = validate_verified_provenance(provenance_path)
    if valid:
        return check_row("measurement_provenance", "Adateredet", "PASS", "verified real_lab provenance rendelkezésre áll")
    return check_row(
        "measurement_provenance",
        "Adateredet",
        "NOT_READY",
        "verified real_lab provenance hiányzik vagy érvénytelen",
        "; ".join(errors),
    )


def to_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def safe_float(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric)


def safe_int(value: Any) -> int | None:
    numeric = safe_float(value)
    if numeric is None:
        return None
    return int(numeric)


def metric_for(df: pd.DataFrame | None, configuration: str, metric: str) -> float | None:
    if df is None or df.empty or "configuration" not in df.columns or metric not in df.columns:
        return None
    matched = df[df["configuration"].astype(str) == configuration]
    if matched.empty:
        return None
    return safe_float(matched.iloc[0][metric])


def format_value(value: Any) -> str:
    if value is None:
        return "nincs adat"
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "nincs adat"
        return f"{value:.4f}"
    if hasattr(value, "item"):
        return format_value(value.item())
    return str(value)


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = ["check_id", "category", "status", "message", "recommendation", "value"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_markdown_report(path: Path, title: str, rows: list[dict[str, Any]], intro: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", f"Összesített státusz: **{aggregate_status(rows)}**"]
    if intro:
        lines.extend(["", intro])
    lines.extend(
        [
            "",
            "| Ellenőrzés | Kategória | Státusz | Üzenet | Javaslat | Érték |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {escape_md(row.get('check_id', ''))} | {escape_md(row.get('category', ''))} | "
            f"{escape_md(row.get('status', ''))} | {escape_md(row.get('message', ''))} | "
            f"{escape_md(row.get('recommendation', ''))} | {escape_md(row.get('value', ''))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    *,
    output_dir: Path,
    basename: str,
    title: str,
    rows: list[dict[str, Any]],
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


def required_file_dataframe(path: Path, label: str) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    df = read_csv_optional(path)
    if df is None:
        return None, check_row(label, "Bemenet", "FAIL", f"hiányzó CSV: {path}", "Futtasd a real-lab pipeline megfelelő lépését.")
    if df.empty:
        return df, check_row(label, "Bemenet", "FAIL", f"üres CSV: {path}", "Ellenőrizd a mérési pipeline kimenetét.")
    return df, check_row(label, "Bemenet", "PASS", f"CSV olvasható: {path}", value=len(df))
