from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from ml.src.repo_hygiene.common import load_provenance, validate_provenance_payload


STATUSES = ("PASS", "WARN", "FAIL", "SKIP")
DEFAULT_PROVENANCE_PATH = Path("reports/real_measurement/measurement_provenance.json")
DEFAULT_RUNTIME_PREFIXES = (
    "reports/real_measurement/",
    "reports/real_measurement_qa/",
    "reports/measurement_quality/",
    "reports/live_integration/",
    "reports/thesis_integration/",
    "reports/final_submission_check/",
    "reports/final_acceptance/",
    "reports/live_smoke/",
    "reports/lab_session/",
    "reports/wazuh_export/",
    "results/real_comparison/",
    "results/wazuh_real/",
    "results/ae_lab/",
    "results/hybrid_real/",
)
DEMO_PATH_PARTS = {"examples", "templates", "tests"}
DEMO_NAME_TOKENS = ("sample", "demo", "fixture")
SENSITIVE_EXTENSIONS = {".pcap", ".pcapng", ".pem", ".key", ".crt", ".p12", ".jks"}
SENSITIVE_FILE_NAMES = {".env"}


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "\\|").replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_markdown_report(
    path: Path,
    title: str,
    rows: list[dict[str, Any]],
    columns: list[str],
    intro: str = "",
    status: str | None = None,
) -> None:
    ensure_dir(path.parent)
    lines = [f"# {title}", ""]
    if status:
        lines.extend([f"Összesített státusz: **{status}**", ""])
    if intro:
        lines.extend([intro, ""])
    lines.append(markdown_table(rows, columns))
    path.write_text("\n".join(lines), encoding="utf-8")


def status_from_rows(rows: list[dict[str, Any]]) -> str:
    statuses = [str(row.get("status", "")) for row in rows]
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    if statuses and all(status == "SKIP" for status in statuses):
        return "SKIP"
    return "PASS"


def bool_from_csv(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "igen"}


def normalize_relative(path: str | Path, root: Path = Path(".")) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        text = candidate.as_posix()
        if text.startswith("./"):
            text = text[2:]
        return text


def is_safe_relative_path(path: str | Path) -> bool:
    text = Path(path).as_posix()
    return not text.startswith("/") and ".." not in Path(text).parts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_matches_pattern(relative_path: str, pattern: str) -> bool:
    normalized = relative_path.replace("\\", "/")
    pattern = pattern.replace("\\", "/")
    if fnmatch.fnmatch(normalized, pattern):
        return True
    return fnmatch.fnmatch(Path(normalized).name, pattern)


def file_matches_any(relative_path: str, patterns: Iterable[str]) -> bool:
    return any(file_matches_pattern(relative_path, pattern) for pattern in patterns)


def iter_existing_files(root: Path, pattern: str) -> list[Path]:
    path = root / pattern
    if pattern.endswith("/**"):
        base = root / pattern[:-3]
        files = [candidate for candidate in base.rglob("*") if candidate.is_file()] if base.exists() else []
    elif any(char in pattern for char in "*?[]"):
        files = [candidate for candidate in root.glob(pattern) if candidate.is_file()]
    elif path.is_file():
        files = [path]
    elif path.is_dir():
        files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    else:
        files = []
    return sorted(set(files))


def path_parts(path: str | Path) -> set[str]:
    return {part.lower() for part in Path(path).as_posix().split("/")}


def is_demo_or_fixture_path(path: str | Path) -> bool:
    text = Path(path).as_posix().lower()
    if DEMO_PATH_PARTS & path_parts(text):
        return True
    return any(token in text for token in DEMO_NAME_TOKENS)


def is_sensitive_path(path: str | Path, policy: dict[str, Any] | None = None) -> bool:
    text = Path(path).as_posix()
    lower = text.lower()
    name = Path(text).name.lower()
    if name in SENSITIVE_FILE_NAMES or name.startswith(".env."):
        return True
    if Path(text).suffix.lower() in SENSITIVE_EXTENSIONS:
        return True
    if "__pycache__" in path_parts(text) or ".pytest_cache" in path_parts(text):
        return True
    if lower.startswith(("data/", "raw/", "artifacts/", ".venv/", ".venv39/")):
        return True
    if lower.startswith("infra/wazuh/certs/") or "/data/" in lower and lower.startswith("infra/wazuh/"):
        return True
    if lower == "reports/real_measurement_redacted/redaction_mapping.json":
        return True
    if policy:
        if file_matches_any(text, policy.get("always_exclude", [])):
            return True
        for pattern in policy.get("sensitive_patterns", []):
            if pattern.lower() in lower:
                return True
    return False


def is_runtime_output_path(path: str | Path, policy: dict[str, Any] | None = None) -> bool:
    text = Path(path).as_posix()
    runtime_patterns = policy.get("runtime_outputs_allowed_if_verified", []) if policy else []
    return text.startswith(DEFAULT_RUNTIME_PREFIXES) or file_matches_any(text, runtime_patterns)


def load_verified_provenance(root: Path = Path("."), provenance_path: Path = DEFAULT_PROVENANCE_PATH) -> tuple[str, dict[str, Any] | None, list[str]]:
    path = provenance_path if provenance_path.is_absolute() else root / provenance_path
    payload = load_provenance(path)
    valid, errors = validate_provenance_payload(payload)
    return ("verified_real_lab" if valid else "missing_provenance", payload, errors)


def runtime_candidate_allowed(provenance_status: str, policy: dict[str, Any]) -> bool:
    require = bool(policy.get("rules", {}).get("require_provenance_for_runtime_results", True))
    return not require or provenance_status == "verified_real_lab"


def git_value(args: list[str], root: Path) -> str:
    try:
        completed = subprocess.run(args, cwd=root, text=True, capture_output=True, check=False, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def create_zip(zip_path: Path, files: list[tuple[Path, str]]) -> None:
    ensure_dir(zip_path.parent)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for source, archive_name in files:
            zf.write(source, archive_name)
