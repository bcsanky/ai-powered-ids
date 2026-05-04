from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEMO_PATH_PARTS = {"examples", "templates", "tests"}
DEMO_NAME_TOKENS = ("fixture", "sample", "demo")
PROVENANCE_PATH = Path("reports/real_measurement/measurement_provenance.json")
REAL_LAB_INPUT_KEYS = ["ground_truth_path", "lab_features_path", "wazuh_alerts_path"]
REQUIRED_INPUT_HASH_KEYS = ["ground_truth_sha256", "lab_features_sha256", "wazuh_alerts_sha256"]
TRACKED_GENERATED_OUTPUT_PREFIXES = (
    "reports/final/",
    "reports/lab/",
    "reports/performance/",
    "reports/real_measurement_qa/",
    "reports/lab_session/",
    "reports/thesis_integration/",
    "reports/final_acceptance/",
    "reports/final_submission_check/",
    "reports/live_smoke/",
    "reports/measurement_quality/",
    "reports/submission_bundle/",
    "results/performance/",
    "dist/submission/",
    "figures/final/",
)
TRACKED_GENERATED_OUTPUT_FILES = ("reports/scored_events.jsonl",)
ALLOWED_TRACKED_OUTPUT_NAMES = {"README.md", ".gitkeep"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_relative(path: Path, root: Path = Path(".")) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def is_demo_or_fixture_path(path: str | Path) -> bool:
    text = Path(path).as_posix().lower()
    parts = set(Path(text).parts)
    if DEMO_PATH_PARTS & parts:
        return True
    return any(token in text for token in DEMO_NAME_TOKENS)


def is_allowed_tracked_output_placeholder(path: str | Path) -> bool:
    return Path(path).name in ALLOWED_TRACKED_OUTPUT_NAMES


def is_forbidden_tracked_generated_output(path: str | Path) -> bool:
    text = Path(path).as_posix()
    if is_allowed_tracked_output_placeholder(text):
        return False
    if text in TRACKED_GENERATED_OUTPUT_FILES:
        return True
    return any(text.startswith(prefix) for prefix in TRACKED_GENERATED_OUTPUT_PREFIXES)


def load_provenance(path: Path = PROVENANCE_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_provenance_payload(payload: dict[str, Any] | None) -> tuple[bool, list[str]]:
    if payload is None:
        return False, ["measurement_provenance.json hiányzik"]
    errors = []
    if payload.get("measurement_source") != "real_lab":
        errors.append("measurement_source nem real_lab")
    for key in REAL_LAB_INPUT_KEYS:
        value = payload.get(key)
        if not value:
            errors.append(f"hiányzó input path: {key}")
        elif is_demo_or_fixture_path(str(value)):
            errors.append(f"demo/sample/fixture eredetű input: {value}")
    for key in REQUIRED_INPUT_HASH_KEYS:
        if not payload.get(key):
            errors.append(f"hiányzó input SHA256: {key}")
    return not errors, errors


def provenance_status_for_path(path: Path, payload: dict[str, Any] | None = None) -> str:
    text = path.as_posix()
    valid, _ = validate_provenance_payload(payload)
    if text.startswith("examples/") or text.startswith("reports/lab/") or text.startswith("reports/final/"):
        return "demo_or_generated"
    if text.startswith("reports/real_measurement") or text.startswith("results/real_comparison"):
        return "verified_real_lab" if valid else "missing_provenance"
    if text.startswith("results/wazuh_real") or text.startswith("results/ae_lab") or text.startswith("results/hybrid_real"):
        return "verified_real_lab" if valid else "missing_provenance"
    return "unknown"
