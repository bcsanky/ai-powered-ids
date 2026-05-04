from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.repo_hygiene.common import is_demo_or_fixture_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/live_integration")
    return parser.parse_args()


def read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó enrichment summary: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Üres enrichment summary: {path}")
    return df.iloc[0].to_dict()


def read_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó live integration metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def add_check(checks: list[dict[str, Any]], check_id: str, status: str, message: str) -> None:
    checks.append({"check_id": check_id, "status": status, "message": message})


def status_rank(status: str) -> int:
    return {"PASS": 0, "WARN": 1, "FAIL": 2}[status]


def numeric_int(value: Any, default: int = 0) -> int:
    converted = pd.to_numeric(value, errors="coerce")
    if pd.isna(converted):
        return default
    return int(converted)


def overall_readiness(checks: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    if any(check["status"] == "FAIL" for check in checks):
        return "NOT_READY"
    scored = numeric_int(summary.get("scored_alerts", 0))
    total = numeric_int(summary.get("total_alerts", 0))
    unmatched = numeric_int(summary.get("unmatched_alerts", 0))
    if scored <= 0:
        return "NOT_READY"
    if total > 0 and unmatched / total > 0.5:
        return "READY_WITH_LIMITATIONS"
    if any(check["status"] == "WARN" for check in checks):
        return "READY_WITH_LIMITATIONS"
    return "READY"


def validation_markdown(readiness: str, checks: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Live integration validáció",
        "",
        f"Státusz: **{readiness}**",
        "",
        "## Összefoglaló",
        "",
        f"- Összes alert: {summary.get('total_alerts', 0)}",
        f"- Pontozott alert: {summary.get('scored_alerts', 0)}",
        f"- Unmatched alert: {summary.get('unmatched_alerts', 0)}",
        f"- Hibrid pozitív döntés: {summary.get('hybrid_positive_count', 0)}",
        "",
        "## Ellenőrzések",
        "",
        "| Ellenőrzés | Státusz | Üzenet |",
        "| --- | --- | --- |",
    ]
    for check in sorted(checks, key=lambda item: (status_rank(item["status"]), item["check_id"])):
        lines.append(f"| {check['check_id']} | {check['status']} | {check['message']} |")
    lines.extend(
        [
            "",
            "## Korlátok",
            "",
            "- A live integration kimenet integrációs demonstráció, nem önálló benchmark.",
            "- A dashboard-ready kimenet csak verified real-lab provenance mellett használható dolgozati bizonyítékként.",
            "- Az unmatched arányt minden értelmezésnél közölni kell.",
        ]
    )
    return "\n".join(lines) + "\n"


def validate_outputs(output_dir: Path) -> dict[str, Path]:
    enriched_path = output_dir / "enriched_alerts.csv"
    summary_path = output_dir / "enrichment_summary.csv"
    metadata_path = output_dir / "run_metadata.json"
    checks: list[dict[str, Any]] = []

    if not enriched_path.exists():
        add_check(checks, "enriched_csv", "FAIL", f"Hiányzó enriched alert CSV: {enriched_path}")
        summary: dict[str, Any] = {}
    else:
        enriched = pd.read_csv(enriched_path)
        if enriched.empty:
            add_check(checks, "enriched_csv", "FAIL", "Az enriched alert CSV üres.")
        else:
            add_check(checks, "enriched_csv", "PASS", "Az enriched alert CSV létezik és nem üres.")

    try:
        summary = read_summary(summary_path)
        add_check(checks, "summary", "PASS", "Az enrichment summary olvasható.")
    except Exception as exc:  # noqa: BLE001
        summary = {}
        add_check(checks, "summary", "FAIL", str(exc))

    try:
        metadata = read_metadata(metadata_path)
        add_check(checks, "metadata", "PASS", "A run metadata olvasható.")
    except Exception as exc:  # noqa: BLE001
        metadata = {}
        add_check(checks, "metadata", "FAIL", str(exc))

    for key in ["wazuh_alerts", "lab_features", "ground_truth"]:
        value = str(metadata.get(key, ""))
        if value and is_demo_or_fixture_path(value):
            add_check(checks, f"{key}_path", "FAIL", f"Demo/sample/fixture eredetű input: {value}")

    provenance_errors = metadata.get("provenance_errors") or []
    if any("demo/sample/fixture" in str(error) for error in provenance_errors):
        add_check(checks, "provenance", "FAIL", "A provenance demo/sample/fixture eredetű inputot jelez.")
    elif metadata and not metadata.get("provenance_valid", False):
        add_check(
            checks,
            "provenance",
            "WARN",
            "A provenance hiányzik vagy nem teljes; a kimenet csak korlátozással értelmezhető.",
        )
    elif metadata:
        add_check(checks, "provenance", "PASS", "A provenance érvényesnek jelölt.")

    scored = numeric_int(summary.get("scored_alerts", 0))
    total = numeric_int(summary.get("total_alerts", 0))
    unmatched = numeric_int(summary.get("unmatched_alerts", 0))
    if scored <= 0:
        add_check(checks, "scored_alerts", "FAIL", "Nincs ML pontszámmal ellátott alert.")
    else:
        add_check(checks, "scored_alerts", "PASS", f"Pontozott alert darabszám: {scored}")
    if total > 0 and unmatched / total > 0.5:
        add_check(checks, "unmatched_ratio", "WARN", "Az unmatched arány meghaladja az 50%-ot.")
    elif total > 0:
        add_check(checks, "unmatched_ratio", "PASS", "Az unmatched arány elfogadható.")

    for key in ["critical_count", "high_count", "medium_count", "normal_count"]:
        if key not in summary:
            add_check(checks, key, "FAIL", f"Hiányzó summary mező: {key}")
        else:
            add_check(checks, key, "PASS", f"{key}: {summary[key]}")

    readiness = overall_readiness(checks, summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "live_integration_validation.md"
    csv_path = output_dir / "live_integration_validation.csv"
    json_path = output_dir / "live_integration_readiness.json"
    pd.DataFrame(checks).to_csv(csv_path, index=False)
    md_path.write_text(validation_markdown(readiness, checks, summary), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "readiness": readiness,
                "summary": summary,
                "checks": checks,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {"markdown": md_path, "csv": csv_path, "json": json_path}


def main() -> None:
    args = parse_args()
    outputs = validate_outputs(Path(args.output_dir))
    for path in outputs.values():
        print(f"[OK] {path}")


if __name__ == "__main__":
    main()
