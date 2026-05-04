from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.thesis_integration.common import (
    DEFAULT_PROVENANCE,
    comparison_ready_checks,
    ensure_output_dir,
    label_coverage_from_frames,
    provenance_validation,
    read_csv_optional,
    status_from_rows,
    write_json,
    write_rows_csv,
)


REQUIRED_INPUTS = [
    "reports/real_measurement/measurement_provenance.json",
    "results/real_comparison/metrics_comparison.csv",
    "results/wazuh_real/metrics_summary.csv",
    "results/ae_lab/metrics_summary.csv",
    "results/hybrid_real/metrics_summary.csv",
]
OPTIONAL_INPUTS = [
    "reports/real_measurement_qa/research_question_answer.json",
    "reports/real_measurement_qa/thesis_readiness.md",
    "reports/live_integration/live_integration_readiness.json",
    "reports/live_integration/enrichment_summary.csv",
    "reports/live_integration/dashboard_summary.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/thesis_integration")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def row(check_id: str, category: str, status: str, message: str, recommendation: str = "") -> dict[str, str]:
    return {
        "check_id": check_id,
        "category": category,
        "status": status,
        "message": message,
        "recommendation": recommendation,
    }


def file_checks(root: Path) -> list[dict[str, str]]:
    rows = []
    for rel_path in REQUIRED_INPUTS:
        path = root / rel_path
        rows.append(
            row(
                f"required_{Path(rel_path).stem}",
                "Bemenetek",
                "PASS" if path.exists() else "FAIL",
                f"rendelkezésre áll: {rel_path}" if path.exists() else f"hiányzik: {rel_path}",
                "" if path.exists() else "A dolgozati integráció csak a teljes real-lab mérési csomag után futtatható.",
            )
        )
    for rel_path in OPTIONAL_INPUTS:
        path = root / rel_path
        rows.append(
            row(
                f"optional_{Path(rel_path).stem}",
                "Opcionális bemenetek",
                "PASS" if path.exists() else "INFO",
                f"rendelkezésre áll: {rel_path}" if path.exists() else f"nem elérhető: {rel_path}",
                "" if path.exists() else "A hozzá tartozó szakasz korlátozott vagy kihagyható.",
            )
        )
    return rows


def provenance_checks(root: Path) -> list[dict[str, str]]:
    _, valid, errors = provenance_validation(root / DEFAULT_PROVENANCE)
    if valid:
        return [row("verified_provenance", "Adateredet", "PASS", "verified real_lab provenance rendelkezésre áll")]
    return [
        row(
            "verified_provenance",
            "Adateredet",
            "FAIL",
            "nincs végleges dolgozati felhasználásra alkalmas provenance",
            "; ".join(errors),
        )
    ]


def metric_checks(root: Path) -> list[dict[str, str]]:
    comparison = read_csv_optional(root / "results/real_comparison/metrics_comparison.csv")
    wazuh = read_csv_optional(root / "results/wazuh_real/metrics_summary.csv")
    ae = read_csv_optional(root / "results/ae_lab/metrics_summary.csv")
    hybrid = read_csv_optional(root / "results/hybrid_real/metrics_summary.csv")
    rows = comparison_ready_checks(comparison)
    rows.append(label_coverage_from_frames([comparison, wazuh, ae, hybrid]))
    if comparison is not None and not comparison.empty and "alert_count" in comparison.columns:
        alerts = pd.to_numeric(comparison["alert_count"], errors="coerce")
        valid = alerts.notna().all() and (alerts >= 0).all()
        rows.append(
            row(
                "alert_count_non_negative",
                "Metrikák",
                "PASS" if valid else "FAIL",
                "alert_count nem negatív" if valid else "hibás alert_count érték",
            )
        )
    return rows


def write_report(rows: list[dict[str, str]], output_path: Path, status: str) -> None:
    lines = [
        "# Dolgozati integrációs bemenetellenőrzés",
        "",
        f"Státusz: **{status}**",
        "",
        "| Ellenőrzés | Kategória | Státusz | Üzenet | Javaslat |",
        "|---|---|---|---|---|",
    ]
    for item in rows:
        lines.append(
            f"| {item['check_id']} | {item['category']} | {item['status']} | {item['message']} | {item.get('recommendation', '')} |"
        )
    lines.extend(
        [
            "",
            "A dolgozati fejezetrészek végleges eredményként csak READY státusz és verified real_lab provenance mellett használhatók.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_check(root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    rows = file_checks(root) + provenance_checks(root) + metric_checks(root)
    status = status_from_rows(rows)
    write_report(rows, output_dir / "thesis_input_check.md", status)
    write_rows_csv(
        output_dir / "thesis_input_check.csv",
        rows,
        ["check_id", "category", "status", "message", "recommendation"],
    )
    payload = {"status": status, "checks": rows}
    write_json(output_dir / "thesis_input_check.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    result = run_check(Path(args.root), Path(args.output_dir))
    print(f"[OK] thesis input check status: {result['status']}")
    if result["status"] == "NOT_READY":
        sys.exit(1)


if __name__ == "__main__":
    main()
