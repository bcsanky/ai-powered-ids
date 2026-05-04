from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_FILES = [
    ("project_makefile", "Projektstruktúra", "Makefile", "Makefile szükséges a mérési célok futtatásához."),
    (
        "ae_minimal_config",
        "Projektstruktúra",
        "experiments/final/ae_minimal.yaml",
        "Az AE-Minimal konfiguráció szükséges a reprodukálhatósághoz.",
    ),
]
MODEL_WARN_FILES = [
    (
        "ae_minimal_model_dir",
        "Projektstruktúra",
        "artifacts/final/final-ae-minimal-v1",
        "A modellfájlok hiánya csak akkor probléma, ha lab AE scoring is fut.",
    ),
    (
        "ae_minimal_preprocess",
        "Projektstruktúra",
        "data/processed/final/ae_minimal/preprocess.pkl",
        "A preprocess fájl hiánya csak akkor probléma, ha lab AE scoring is fut.",
    ),
]
TEMPLATE_FILES = [
    "templates/lab/lab_ground_truth_template.csv",
    "templates/lab/lab_features_template.csv",
    "templates/lab/lab_scenarios_template.yaml",
]
DOC_FILES = [
    "docs/lab_attack_scenarios_runbook.md",
    "docs/final_real_measurement_checklist.md",
    "docs/real_hybrid_lab_evaluation_runbook.md",
    "docs/real_measurement_export_and_packaging_runbook.md",
]
INPUT_DIRS = ["data/lab", "data/wazuh", "data/processed", "reports"]
OPTIONAL_INPUTS = [
    "data/lab/lab_ground_truth.csv",
    "data/lab/lab_features.csv",
    "data/lab/zeek/conn.log",
    "data/lab/flows.csv",
    "data/wazuh/alerts.jsonl",
]
OPENSEARCH_ENV_VARS = ["OPENSEARCH_URL", "OPENSEARCH_USERNAME", "OPENSEARCH_PASSWORD"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/real_measurement_qa")
    parser.add_argument("--root", default=".")
    parser.add_argument("--require-opensearch", action="store_true")
    return parser.parse_args()


def row(
    check_id: str,
    category: str,
    status: str,
    message: str,
    path: str = "",
    recommendation: str = "",
) -> dict[str, str]:
    return {
        "check_id": check_id,
        "category": category,
        "status": status,
        "message": message,
        "path": path,
        "recommendation": recommendation,
    }


def check_required_file(root: Path, check_id: str, category: str, rel_path: str, recommendation: str) -> dict[str, str]:
    path = root / rel_path
    if path.exists():
        return row(check_id, category, "PASS", "létezik", rel_path, "")
    return row(check_id, category, "FAIL", "hiányzik", rel_path, recommendation)


def check_warning_file(root: Path, check_id: str, category: str, rel_path: str, recommendation: str) -> dict[str, str]:
    path = root / rel_path
    if path.exists():
        return row(check_id, category, "PASS", "létezik", rel_path, "")
    return row(check_id, category, "WARN", "hiányzik", rel_path, recommendation)


def ensure_input_dir(root: Path, rel_path: str) -> dict[str, str]:
    path = root / rel_path
    if path.exists():
        return row(f"dir_{rel_path.replace('/', '_')}", "Input könyvtárak", "PASS", "létezik", rel_path, "")
    path.mkdir(parents=True, exist_ok=True)
    return row(
        f"dir_{rel_path.replace('/', '_')}",
        "Input könyvtárak",
        "WARN",
        "a könyvtár hiányzott, ezért létrejött",
        rel_path,
        "Ellenőrizd, hogy a lab mérési bemenetek a megfelelő könyvtárba kerülnek-e.",
    )


def check_environment(require_opensearch: bool) -> list[dict[str, str]]:
    rows = []
    for name in OPENSEARCH_ENV_VARS:
        exists = bool(os.environ.get(name))
        if require_opensearch:
            status = "PASS" if exists else "FAIL"
            message = "beállítva" if exists else "hiányzik"
            recommendation = "" if exists else f"Állítsd be a(z) {name} környezeti változót."
        else:
            status = "PASS" if exists else "WARN"
            message = "beállítva" if exists else "nincs beállítva, de nem kötelező"
            recommendation = "" if exists else "Csak OpenSearch export futtatásakor szükséges."
        rows.append(row(f"env_{name.lower()}", "Környezeti változók", status, message, name, recommendation))
    return rows


def write_report(rows: list[dict[str, str]], output_path: Path, overall_status: str) -> None:
    lines = [
        "# Real-lab preflight ellenőrzés",
        "",
        f"Összesített státusz: **{overall_status}**",
        "",
        "| Ellenőrzés | Kategória | Státusz | Üzenet | Útvonal | Javaslat |",
        "|---|---|---|---|---|---|",
    ]
    for item in rows:
        lines.append(
            f"| {item['check_id']} | {item['category']} | {item['status']} | "
            f"{item['message']} | {item['path']} | {item['recommendation']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_preflight(root: Path, output_dir: Path, *, require_opensearch: bool = False) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for check_id, category, rel_path, recommendation in PROJECT_FILES:
        rows.append(check_required_file(root, check_id, category, rel_path, recommendation))
    for check_id, category, rel_path, recommendation in MODEL_WARN_FILES:
        rows.append(check_warning_file(root, check_id, category, rel_path, recommendation))
    for rel_path in TEMPLATE_FILES:
        rows.append(
            check_required_file(
                root,
                f"template_{Path(rel_path).stem}",
                "Lab template-ek",
                rel_path,
                "Futtasd a make lab-templates parancsot.",
            )
        )
    for rel_path in DOC_FILES:
        rows.append(
            check_required_file(
                root,
                f"doc_{Path(rel_path).stem}",
                "Dokumentáció",
                rel_path,
                "Pótold a real-lab mérési runbook dokumentációt.",
            )
        )
    for rel_path in INPUT_DIRS:
        rows.append(ensure_input_dir(root, rel_path))
    for rel_path in OPTIONAL_INPUTS:
        rows.append(
            check_warning_file(
                root,
                f"optional_{Path(rel_path).stem}",
                "Opcionális inputok",
                rel_path,
                "A tényleges mérés előtt vagy közben kell előállítani.",
            )
        )
    rows.extend(check_environment(require_opensearch))

    overall_status = "FAIL" if any(item["status"] == "FAIL" for item in rows) else "PASS"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "preflight_summary.csv"
    report_path = output_dir / "preflight_report.md"
    metadata_path = output_dir / "preflight_metadata.json"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    write_report(rows, report_path, overall_status)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "output_dir": str(output_dir),
        "require_opensearch": require_opensearch,
        "overall_status": overall_status,
        "fail_count": sum(item["status"] == "FAIL" for item in rows),
        "warn_count": sum(item["status"] == "WARN" for item in rows),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "overall_status": overall_status,
        "summary": summary_path,
        "report": report_path,
        "metadata": metadata_path,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    result = run_preflight(Path(args.root), Path(args.output_dir), require_opensearch=args.require_opensearch)
    for key in ["summary", "report", "metadata"]:
        print(f"[OK] Kimenet: {result[key]}")
    if result["overall_status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
