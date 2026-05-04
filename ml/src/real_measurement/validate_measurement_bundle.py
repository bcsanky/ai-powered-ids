from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_INPUTS = [
    "data/lab/lab_ground_truth.csv",
    "data/lab/lab_features.csv",
    "data/wazuh/alerts.jsonl",
]
REQUIRED_INTERMEDIATE = [
    "data/processed/wazuh_real/alerts_parsed.csv",
    "data/processed/wazuh_real/wazuh_correlated.csv",
    "data/processed/lab_ae/ae_lab_predictions.csv",
]
REQUIRED_RESULTS = [
    "results/wazuh_real/metrics_summary.csv",
    "results/ae_lab/metrics_summary.csv",
    "results/hybrid_real/metrics_summary.csv",
    "results/real_comparison/metrics_comparison.csv",
    "results/real_comparison/metrics_comparison.md",
]
REQUIRED_FIGURES = [
    "results/real_comparison/fig_precision_recall_f1.png",
    "results/real_comparison/fig_false_positive_rate.png",
    "results/real_comparison/fig_alert_count.png",
]
EXPECTED_CONFIGURATIONS = [
    "Wazuh-only",
    "AE-Minimal lab",
    "Hybrid OR",
    "Hybrid weighted",
    "Hybrid priority",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/real_measurement")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def check_row(check: str, status: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "detail": detail}


def csv_non_empty(path: Path) -> bool:
    try:
        return not pd.read_csv(path).empty
    except pd.errors.EmptyDataError:
        return False


def check_file(path: Path, *, required: bool = True, non_empty_csv: bool = False) -> dict[str, str]:
    if not path.exists():
        return check_row(str(path), "FAIL" if required else "WARN", "hiányzik")
    if non_empty_csv and path.suffix.lower() == ".csv" and not csv_non_empty(path):
        return check_row(str(path), "FAIL", "üres CSV")
    return check_row(str(path), "PASS", "létezik")


def write_report(rows: list[dict[str, str]], output_path: Path, overall_status: str) -> None:
    lines = [
        "# Real-lab mérési csomag validáció",
        "",
        f"Összesített státusz: **{overall_status}**",
        "",
        "| Ellenőrzés | Státusz | Részlet |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['check']} | {row['status']} | {row['detail']} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_metric_value(path: Path, column: str) -> Any:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or column not in df.columns:
        return None
    return df.iloc[0][column]


def json_safe(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def validate_measurement_bundle(root: Path, output_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for rel_path in REQUIRED_INPUTS:
        rows.append(check_file(root / rel_path, non_empty_csv=rel_path.endswith(".csv")))
    for rel_path in REQUIRED_INTERMEDIATE:
        rows.append(check_file(root / rel_path, non_empty_csv=True))
    for rel_path in REQUIRED_RESULTS:
        rows.append(check_file(root / rel_path, non_empty_csv=rel_path.endswith(".csv")))
    for rel_path in REQUIRED_FIGURES:
        rows.append(check_file(root / rel_path))
    rows.append(check_file(root / "results/real_comparison/fig_mean_ttd.png", required=False))

    comparison_path = root / "results/real_comparison/metrics_comparison.csv"
    if comparison_path.exists() and csv_non_empty(comparison_path):
        comparison = pd.read_csv(comparison_path)
        configs = set(comparison.get("configuration", pd.Series(dtype=str)).astype(str))
        missing = [name for name in EXPECTED_CONFIGURATIONS if name not in configs]
        status = "FAIL" if missing else "PASS"
        detail = "hiányzó konfigurációk: " + ", ".join(missing) if missing else "minden konfiguráció szerepel"
        rows.append(check_row("metrics_comparison_configurations", status, detail))
    else:
        rows.append(check_row("metrics_comparison_configurations", "FAIL", "metrics_comparison nem olvasható"))

    ground_truth_path = root / "data/lab/lab_ground_truth.csv"
    if ground_truth_path.exists() and csv_non_empty(ground_truth_path):
        gt = pd.read_csv(ground_truth_path)
        labels = gt.get("label", pd.Series(dtype=str)).astype(str).str.lower()
        has_benign = bool((labels == "benign").any())
        has_attack = bool((labels == "attack").any())
        rows.append(
            check_row(
                "label_coverage",
                "PASS" if has_benign and has_attack else "FAIL",
                f"benign={has_benign}, attack={has_attack}",
            )
        )
    else:
        rows.append(check_row("label_coverage", "FAIL", "ground truth nem olvasható"))

    n_values = {
        "wazuh": json_safe(read_metric_value(root / "results/wazuh_real/metrics_summary.csv", "n_samples")),
        "ae": json_safe(read_metric_value(root / "results/ae_lab/metrics_summary.csv", "n_samples")),
        "hybrid": json_safe(read_metric_value(root / "results/hybrid_real/metrics_summary.csv", "n_samples")),
    }
    numeric_values = [int(value) for value in n_values.values() if value is not None and not pd.isna(value)]
    same_n = len(numeric_values) == 3 and len(set(numeric_values)) == 1
    rows.append(check_row("n_samples_consistency", "PASS" if same_n else "FAIL", json.dumps(n_values)))

    overall_status = "PASS" if all(row["status"] != "FAIL" for row in rows) else "FAIL"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "bundle_validation_summary.csv"
    report_path = output_dir / "bundle_validation_report.md"
    metadata_path = output_dir / "run_metadata.json"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    write_report(rows, report_path, overall_status)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "output_dir": str(output_dir),
        "overall_status": overall_status,
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
    result = validate_measurement_bundle(Path(args.root), Path(args.output_dir))
    for key in ["summary", "report", "metadata"]:
        print(f"[OK] Kimenet: {result[key]}")
    if result["overall_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
