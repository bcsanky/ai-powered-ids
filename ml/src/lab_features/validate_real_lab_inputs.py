from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.lab_ae_eval.validate_lab_features import validate_lab_features
from ml.src.wazuh_baseline.build_ground_truth import validate_ground_truth
from ml.src.wazuh_baseline.parse_wazuh_alerts import load_alert_objects, normalize_alert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--wazuh-alerts", required=True)
    parser.add_argument("--output-dir", default="reports/lab_input_validation")
    return parser.parse_args()


def status_row(check: str, status: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "detail": detail}


def validate_wazuh_alert_input(path: Path) -> tuple[int, str]:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó Wazuh alert input: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            raise ValueError("A parsed Wazuh alert CSV-nek tartalmaznia kell timestamp oszlopot.")
        return len(df), "parsed_csv"
    alerts = load_alert_objects(path)
    normalized = [normalize_alert(alert) for alert in alerts if isinstance(alert, dict)]
    return len(normalized), "json_or_jsonl"


def write_report(rows: list[dict[str, str]], output_path: Path) -> None:
    lines = [
        "# Lab input validációs jelentés",
        "",
        "| Ellenőrzés | Státusz | Részlet |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['check']} | {row['status']} | {row['detail']} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_real_lab_inputs(
    *,
    ground_truth_path: Path,
    features_path: Path,
    wazuh_alerts_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    ground_truth = validate_ground_truth(ground_truth_path)
    rows.append(status_row("ground_truth_schema", "ok", f"{len(ground_truth)} esemény"))

    features = validate_lab_features(features_path)
    rows.append(status_row("lab_features_schema", "ok", f"{len(features)} feature sor"))

    truth_ids = set(ground_truth["event_id"].astype(str))
    feature_ids = set(features["event_id"].astype(str))
    if truth_ids != feature_ids:
        raise ValueError("A ground truth és lab_features event_id készlete nem egyezik.")
    rows.append(status_row("event_id_consistency", "ok", f"{len(truth_ids)} egyező event_id"))

    label_counts = ground_truth["label"].value_counts().to_dict()
    if label_counts.get("benign", 0) < 1 or label_counts.get("attack", 0) < 1:
        raise ValueError("Legalább 1 benign és 1 attack esemény szükséges.")
    rows.append(status_row("label_coverage", "ok", json.dumps(label_counts, ensure_ascii=False)))

    scenario_count = int(ground_truth["scenario"].nunique())
    if scenario_count < 2:
        raise ValueError("Legalább 2 különböző scenario szükséges.")
    rows.append(status_row("scenario_coverage", "ok", f"{scenario_count} scenario"))

    alert_count, alert_format = validate_wazuh_alert_input(wazuh_alerts_path)
    rows.append(status_row("wazuh_alert_input", "ok", f"{alert_count} alert sor, formátum: {alert_format}"))

    summary_path = output_dir / "input_validation_summary.csv"
    report_path = output_dir / "input_validation_report.md"
    metadata_path = output_dir / "run_metadata.json"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    write_report(rows, report_path)
    metadata: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ground_truth": str(ground_truth_path),
        "features": str(features_path),
        "wazuh_alerts": str(wazuh_alerts_path),
        "output_dir": str(output_dir),
        "n_events": len(ground_truth),
        "n_alerts": alert_count,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"summary": summary_path, "report": report_path, "metadata": metadata_path}


def main() -> None:
    args = parse_args()
    outputs = validate_real_lab_inputs(
        ground_truth_path=Path(args.ground_truth),
        features_path=Path(args.features),
        wazuh_alerts_path=Path(args.wazuh_alerts),
        output_dir=Path(args.output_dir),
    )
    for path in outputs.values():
        print(f"[OK] Kimenet: {path}")


if __name__ == "__main__":
    main()
