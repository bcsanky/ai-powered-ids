from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "configuration",
    "TP",
    "FP",
    "TN",
    "FN",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "false_negative_rate",
    "alert_count",
    "mean_ttd",
    "n_samples",
    "n_attack",
    "n_benign",
]
COUNT_OUTPUT_COLUMNS = ["TP", "FP", "TN", "FN", "alert_count", "n_samples", "n_attack", "n_benign"]
CONTROL_BASELINE_NAME = "All-positive baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wazuh-metrics", default="results/wazuh_real/metrics_summary.csv")
    parser.add_argument("--ae-metrics", default="results/ae_lab/metrics_summary.csv")
    parser.add_argument("--hybrid-metrics", default="results/hybrid_real/metrics_summary.csv")
    parser.add_argument("--ground-truth", default="data/lab/lab_ground_truth.csv")
    parser.add_argument("--output-dir", default="results/real_comparison")
    return parser.parse_args()


def read_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó metrika CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Üres metrika CSV: {path}")
    return df


def numeric_value(row: pd.Series, column: str) -> Any:
    if column not in row.index or pd.isna(row[column]):
        return np.nan
    value = row[column]
    if isinstance(value, np.generic):
        return value.item()
    return value


def comparison_row(configuration: str, row: pd.Series) -> dict[str, Any]:
    return {
        "configuration": configuration,
        "TP": numeric_value(row, "TP"),
        "FP": numeric_value(row, "FP"),
        "TN": numeric_value(row, "TN"),
        "FN": numeric_value(row, "FN"),
        "precision": numeric_value(row, "precision"),
        "recall": numeric_value(row, "recall"),
        "f1": numeric_value(row, "f1"),
        "false_positive_rate": numeric_value(row, "false_positive_rate"),
        "false_negative_rate": numeric_value(row, "false_negative_rate"),
        "alert_count": numeric_value(row, "alert_count"),
        "mean_ttd": numeric_value(row, "mean_ttd"),
        "n_samples": numeric_value(row, "n_samples"),
        "n_attack": numeric_value(row, "n_attack"),
        "n_benign": numeric_value(row, "n_benign"),
    }


def safe_rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


def read_ground_truth_labels(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó ground truth CSV az all-positive baseline számításához: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Üres ground truth CSV az all-positive baseline számításához: {path}")
    if "label" not in df.columns:
        raise ValueError("A ground truth CSV nem tartalmaz label oszlopot.")
    labels = df["label"].astype(str).str.strip().str.lower()
    allowed = {"attack", "benign"}
    unknown = sorted(set(labels) - allowed)
    if unknown:
        raise ValueError("Ismeretlen ground truth label érték(ek): " + ", ".join(unknown))
    return labels


def all_positive_baseline_row(ground_truth_path: Path) -> dict[str, Any]:
    labels = read_ground_truth_labels(ground_truth_path)
    n_attack = int((labels == "attack").sum())
    n_benign = int((labels == "benign").sum())
    tp = n_attack
    fp = n_benign
    tn = 0
    fn = 0
    precision = safe_rate(tp, tp + fp)
    recall = safe_rate(tp, tp + fn)
    f1 = safe_rate(2 * precision * recall, precision + recall)
    return {
        "configuration": CONTROL_BASELINE_NAME,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": safe_rate(fp, fp + tn),
        "false_negative_rate": safe_rate(fn, fn + tp),
        "alert_count": int(tp + fp),
        "mean_ttd": np.nan,
        "n_samples": int(len(labels)),
        "n_attack": n_attack,
        "n_benign": n_benign,
    }


def collect_real_comparison(
    *,
    wazuh_metrics_path: Path,
    ae_metrics_path: Path,
    hybrid_metrics_path: Path,
    ground_truth_path: Path,
) -> pd.DataFrame:
    wazuh = read_metrics(wazuh_metrics_path)
    ae = read_metrics(ae_metrics_path)
    hybrid = read_metrics(hybrid_metrics_path)
    rows = [
        comparison_row("Wazuh-only", wazuh.iloc[0]),
        comparison_row("AE-Minimal lab", ae.iloc[0]),
        all_positive_baseline_row(ground_truth_path),
    ]
    strategy_names = {
        "hybrid_or": "Hybrid OR",
        "hybrid_weighted": "Hybrid weighted",
        "hybrid_priority": "Hybrid priority",
    }
    if "strategy" not in hybrid.columns:
        raise ValueError("A hybrid metrics_summary.csv nem tartalmaz strategy oszlopot.")
    for strategy, label in strategy_names.items():
        matched = hybrid[hybrid["strategy"] == strategy]
        if matched.empty:
            raise ValueError(f"Hiányzó hibrid stratégia a metrika CSV-ben: {strategy}")
        rows.append(comparison_row(label, matched.iloc[0]))
    comparison = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    for column in COUNT_OUTPUT_COLUMNS:
        comparison[column] = pd.to_numeric(comparison[column], errors="coerce").astype("Int64")
    return comparison


def format_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def save_markdown(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(format_value(row[col]) for col in df.columns) + " |")
    if CONTROL_BASELINE_NAME in set(df["configuration"].astype(str)):
        lines.extend(
            [
                "",
                "Módszertani megjegyzés: az All-positive baseline naiv kontrollsor, amely minden eseményt pozitívnak jelöl. "
                "Nem Wazuh-, Zeek-, AE- vagy hibrid detektor, hanem a mérési értelmezéshez használt triviális viszonyítási alap.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_real_comparison(
    *,
    wazuh_metrics_path: Path,
    ae_metrics_path: Path,
    hybrid_metrics_path: Path,
    ground_truth_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    comparison = collect_real_comparison(
        wazuh_metrics_path=wazuh_metrics_path,
        ae_metrics_path=ae_metrics_path,
        hybrid_metrics_path=hybrid_metrics_path,
        ground_truth_path=ground_truth_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics_comparison.csv"
    md_path = output_dir / "metrics_comparison.md"
    metadata_path = output_dir / "run_metadata.json"
    comparison.to_csv(csv_path, index=False)
    save_markdown(comparison, md_path)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "wazuh_metrics": str(wazuh_metrics_path),
        "ae_metrics": str(ae_metrics_path),
        "hybrid_metrics": str(hybrid_metrics_path),
        "ground_truth": str(ground_truth_path),
        "output_dir": str(output_dir),
        "configurations": comparison["configuration"].tolist(),
        "methodological_note": "Az All-positive baseline naiv kontrollsor, nem valós detektor.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path, "metadata": metadata_path}


def main() -> None:
    args = parse_args()
    outputs = save_real_comparison(
        wazuh_metrics_path=Path(args.wazuh_metrics),
        ae_metrics_path=Path(args.ae_metrics),
        hybrid_metrics_path=Path(args.hybrid_metrics),
        ground_truth_path=Path(args.ground_truth),
        output_dir=Path(args.output_dir),
    )
    for path in outputs.values():
        print(f"[OK] Kimenet: {path}")


if __name__ == "__main__":
    main()
