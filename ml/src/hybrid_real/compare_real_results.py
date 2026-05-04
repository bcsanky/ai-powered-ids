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
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "false_negative_rate",
    "alert_count",
    "mean_ttd",
    "n_samples",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wazuh-metrics", default="results/wazuh_real/metrics_summary.csv")
    parser.add_argument("--ae-metrics", default="results/ae_lab/metrics_summary.csv")
    parser.add_argument("--hybrid-metrics", default="results/hybrid_real/metrics_summary.csv")
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
        "precision": numeric_value(row, "precision"),
        "recall": numeric_value(row, "recall"),
        "f1": numeric_value(row, "f1"),
        "false_positive_rate": numeric_value(row, "false_positive_rate"),
        "false_negative_rate": numeric_value(row, "false_negative_rate"),
        "alert_count": numeric_value(row, "alert_count"),
        "mean_ttd": numeric_value(row, "mean_ttd"),
        "n_samples": numeric_value(row, "n_samples"),
    }


def collect_real_comparison(
    *,
    wazuh_metrics_path: Path,
    ae_metrics_path: Path,
    hybrid_metrics_path: Path,
) -> pd.DataFrame:
    wazuh = read_metrics(wazuh_metrics_path)
    ae = read_metrics(ae_metrics_path)
    hybrid = read_metrics(hybrid_metrics_path)
    rows = [
        comparison_row("Wazuh-only", wazuh.iloc[0]),
        comparison_row("AE-Minimal lab", ae.iloc[0]),
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
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def format_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_real_comparison(
    *,
    wazuh_metrics_path: Path,
    ae_metrics_path: Path,
    hybrid_metrics_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    comparison = collect_real_comparison(
        wazuh_metrics_path=wazuh_metrics_path,
        ae_metrics_path=ae_metrics_path,
        hybrid_metrics_path=hybrid_metrics_path,
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
        "output_dir": str(output_dir),
        "configurations": comparison["configuration"].tolist(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path, "metadata": metadata_path}


def main() -> None:
    args = parse_args()
    outputs = save_real_comparison(
        wazuh_metrics_path=Path(args.wazuh_metrics),
        ae_metrics_path=Path(args.ae_metrics),
        hybrid_metrics_path=Path(args.hybrid_metrics),
        output_dir=Path(args.output_dir),
    )
    for path in outputs.values():
        print(f"[OK] Kimenet: {path}")


if __name__ == "__main__":
    main()
