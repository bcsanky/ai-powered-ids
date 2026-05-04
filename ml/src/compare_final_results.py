from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIGS = [
    ("ae_minimal", "final-ae-minimal-v1"),
    ("ae_context", "final-ae-context-v1"),
    ("baseline_stat", "final-baseline-stat-v1"),
    ("rule_proxy", "final-rule-proxy-v1"),
    ("hybrid", "final-hybrid-v1"),
    ("baseline_wazuh_real", "final-baseline-wazuh-v1"),
]

OUTPUT_COLUMNS = [
    "config_name",
    "status",
    "run_dir",
    "threshold_name",
    "threshold_value",
    "tn",
    "fp",
    "fn",
    "tp",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "false_negative_rate",
    "true_positive_rate",
    "true_negative_rate",
    "alert_count",
    "roc_auc",
    "n_samples",
    "n_attack",
    "n_benign",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/final")
    parser.add_argument("--output-dir", default="results/final/comparison")
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def find_latest_run(config_dir: Path) -> Path | None:
    if not config_dir.exists():
        return None

    candidates = [
        p for p in config_dir.iterdir()
        if p.is_dir() and (p / "metrics_summary.csv").exists()
    ]
    if not candidates:
        return None

    return sorted(candidates, key=lambda p: p.name)[-1]


def first_matching_row(df: pd.DataFrame, mask: pd.Series) -> pd.Series | None:
    matches = df.loc[mask]
    if matches.empty:
        return None
    return matches.iloc[0]


def select_metric_row(df: pd.DataFrame) -> pd.Series:
    if "threshold_name" in df.columns:
        threshold_names = df["threshold_name"].fillna("").astype(str)

        row = first_matching_row(df, threshold_names == "f1_optimum")
        if row is not None:
            return row

        row = first_matching_row(df, threshold_names.str.contains("percentile_95", regex=False))
        if row is not None:
            return row

        row = first_matching_row(df, threshold_names == "fixed")
        if row is not None:
            return row

    return df.iloc[0]


def value_or_none(row: pd.Series, column: str) -> Any:
    if column not in row.index:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def int_or_none(value: Any) -> int | None:
    number = numeric_or_none(value)
    if number is None:
        return None
    return int(number)


def safe_rate(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def complete_metrics(row: pd.Series, config_name: str, run_dir: Path) -> dict:
    tn = int_or_none(value_or_none(row, "tn"))
    fp = int_or_none(value_or_none(row, "fp"))
    fn = int_or_none(value_or_none(row, "fn"))
    tp = int_or_none(value_or_none(row, "tp"))

    benign_total = None if tn is None or fp is None else tn + fp
    attack_total = None if tp is None or fn is None else tp + fn

    threshold_name = value_or_none(row, "threshold_name")
    if threshold_name is None:
        threshold_name = value_or_none(row, "baseline")
    if threshold_name is None:
        threshold_name = ""

    threshold_value = value_or_none(row, "threshold_value")
    if threshold_value is None:
        threshold_value = value_or_none(row, "threshold")

    false_positive_rate = numeric_or_none(value_or_none(row, "false_positive_rate"))
    if false_positive_rate is None:
        false_positive_rate = safe_rate(fp, benign_total)

    false_negative_rate = numeric_or_none(value_or_none(row, "false_negative_rate"))
    if false_negative_rate is None:
        false_negative_rate = safe_rate(fn, attack_total)

    true_positive_rate = numeric_or_none(value_or_none(row, "true_positive_rate"))
    if true_positive_rate is None:
        true_positive_rate = safe_rate(tp, attack_total)

    true_negative_rate = numeric_or_none(value_or_none(row, "true_negative_rate"))
    if true_negative_rate is None:
        true_negative_rate = safe_rate(tn, benign_total)

    alert_count = int_or_none(value_or_none(row, "alert_count"))
    if alert_count is None and tp is not None and fp is not None:
        alert_count = tp + fp

    n_samples = int_or_none(value_or_none(row, "n_samples"))
    if n_samples is None and None not in {tn, fp, fn, tp}:
        n_samples = int(tn + fp + fn + tp)

    n_attack = int_or_none(value_or_none(row, "n_attack"))
    if n_attack is None and attack_total is not None:
        n_attack = attack_total

    n_benign = int_or_none(value_or_none(row, "n_benign"))
    if n_benign is None and benign_total is not None:
        n_benign = benign_total

    return {
        "config_name": config_name,
        "status": "ok",
        "run_dir": str(run_dir),
        "threshold_name": str(threshold_name),
        "threshold_value": numeric_or_none(threshold_value),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": numeric_or_none(value_or_none(row, "precision")),
        "recall": numeric_or_none(value_or_none(row, "recall")),
        "f1": numeric_or_none(value_or_none(row, "f1")),
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "true_positive_rate": true_positive_rate,
        "true_negative_rate": true_negative_rate,
        "alert_count": alert_count,
        "roc_auc": numeric_or_none(value_or_none(row, "roc_auc")),
        "n_samples": n_samples,
        "n_attack": n_attack,
        "n_benign": n_benign,
    }


def missing_row(config_name: str, reason: str) -> dict:
    row = {column: None for column in OUTPUT_COLUMNS}
    row["config_name"] = config_name
    row["status"] = "missing"
    row["threshold_name"] = reason
    return row


def collect_results(results_root: Path) -> pd.DataFrame:
    rows = []
    for config_name, dirname in CONFIGS:
        config_dir = results_root / dirname
        run_dir = find_latest_run(config_dir)
        if run_dir is None:
            warn(f"Nincs metrics_summary.csv ehhez a konfigurációhoz: {dirname}")
            rows.append(missing_row(config_name, "metrics_summary.csv missing"))
            continue

        metrics_path = run_dir / "metrics_summary.csv"
        metrics_df = pd.read_csv(metrics_path)
        if metrics_df.empty:
            warn(f"Üres metrics_summary.csv: {metrics_path}")
            rows.append(missing_row(config_name, "metrics_summary.csv empty"))
            continue

        selected = select_metric_row(metrics_df)
        rows.append(complete_metrics(selected, config_name, run_dir))

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def save_markdown_table(df: pd.DataFrame, path: Path) -> None:
    table_columns = [
        "config_name",
        "status",
        "threshold_name",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
        "alert_count",
        "roc_auc",
        "n_samples",
    ]
    table = df[table_columns].copy()
    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    for _, row in table.iterrows():
        values = [format_markdown_value(row[column]) for column in table.columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_markdown_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def ok_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"] == "ok"].copy()


def save_precision_recall_f1(df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = ok_rows(df)
    if plot_df.empty:
        warn("Precision/recall/F1 ábra nem készül: nincs sikeres konfiguráció.")
        return

    x = np.arange(len(plot_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(x - width, plot_df["precision"], width=width, label="Precision")
    ax.bar(x, plot_df["recall"], width=width, label="Recall")
    ax.bar(x + width, plot_df["f1"], width=width, label="F1")
    ax.set_title("Precision, recall és F1 összehasonlítása")
    ax.set_ylabel("Metrika értéke")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x, labels=plot_df["config_name"], rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig_comparison_precision_recall_f1.png", dpi=160)
    plt.close(fig)


def save_single_metric_bar(
    df: pd.DataFrame,
    output_dir: Path,
    column: str,
    filename: str,
    title: str,
    ylabel: str,
) -> None:
    plot_df = ok_rows(df)
    if plot_df.empty:
        warn(f"{title} ábra nem készül: nincs sikeres konfiguráció.")
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(plot_df["config_name"], plot_df[column].fillna(0))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)


def save_figures(df: pd.DataFrame, output_dir: Path) -> None:
    save_precision_recall_f1(df, output_dir)
    save_single_metric_bar(
        df,
        output_dir,
        column="false_positive_rate",
        filename="fig_comparison_false_positive_rate.png",
        title="Hamis pozitív arány",
        ylabel="Hamis pozitív arány",
    )
    save_single_metric_bar(
        df,
        output_dir,
        column="alert_count",
        filename="fig_comparison_alert_count.png",
        title="Riasztások száma",
        ylabel="Riasztások",
    )


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = collect_results(results_root)
    comparison.to_csv(output_dir / "metrics_comparison.csv", index=False)
    save_markdown_table(comparison, output_dir / "metrics_comparison.md")
    save_figures(comparison, output_dir)

    print(f"[OK] Összehasonlítás mentve ide: {output_dir}")


if __name__ == "__main__":
    main()
