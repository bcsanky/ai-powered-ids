from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", default="results/real_comparison/metrics_comparison.csv")
    parser.add_argument("--output-dir", default="results/real_comparison")
    return parser.parse_args()


def read_comparison(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó valós lab összehasonlító CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Üres valós lab összehasonlító CSV: {path}")
    return df


def save_precision_recall_f1(df: pd.DataFrame, output_dir: Path) -> None:
    x = np.arange(len(df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width, df["precision"], width=width, label="Precision")
    ax.bar(x, df["recall"], width=width, label="Recall")
    ax.bar(x + width, df["f1"], width=width, label="F1")
    ax.set_title("Precision, recall és F1 összehasonlítása")
    ax.set_ylabel("Metrika értéke")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x, labels=df["configuration"], rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig_precision_recall_f1.png", dpi=160)
    plt.close(fig)


def save_bar(df: pd.DataFrame, output_dir: Path, column: str, filename: str, title: str, ylabel: str) -> None:
    plot_df = df[["configuration", column]].copy()
    plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=[column])
    if plot_df.empty:
        print(f"[WARN] Nincs ábrázolható adat ehhez: {column}")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(plot_df["configuration"], plot_df[column])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)


def plot_real_comparison(comparison_path: Path, output_dir: Path) -> list[Path]:
    df = read_comparison(comparison_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_precision_recall_f1(df, output_dir)
    save_bar(
        df,
        output_dir,
        "false_positive_rate",
        "fig_false_positive_rate.png",
        "Hamis pozitív arány összehasonlítása",
        "Hamis pozitív arány",
    )
    save_bar(
        df,
        output_dir,
        "alert_count",
        "fig_alert_count.png",
        "Riasztásszám összehasonlítása",
        "Riasztások száma",
    )
    save_bar(
        df,
        output_dir,
        "mean_ttd",
        "fig_mean_ttd.png",
        "Átlagos detektálási idő összehasonlítása",
        "Másodperc",
    )
    return [
        output_dir / "fig_precision_recall_f1.png",
        output_dir / "fig_false_positive_rate.png",
        output_dir / "fig_alert_count.png",
        output_dir / "fig_mean_ttd.png",
    ]


def main() -> None:
    args = parse_args()
    paths = plot_real_comparison(Path(args.comparison), Path(args.output_dir))
    for path in paths:
        if path.exists():
            print(f"[OK] Kimenet: {path}")


if __name__ == "__main__":
    main()
