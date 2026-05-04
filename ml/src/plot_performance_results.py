from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/performance/benchmark_results.csv")
    parser.add_argument("--output-dir", default="reports/performance")
    return parser.parse_args()


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó benchmark CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("A benchmark CSV üres.")
    return df


def grouped_mean(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        df.groupby(["total_events", "batch_size"], as_index=False)[metric]
        .mean()
        .sort_values(["total_events", "batch_size"])
    )


def save_metric_by_batch(
    df: pd.DataFrame,
    output_path: Path,
    *,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    plot_df = grouped_mean(df, metric)
    fig, ax = plt.subplots(figsize=(8, 5))
    for total_events, group in plot_df.groupby("total_events"):
        ax.plot(
            group["batch_size"],
            group[metric],
            marker="o",
            label=f"{int(total_events)} esemény",
        )
    ax.set_title(title)
    ax.set_xlabel("Batch size")
    ax.set_ylabel(ylabel)
    ax.legend(title="Eseményszám")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_scoring_time_distribution(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = grouped_mean(df, "avg_latency_ms")
    labels = [f"{int(row.total_events)}/{int(row.batch_size)}" for row in plot_df.itertuples()]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, plot_df["avg_latency_ms"])
    ax.set_title("Átlagos scoring késleltetés eloszlása")
    ax.set_xlabel("Eseményszám / batch size")
    ax.set_ylabel("Átlagos késleltetés (ms)")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_failed_events(df: pd.DataFrame, output_path: Path) -> None:
    failed = grouped_mean(df, "failed_events")
    if failed["failed_events"].sum() <= 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for total_events, group in failed.groupby("total_events"):
        ax.plot(
            group["batch_size"],
            group["failed_events"],
            marker="o",
            label=f"{int(total_events)} esemény",
        )
    ax.set_title("Hibás események batch size szerint")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Hibás események")
    ax.legend(title="Eseményszám")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def first_available_metric(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for metric in candidates:
        if metric in df.columns and pd.to_numeric(df[metric], errors="coerce").notna().any():
            return metric
    return None


def save_resource_usage(df: pd.DataFrame, output_path: Path) -> bool:
    cpu_metric = first_available_metric(df, ["cpu_time_per_event_ms", "process_cpu_time_s"])
    memory_metric = first_available_metric(df, ["memory_rss_delta_mb", "peak_memory_mb"])
    if cpu_metric is None and memory_metric is None:
        print("[WARN] Nincs elérhető CPU- vagy memóriaadat az erőforrás-ábrához.")
        return False

    fig, ax_cpu = plt.subplots(figsize=(8, 5))
    if cpu_metric is not None:
        cpu_df = grouped_mean(df, cpu_metric)
        for total_events, group in cpu_df.groupby("total_events"):
            ax_cpu.plot(
                group["batch_size"],
                group[cpu_metric],
                marker="o",
                label=f"CPU, {int(total_events)} esemény",
            )
        ylabel = "CPU-idő eseményenként (ms)" if cpu_metric == "cpu_time_per_event_ms" else "CPU-idő (s)"
        ax_cpu.set_ylabel(ylabel)

    ax_memory = None
    if memory_metric is not None:
        memory_df = grouped_mean(df, memory_metric)
        ax_memory = ax_cpu.twinx()
        for total_events, group in memory_df.groupby("total_events"):
            ax_memory.plot(
                group["batch_size"],
                group[memory_metric],
                marker="s",
                linestyle="--",
                label=f"Memória, {int(total_events)} esemény",
            )
        ylabel = "Memória RSS delta (MB)" if memory_metric == "memory_rss_delta_mb" else "Csúcsmemória (MB)"
        ax_memory.set_ylabel(ylabel)

    ax_cpu.set_title("CPU- és memóriahasználat batch size szerint")
    ax_cpu.set_xlabel("Batch size")
    ax_cpu.grid(True, alpha=0.25)
    handles, labels = ax_cpu.get_legend_handles_labels()
    if ax_memory is not None:
        mem_handles, mem_labels = ax_memory.get_legend_handles_labels()
        handles.extend(mem_handles)
        labels.extend(mem_labels)
    if handles:
        ax_cpu.legend(handles, labels, title="Mért érték", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_performance_results(input_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(input_path)
    outputs = [
        output_dir / "latency_by_batch_size.png",
        output_dir / "throughput_by_batch_size.png",
        output_dir / "scoring_time_distribution.png",
    ]

    save_metric_by_batch(
        df,
        outputs[0],
        metric="p95_latency_ms",
        title="P95 késleltetés batch size szerint",
        ylabel="P95 késleltetés (ms)",
    )
    save_metric_by_batch(
        df,
        outputs[1],
        metric="events_per_second",
        title="Áteresztőképesség batch size szerint",
        ylabel="Esemény/másodperc",
    )
    save_scoring_time_distribution(df, outputs[2])

    failed_output = output_dir / "failed_events_by_batch_size.png"
    save_failed_events(df, failed_output)
    if failed_output.exists():
        outputs.append(failed_output)
    resource_output = output_dir / "resource_usage_by_batch_size.png"
    if save_resource_usage(df, resource_output):
        outputs.append(resource_output)
    return outputs


def main() -> None:
    args = parse_args()
    outputs = plot_performance_results(Path(args.input), Path(args.output_dir))
    for path in outputs:
        print(f"[OK] Ábra: {path}")


if __name__ == "__main__":
    main()
