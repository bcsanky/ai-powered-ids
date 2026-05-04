from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="reports/performance/benchmark_results.csv")
    parser.add_argument("--latency-figure", default="reports/performance/latency_by_batch_size.png")
    parser.add_argument("--throughput-figure", default="reports/performance/throughput_by_batch_size.png")
    parser.add_argument("--resource-figure", default="reports/performance/resource_usage_by_batch_size.png")
    parser.add_argument("--results-dir", default="results/performance")
    parser.add_argument("--figures-dir", default="figures/final")
    return parser.parse_args()


def copy_required(source: Path, target: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Hiányzó kimeneti állomány: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def copy_optional(source: Path, target: Path) -> Path | None:
    if not source.exists():
        print(f"[WARN] Opcionális ábra nem található: {source}")
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def export_performance_outputs(
    *,
    benchmark_path: Path,
    latency_figure: Path,
    throughput_figure: Path,
    resource_figure: Path,
    results_dir: Path,
    figures_dir: Path,
) -> list[Path]:
    outputs = [
        copy_required(benchmark_path, results_dir / "performance_metrics.csv"),
        copy_required(latency_figure, figures_dir / "latency_by_load.png"),
        copy_required(throughput_figure, figures_dir / "throughput.png"),
    ]
    optional_resource = copy_optional(resource_figure, figures_dir / "resource_usage_by_batch_size.png")
    if optional_resource is not None:
        outputs.append(optional_resource)
    return outputs


def main() -> None:
    args = parse_args()
    outputs = export_performance_outputs(
        benchmark_path=Path(args.benchmark),
        latency_figure=Path(args.latency_figure),
        throughput_figure=Path(args.throughput_figure),
        resource_figure=Path(args.resource_figure),
        results_dir=Path(args.results_dir),
        figures_dir=Path(args.figures_dir),
    )
    for path in outputs:
        print(f"[OK] Exportált kimenet: {path}")


if __name__ == "__main__":
    main()
