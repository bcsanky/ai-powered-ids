from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd

from ml.src.score_events import read_events, split_event
from ml.src.scoring_runtime import AEScorer

try:
    import resource as _resource
except ImportError:  # pragma: no cover - platform dependent
    _resource = None

try:
    import psutil as _psutil
except ImportError:  # pragma: no cover - optional dependency
    _psutil = None


RESULT_COLUMNS = [
    "model_load_time_s",
    "total_events",
    "batch_size",
    "repeat_index",
    "total_time_s",
    "events_per_second",
    "avg_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "max_latency_ms",
    "failed_events",
    "process_cpu_time_s",
    "cpu_time_per_event_ms",
    "memory_rss_mb_before",
    "memory_rss_mb_after",
    "memory_rss_delta_mb",
    "peak_memory_mb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="examples/lab/lab_events.jsonl")
    parser.add_argument("--output-dir", default="reports/performance")
    parser.add_argument("--model-root", default="artifacts/final/final-ae-minimal-v1")
    parser.add_argument("--preprocess", default="data/processed/final/ae_minimal/preprocess.pkl")
    parser.add_argument("--event-counts", nargs="+", type=int, default=[100, 500, 1000, 5000])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 10, 50, 100])
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def expand_events(events: list[dict[str, Any]], total_events: int) -> list[dict[str, Any]]:
    if not events:
        raise ValueError("A benchmark bemenet nem tartalmaz eseményt.")
    if total_events <= 0:
        raise ValueError("A total_events értékének pozitívnak kell lennie.")

    expanded = []
    for idx in range(total_events):
        source = dict(events[idx % len(events)])
        source["event_id"] = f"{source.get('event_id', 'event')}-bench-{idx:06d}"
        expanded.append(source)
    return expanded


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def current_rss_mb() -> float | None:
    if _psutil is None:
        return None
    try:
        return float(_psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except Exception:
        return None


def peak_memory_mb() -> float | None:
    if _resource is None:
        return None
    try:
        usage = _resource.getrusage(_resource.RUSAGE_SELF)
        value = float(usage.ru_maxrss)
    except Exception:
        return None
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


def resource_measurement_info() -> dict[str, str]:
    return {
        "cpu_time_method": "time.process_time",
        "memory_rss_method": "psutil.Process.memory_info.rss" if _psutil is not None else "not_available",
        "peak_memory_method": "resource.getrusage.ru_maxrss" if _resource is not None else "not_available",
    }


def score_records(
    scorer: Any,
    records: list[dict[str, Any]],
    batch_size: int,
) -> tuple[float, list[float], int, dict[str, float | None]]:
    if batch_size <= 0:
        raise ValueError("A batch_size értékének pozitívnak kell lennie.")

    latencies_ms = []
    failed_events = 0
    cpu_before = time.process_time()
    memory_before = current_rss_mb()
    start_total = time.perf_counter()

    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start: batch_start + batch_size]
        for record in batch:
            start_event = time.perf_counter()
            try:
                event_id, features, rule_flag, rule_level, _ = split_event(record)
                scorer.score_event(
                    event_id=event_id,
                    features=features,
                    rule_flag=rule_flag,
                    rule_level=rule_level,
                )
            except Exception:
                failed_events += 1
            finally:
                latencies_ms.append((time.perf_counter() - start_event) * 1000.0)

    total_time_s = time.perf_counter() - start_total
    cpu_after = time.process_time()
    memory_after = current_rss_mb()
    process_cpu_time_s = max(cpu_after - cpu_before, 0.0)
    memory_delta = None
    if memory_before is not None and memory_after is not None:
        memory_delta = memory_after - memory_before
    resources = {
        "process_cpu_time_s": process_cpu_time_s,
        "cpu_time_per_event_ms": (process_cpu_time_s / len(records) * 1000.0) if records else 0.0,
        "memory_rss_mb_before": memory_before,
        "memory_rss_mb_after": memory_after,
        "memory_rss_delta_mb": memory_delta,
        "peak_memory_mb": peak_memory_mb(),
    }
    return total_time_s, latencies_ms, failed_events, resources


def summarize_run(
    *,
    model_load_time_s: float,
    total_events: int,
    batch_size: int,
    repeat_index: int,
    total_time_s: float,
    latencies_ms: list[float],
    failed_events: int,
    resources: dict[str, float | None],
) -> dict[str, Any]:
    events_per_second = float(total_events / total_time_s) if total_time_s > 0 else 0.0
    return {
        "model_load_time_s": model_load_time_s,
        "total_events": total_events,
        "batch_size": batch_size,
        "repeat_index": repeat_index,
        "total_time_s": total_time_s,
        "events_per_second": events_per_second,
        "avg_latency_ms": float(mean(latencies_ms)) if latencies_ms else 0.0,
        "p50_latency_ms": percentile(latencies_ms, 50),
        "p95_latency_ms": percentile(latencies_ms, 95),
        "p99_latency_ms": percentile(latencies_ms, 99),
        "max_latency_ms": float(max(latencies_ms)) if latencies_ms else 0.0,
        "failed_events": int(failed_events),
        "process_cpu_time_s": resources.get("process_cpu_time_s"),
        "cpu_time_per_event_ms": resources.get("cpu_time_per_event_ms"),
        "memory_rss_mb_before": resources.get("memory_rss_mb_before"),
        "memory_rss_mb_after": resources.get("memory_rss_mb_after"),
        "memory_rss_delta_mb": resources.get("memory_rss_delta_mb"),
        "peak_memory_mb": resources.get("peak_memory_mb"),
    }


def write_system_info(
    *,
    output_dir: Path,
    input_file: Path,
    model_root: Path,
    preprocess_file: Path,
) -> Path:
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "input_file": str(input_file),
        "model_root": str(model_root),
        "preprocess_file": str(preprocess_file),
        "note": "lokális/labor mérés",
    }
    info.update(resource_measurement_info())
    path = output_dir / "system_info.json"
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_benchmark_summary(results: pd.DataFrame, output_dir: Path) -> Path:
    best_throughput = results.sort_values("events_per_second", ascending=False).iloc[0]
    best_latency = results.sort_values("p95_latency_ms", ascending=True).iloc[0]
    failed_total = int(results["failed_events"].sum())
    cpu_series = pd.to_numeric(results.get("cpu_time_per_event_ms"), errors="coerce")
    memory_delta = pd.to_numeric(results.get("memory_rss_delta_mb"), errors="coerce")
    peak_memory = pd.to_numeric(results.get("peak_memory_mb"), errors="coerce")

    lines = [
        "# Batch scoring teljesítménymérési összefoglaló",
        "",
        "A mérés a meglévő AE-Minimal scoring lánc lokális/labor teljesítményét vizsgálja. Az eseményszám növelése determinisztikus ismétléssel történt, kizárólag a feldolgozási idő méréséhez.",
        "",
        "## Fő eredmények",
        "",
        f"- Legjobb áteresztőképesség: {best_throughput['events_per_second']:.2f} esemény/másodperc, batch size {int(best_throughput['batch_size'])}, eseményszám {int(best_throughput['total_events'])}.",
        f"- Legalacsonyabb p95 késleltetés: {best_latency['p95_latency_ms']:.4f} ms, batch size {int(best_latency['batch_size'])}, eseményszám {int(best_latency['total_events'])}.",
        f"- Hibás események összesen: {failed_total}.",
    ]
    if cpu_series.notna().any():
        lines.append(f"- Legalacsonyabb CPU-idő eseményenként: {cpu_series.min():.4f} ms.")
    if memory_delta.notna().any():
        lines.append(f"- Legnagyobb mért memória RSS delta: {memory_delta.max():.4f} MB.")
    elif peak_memory.notna().any():
        lines.append(f"- Legnagyobb mért csúcsmemória: {peak_memory.max():.4f} MB.")
    lines.extend(
        [
            "",
            "## Erőforrás-mérés",
            "",
            "A CPU-idő mérése `time.process_time()` alapján történik. A memória RSS érték psutil jelenléte esetén érhető el; Unix/Linux környezetben a csúcsmemória `resource.getrusage()` alapján is rögzíthető. Ha egy memóriaadat nem érhető el az adott platformon, az oszlop üresen maradhat.",
            "",
            "## Korlát",
            "",
            "Ez lokális/labor mérés, nem éles üzemi teljesítménygarancia, nem hosszú idejű SOC-terhelés, és nem natív Wazuh indexelési teljesítménymérés.",
            "",
        ]
    )
    path = output_dir / "benchmark_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_benchmark(
    *,
    input_path: Path,
    output_dir: Path,
    model_root: Path,
    preprocess_path: Path,
    event_counts: list[int],
    batch_sizes: list[int],
    repeats: int,
    scorer_factory: Any = AEScorer,
) -> pd.DataFrame:
    if repeats <= 0:
        raise ValueError("A repeats értékének pozitívnak kell lennie.")
    if any(count <= 0 for count in event_counts):
        raise ValueError("Minden event-count értéknek pozitívnak kell lennie.")
    if any(size <= 0 for size in batch_sizes):
        raise ValueError("Minden batch-size értéknek pozitívnak kell lennie.")

    output_dir.mkdir(parents=True, exist_ok=True)
    base_events = read_events(input_path)
    if not base_events:
        raise ValueError("A benchmark bemenet nem tartalmaz eseményt.")

    scorer = scorer_factory(model_root=model_root, preprocess_path=preprocess_path)
    load_start = time.perf_counter()
    scorer.load()
    model_load_time_s = time.perf_counter() - load_start

    rows = []
    for total_events in event_counts:
        records = expand_events(base_events, total_events)
        for batch_size in batch_sizes:
            for repeat_index in range(1, repeats + 1):
                total_time_s, latencies_ms, failed_events, resources = score_records(
                    scorer,
                    records,
                    batch_size,
                )
                rows.append(
                    summarize_run(
                        model_load_time_s=model_load_time_s,
                        total_events=total_events,
                        batch_size=batch_size,
                        repeat_index=repeat_index,
                        total_time_s=total_time_s,
                        latencies_ms=latencies_ms,
                        failed_events=failed_events,
                        resources=resources,
                    )
                )

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    results.to_csv(output_dir / "benchmark_results.csv", index=False)
    write_benchmark_summary(results, output_dir)
    write_system_info(
        output_dir=output_dir,
        input_file=input_path,
        model_root=model_root,
        preprocess_file=preprocess_path,
    )
    return results


def main() -> None:
    args = parse_args()
    results = run_benchmark(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        model_root=Path(args.model_root),
        preprocess_path=Path(args.preprocess),
        event_counts=args.event_counts,
        batch_sizes=args.batch_sizes,
        repeats=args.repeats,
    )
    print(f"[OK] Benchmark sorok: {len(results)}")
    print(f"[OK] Kimeneti könyvtár: {args.output_dir}")


if __name__ == "__main__":
    main()
