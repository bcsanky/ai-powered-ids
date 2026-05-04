from __future__ import annotations

import json

import pandas as pd

from ml.src.generate_performance_report import write_performance_report


def write_benchmark(path):
    pd.DataFrame(
        [
            {
                "model_load_time_s": 0.1,
                "total_events": 10,
                "batch_size": 1,
                "repeat_index": 1,
                "total_time_s": 0.5,
                "events_per_second": 20.0,
                "avg_latency_ms": 2.0,
                "p50_latency_ms": 1.8,
                "p95_latency_ms": 3.0,
                "p99_latency_ms": 4.0,
                "max_latency_ms": 5.0,
                "failed_events": 0,
            }
        ]
    ).to_csv(path, index=False)


def test_performance_report_is_written(tmp_path):
    benchmark = tmp_path / "benchmark_results.csv"
    system_info = tmp_path / "system_info.json"
    write_benchmark(benchmark)
    system_info.write_text(
        json.dumps(
            {
                "input_file": "examples/lab/lab_events.jsonl",
                "model_root": "model",
                "preprocess_file": "preprocess.pkl",
                "note": "lokális/labor mérés",
            }
        ),
        encoding="utf-8",
    )

    outputs = write_performance_report(benchmark, system_info, tmp_path / "out")

    assert outputs["markdown"].exists()
    assert outputs["html"].exists()
    text = outputs["markdown"].read_text(encoding="utf-8")
    assert "lokális/labor" in text
    assert "Legjobb áteresztőképesség" in text


def test_performance_report_handles_missing_system_info(tmp_path):
    benchmark = tmp_path / "benchmark_results.csv"
    write_benchmark(benchmark)

    outputs = write_performance_report(benchmark, tmp_path / "missing.json", tmp_path / "out")

    text = outputs["markdown"].read_text(encoding="utf-8")
    assert "system_info.json nem található" in text
    assert "lokális/labor mérés" in text
