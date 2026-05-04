from __future__ import annotations

import pytest

from ml.src.benchmark_scoring import run_benchmark


class FakeScorer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def load(self):
        return None

    def score_event(self, **kwargs):
        return {"ok": True}


def write_lab_input(path):
    path.write_text(
        '{"event_id":"e1","destination_port":80,"flow_duration":1,'
        '"total_fwd_packets":1,"total_backward_packets":1,'
        '"flow_bytes_per_sec":1.0,"flow_packets_per_sec":1.0,'
        '"protocol":"6","rule_flag":false,"rule_level":0}\n',
        encoding="utf-8",
    )


def test_benchmark_creates_results_csv(tmp_path):
    input_path = tmp_path / "events.jsonl"
    output_dir = tmp_path / "out"
    write_lab_input(input_path)

    results = run_benchmark(
        input_path=input_path,
        output_dir=output_dir,
        model_root=tmp_path / "model",
        preprocess_path=tmp_path / "preprocess.pkl",
        event_counts=[2],
        batch_sizes=[1],
        repeats=1,
        scorer_factory=FakeScorer,
    )

    csv_path = output_dir / "benchmark_results.csv"
    assert csv_path.exists()
    required = {
        "total_events",
        "batch_size",
        "total_time_s",
        "events_per_second",
        "avg_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "failed_events",
    }
    assert required.issubset(results.columns)
    assert (output_dir / "benchmark_summary.md").exists()
    assert (output_dir / "system_info.json").exists()


def test_benchmark_rejects_zero_event_count(tmp_path):
    input_path = tmp_path / "events.jsonl"
    write_lab_input(input_path)

    with pytest.raises(ValueError, match="event-count"):
        run_benchmark(
            input_path=input_path,
            output_dir=tmp_path / "out",
            model_root=tmp_path / "model",
            preprocess_path=tmp_path / "preprocess.pkl",
            event_counts=[0],
            batch_sizes=[1],
            repeats=1,
            scorer_factory=FakeScorer,
        )


def test_benchmark_rejects_empty_input(tmp_path):
    input_path = tmp_path / "empty.jsonl"
    input_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="nem tartalmaz eseményt"):
        run_benchmark(
            input_path=input_path,
            output_dir=tmp_path / "out",
            model_root=tmp_path / "model",
            preprocess_path=tmp_path / "preprocess.pkl",
            event_counts=[1],
            batch_sizes=[1],
            repeats=1,
            scorer_factory=FakeScorer,
        )


def test_benchmark_does_not_fallback_when_model_is_missing(tmp_path):
    input_path = tmp_path / "events.jsonl"
    write_lab_input(input_path)

    with pytest.raises(FileNotFoundError):
        run_benchmark(
            input_path=input_path,
            output_dir=tmp_path / "out",
            model_root=tmp_path / "missing_model_root",
            preprocess_path=tmp_path / "missing_preprocess.pkl",
            event_counts=[1],
            batch_sizes=[1],
            repeats=1,
        )
