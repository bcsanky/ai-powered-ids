from __future__ import annotations

from ml.src.export_performance_outputs import export_performance_outputs


def test_export_performance_outputs_creates_compatible_files(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    benchmark = source_dir / "benchmark_results.csv"
    latency = source_dir / "latency_by_batch_size.png"
    throughput = source_dir / "throughput_by_batch_size.png"
    resource = source_dir / "resource_usage_by_batch_size.png"
    benchmark.write_text("total_events,batch_size\n1,1\n", encoding="utf-8")
    latency.write_bytes(b"latency")
    throughput.write_bytes(b"throughput")
    resource.write_bytes(b"resource")

    outputs = export_performance_outputs(
        benchmark_path=benchmark,
        latency_figure=latency,
        throughput_figure=throughput,
        resource_figure=resource,
        results_dir=tmp_path / "results" / "performance",
        figures_dir=tmp_path / "figures" / "final",
    )

    assert tmp_path.joinpath("results/performance/performance_metrics.csv").exists()
    assert tmp_path.joinpath("figures/final/latency_by_load.png").exists()
    assert tmp_path.joinpath("figures/final/throughput.png").exists()
    assert tmp_path.joinpath("figures/final/resource_usage_by_batch_size.png").exists()
    assert len(outputs) == 4


def test_export_performance_outputs_allows_missing_optional_resource_figure(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    benchmark = source_dir / "benchmark_results.csv"
    latency = source_dir / "latency_by_batch_size.png"
    throughput = source_dir / "throughput_by_batch_size.png"
    benchmark.write_text("total_events,batch_size\n1,1\n", encoding="utf-8")
    latency.write_bytes(b"latency")
    throughput.write_bytes(b"throughput")

    outputs = export_performance_outputs(
        benchmark_path=benchmark,
        latency_figure=latency,
        throughput_figure=throughput,
        resource_figure=source_dir / "missing_resource.png",
        results_dir=tmp_path / "results" / "performance",
        figures_dir=tmp_path / "figures" / "final",
    )

    assert tmp_path.joinpath("results/performance/performance_metrics.csv").exists()
    assert tmp_path.joinpath("figures/final/latency_by_load.png").exists()
    assert tmp_path.joinpath("figures/final/throughput.png").exists()
    assert not tmp_path.joinpath("figures/final/resource_usage_by_batch_size.png").exists()
    assert len(outputs) == 3
