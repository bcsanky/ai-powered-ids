from __future__ import annotations

from ml.src.repo_hygiene.check_tracked_generated_outputs import check_tracked_generated_outputs


def run_check(tmp_path, tracked_files):  # noqa: ANN001
    return check_tracked_generated_outputs(
        root=tmp_path,
        output_dir=tmp_path / "reports/repo_hygiene",
        tracked_files=tracked_files,
    )


def test_reports_lab_tracked_output_fails(tmp_path):
    result = run_check(tmp_path, ["reports/lab/lab_scored_events.csv"])

    assert result["status"] == "FAIL"
    assert "reports/lab/lab_scored_events.csv" in result["forbidden"]


def test_readme_and_gitkeep_pass(tmp_path):
    result = run_check(tmp_path, ["reports/README.md", "reports/lab/.gitkeep", "figures/final/.gitkeep"])

    assert result["status"] == "PASS"


def test_reports_final_dashboard_summary_fails(tmp_path):
    result = run_check(tmp_path, ["reports/final/dashboard_summary.csv"])

    assert result["status"] == "FAIL"
    assert "reports/final/dashboard_summary.csv" in result["forbidden"]


def test_figures_final_png_fails(tmp_path):
    result = run_check(tmp_path, ["figures/final/throughput.png"])

    assert result["status"] == "FAIL"
    assert "figures/final/throughput.png" in result["forbidden"]


def test_examples_lab_demo_input_is_allowed(tmp_path):
    result = run_check(tmp_path, ["examples/lab/lab_events.jsonl"])

    assert result["status"] == "PASS"

