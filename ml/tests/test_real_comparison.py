from __future__ import annotations

import pandas as pd
import pytest

from ml.src.hybrid_real.compare_real_results import all_positive_baseline_row, collect_real_comparison, save_real_comparison
from ml.src.hybrid_real.plot_real_comparison import plot_real_comparison


def write_single_metrics(path, precision=0.5):
    pd.DataFrame(
        [
            {
                "precision": precision,
                "recall": 0.6,
                "f1": 0.55,
                "false_positive_rate": 0.1,
                "false_negative_rate": 0.4,
                "alert_count": 3,
                "mean_ttd": 5.0,
                "n_samples": 10,
                "n_attack": 4,
                "n_benign": 6,
                "TP": 2,
                "FP": 1,
                "TN": 5,
                "FN": 2,
            }
        ]
    ).to_csv(path, index=False)


def write_hybrid_metrics(path):
    pd.DataFrame(
        [
            {
                "strategy": "hybrid_or",
                "precision": 0.4,
                "recall": 0.9,
                "f1": 0.55,
                "false_positive_rate": 0.3,
                "false_negative_rate": 0.1,
                "alert_count": 6,
                "mean_ttd": 4.0,
                "n_samples": 10,
                "n_attack": 4,
                "n_benign": 6,
                "TP": 4,
                "FP": 6,
                "TN": 0,
                "FN": 0,
            },
            {
                "strategy": "hybrid_weighted",
                "precision": 0.7,
                "recall": 0.5,
                "f1": 0.58,
                "false_positive_rate": 0.1,
                "false_negative_rate": 0.5,
                "alert_count": 2,
                "mean_ttd": 4.0,
                "n_samples": 10,
                "n_attack": 4,
                "n_benign": 6,
                "TP": 2,
                "FP": 0,
                "TN": 6,
                "FN": 2,
            },
            {
                "strategy": "hybrid_priority",
                "precision": 0.4,
                "recall": 0.9,
                "f1": 0.55,
                "false_positive_rate": 0.3,
                "false_negative_rate": 0.1,
                "alert_count": 6,
                "mean_ttd": 4.0,
                "n_samples": 10,
                "n_attack": 4,
                "n_benign": 6,
                "TP": 4,
                "FP": 6,
                "TN": 0,
                "FN": 0,
            },
        ]
    ).to_csv(path, index=False)


def write_ground_truth(path, *, benign=6, attack=4):
    rows = [{"event_id": f"b{i}", "label": "benign"} for i in range(benign)]
    rows.extend({"event_id": f"a{i}", "label": "attack"} for i in range(attack))
    pd.DataFrame(rows).to_csv(path, index=False)


def test_real_comparison_contains_all_configurations(tmp_path):
    wazuh_path = tmp_path / "wazuh_metrics.csv"
    ae_path = tmp_path / "ae_metrics.csv"
    hybrid_path = tmp_path / "hybrid_metrics.csv"
    ground_truth_path = tmp_path / "lab_ground_truth.csv"
    write_single_metrics(wazuh_path, precision=0.3)
    write_single_metrics(ae_path, precision=0.8)
    write_hybrid_metrics(hybrid_path)
    write_ground_truth(ground_truth_path)

    comparison = collect_real_comparison(
        wazuh_metrics_path=wazuh_path,
        ae_metrics_path=ae_path,
        hybrid_metrics_path=hybrid_path,
        ground_truth_path=ground_truth_path,
    )

    assert comparison["configuration"].tolist() == [
        "Wazuh-only",
        "AE-Minimal lab",
        "All-positive baseline",
        "Hybrid OR",
        "Hybrid weighted",
        "Hybrid priority",
    ]


def test_all_positive_baseline_is_computed_from_ground_truth(tmp_path):
    ground_truth_path = tmp_path / "lab_ground_truth.csv"
    write_ground_truth(ground_truth_path, benign=60, attack=40)

    row = all_positive_baseline_row(ground_truth_path)

    assert row["configuration"] == "All-positive baseline"
    assert row["TP"] == 40
    assert row["FP"] == 60
    assert row["TN"] == 0
    assert row["FN"] == 0
    assert row["precision"] == pytest.approx(0.4)
    assert row["recall"] == pytest.approx(1.0)
    assert row["f1"] == pytest.approx(0.571429, rel=1e-6)
    assert row["false_positive_rate"] == pytest.approx(1.0)
    assert row["false_negative_rate"] == pytest.approx(0.0)
    assert row["alert_count"] == 100
    assert pd.isna(row["mean_ttd"])
    assert row["n_samples"] == 100


def test_save_real_comparison_writes_csv_markdown_and_metadata(tmp_path):
    wazuh_path = tmp_path / "wazuh_metrics.csv"
    ae_path = tmp_path / "ae_metrics.csv"
    hybrid_path = tmp_path / "hybrid_metrics.csv"
    ground_truth_path = tmp_path / "lab_ground_truth.csv"
    output_dir = tmp_path / "real_comparison"
    write_single_metrics(wazuh_path, precision=0.3)
    write_single_metrics(ae_path, precision=0.8)
    write_hybrid_metrics(hybrid_path)
    write_ground_truth(ground_truth_path)

    outputs = save_real_comparison(
        wazuh_metrics_path=wazuh_path,
        ae_metrics_path=ae_path,
        hybrid_metrics_path=hybrid_path,
        ground_truth_path=ground_truth_path,
        output_dir=output_dir,
    )

    assert outputs["csv"].exists()
    assert outputs["markdown"].exists()
    assert outputs["metadata"].exists()
    assert "naiv kontrollsor" in outputs["markdown"].read_text(encoding="utf-8")


def test_plot_real_comparison_writes_figures(tmp_path):
    comparison_path = tmp_path / "metrics_comparison.csv"
    output_dir = tmp_path / "real_comparison"
    pd.DataFrame(
        [
            {
                "configuration": "Wazuh-only",
                "precision": 0.5,
                "recall": 0.6,
                "f1": 0.55,
                "false_positive_rate": 0.2,
                "false_negative_rate": 0.4,
                "alert_count": 3,
                "mean_ttd": 5.0,
                "n_samples": 10,
            },
            {
                "configuration": "AE-Minimal lab",
                "precision": 0.8,
                "recall": 0.7,
                "f1": 0.75,
                "false_positive_rate": 0.1,
                "false_negative_rate": 0.3,
                "alert_count": 4,
                "mean_ttd": "",
                "n_samples": 10,
            },
        ]
    ).to_csv(comparison_path, index=False)

    paths = plot_real_comparison(comparison_path, output_dir)

    assert (output_dir / "fig_precision_recall_f1.png").exists()
    assert (output_dir / "fig_false_positive_rate.png").exists()
    assert (output_dir / "fig_alert_count.png").exists()
    assert (output_dir / "fig_mean_ttd.png").exists()
    assert paths[0].name == "fig_precision_recall_f1.png"
