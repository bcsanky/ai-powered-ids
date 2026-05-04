from __future__ import annotations

import pandas as pd

from ml.src.hybrid_real.compare_real_results import collect_real_comparison, save_real_comparison
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
            },
        ]
    ).to_csv(path, index=False)


def test_real_comparison_contains_all_configurations(tmp_path):
    wazuh_path = tmp_path / "wazuh_metrics.csv"
    ae_path = tmp_path / "ae_metrics.csv"
    hybrid_path = tmp_path / "hybrid_metrics.csv"
    write_single_metrics(wazuh_path, precision=0.3)
    write_single_metrics(ae_path, precision=0.8)
    write_hybrid_metrics(hybrid_path)

    comparison = collect_real_comparison(
        wazuh_metrics_path=wazuh_path,
        ae_metrics_path=ae_path,
        hybrid_metrics_path=hybrid_path,
    )

    assert comparison["configuration"].tolist() == [
        "Wazuh-only",
        "AE-Minimal lab",
        "Hybrid OR",
        "Hybrid weighted",
        "Hybrid priority",
    ]


def test_save_real_comparison_writes_csv_markdown_and_metadata(tmp_path):
    wazuh_path = tmp_path / "wazuh_metrics.csv"
    ae_path = tmp_path / "ae_metrics.csv"
    hybrid_path = tmp_path / "hybrid_metrics.csv"
    output_dir = tmp_path / "real_comparison"
    write_single_metrics(wazuh_path, precision=0.3)
    write_single_metrics(ae_path, precision=0.8)
    write_hybrid_metrics(hybrid_path)

    outputs = save_real_comparison(
        wazuh_metrics_path=wazuh_path,
        ae_metrics_path=ae_path,
        hybrid_metrics_path=hybrid_path,
        output_dir=output_dir,
    )

    assert outputs["csv"].exists()
    assert outputs["markdown"].exists()
    assert outputs["metadata"].exists()


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
