from __future__ import annotations

import pandas as pd

from ml.src.real_measurement.generate_real_measurement_report import generate_report


def write_metrics(path, f1=0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"precision": f1, "recall": f1, "f1": f1, "n_samples": 4}]).to_csv(path, index=False)


def write_hybrid(path, f1=0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"strategy": "hybrid_or", "precision": f1, "recall": f1, "f1": f1, "n_samples": 4}]).to_csv(
        path,
        index=False,
    )


def write_comparison(path, hybrid_f1):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"configuration": "Wazuh-only", "precision": 0.5, "recall": 0.5, "f1": 0.5, "false_positive_rate": 0.2, "alert_count": 2},
            {"configuration": "AE-Minimal lab", "precision": 0.4, "recall": 0.4, "f1": 0.4, "false_positive_rate": 0.3, "alert_count": 3},
            {"configuration": "Hybrid OR", "precision": hybrid_f1, "recall": hybrid_f1, "f1": hybrid_f1, "false_positive_rate": 0.1, "alert_count": 2},
        ]
    ).to_csv(path, index=False)


def test_report_does_not_claim_improvement_without_metric_improvement(tmp_path):
    comparison = tmp_path / "results/real_comparison/metrics_comparison.csv"
    write_comparison(comparison, hybrid_f1=0.4)
    wazuh = tmp_path / "results/wazuh_real/metrics_summary.csv"
    ae = tmp_path / "results/ae_lab/metrics_summary.csv"
    hybrid = tmp_path / "results/hybrid_real/metrics_summary.csv"
    write_metrics(wazuh)
    write_metrics(ae)
    write_hybrid(hybrid)

    outputs = generate_report(
        comparison_path=comparison,
        wazuh_metrics_path=wazuh,
        ae_metrics_path=ae,
        hybrid_metrics_path=hybrid,
        input_validation_path=tmp_path / "missing.md",
        wazuh_summary_path=tmp_path / "missing2.md",
        output_dir=tmp_path / "reports/real_measurement",
    )

    text = outputs["thesis"].read_text(encoding="utf-8")
    assert "nem igazolható egyértelmű javulás" in text
    assert "## Korlátok" in text


def test_report_uses_cautious_improvement_sentence(tmp_path):
    comparison = tmp_path / "results/real_comparison/metrics_comparison.csv"
    write_comparison(comparison, hybrid_f1=0.7)
    wazuh = tmp_path / "results/wazuh_real/metrics_summary.csv"
    ae = tmp_path / "results/ae_lab/metrics_summary.csv"
    hybrid = tmp_path / "results/hybrid_real/metrics_summary.csv"
    write_metrics(wazuh)
    write_metrics(ae)
    write_hybrid(hybrid, f1=0.7)

    outputs = generate_report(
        comparison_path=comparison,
        wazuh_metrics_path=wazuh,
        ae_metrics_path=ae,
        hybrid_metrics_path=hybrid,
        input_validation_path=tmp_path / "missing.md",
        wazuh_summary_path=tmp_path / "missing2.md",
        output_dir=tmp_path / "reports/real_measurement",
    )

    text = outputs["thesis"].read_text(encoding="utf-8")
    assert "a vizsgált lab mérésben javulás figyelhető meg" in text
