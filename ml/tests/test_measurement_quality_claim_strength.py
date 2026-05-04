from __future__ import annotations

import pandas as pd

from ml.src.measurement_quality.check_research_claim_strength import run_check


def write_thresholds(path):
    path.write_text(
        """
claim_thresholds:
  min_f1_delta_for_improvement: 0.01
  max_fpr_increase_without_warning: 0.10
  max_alert_count_multiplier_without_warning: 2.0
""",
        encoding="utf-8",
    )


def write_comparison(path, hybrid_f1, hybrid_recall=0.7, hybrid_fpr=0.1, hybrid_alert_count=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"configuration": "Wazuh-only", "precision": 0.8, "recall": 0.5, "f1": 0.60, "false_positive_rate": 0.05, "alert_count": 5, "n_samples": 20},
        {"configuration": "AE-Minimal lab", "precision": 0.7, "recall": 0.6, "f1": 0.64, "false_positive_rate": 0.08, "alert_count": 7, "n_samples": 20},
        {"configuration": "Hybrid OR", "precision": 0.8, "recall": hybrid_recall, "f1": hybrid_f1, "false_positive_rate": hybrid_fpr, "alert_count": hybrid_alert_count, "n_samples": 20},
        {"configuration": "Hybrid weighted", "precision": 0.6, "recall": 0.4, "f1": 0.48, "false_positive_rate": 0.02, "alert_count": 4, "n_samples": 20},
        {"configuration": "Hybrid priority", "precision": 0.6, "recall": 0.4, "f1": 0.48, "false_positive_rate": 0.02, "alert_count": 4, "n_samples": 20},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_claim_supported_with_limitations_when_f1_improves(tmp_path):
    comparison = tmp_path / "results/real_comparison/metrics_comparison.csv"
    thresholds = tmp_path / "thresholds.yaml"
    write_thresholds(thresholds)
    write_comparison(comparison, hybrid_f1=0.75, hybrid_recall=0.7, hybrid_fpr=0.07, hybrid_alert_count=8)

    result = run_check(comparison, thresholds, tmp_path / "reports/measurement_quality")

    assert result["claim"]["claim_category"] == "CLAIM_SUPPORTED_WITH_LIMITATIONS"


def test_claim_tradeoff_when_recall_improves_but_fpr_worsens(tmp_path):
    comparison = tmp_path / "results/real_comparison/metrics_comparison.csv"
    thresholds = tmp_path / "thresholds.yaml"
    write_thresholds(thresholds)
    write_comparison(comparison, hybrid_f1=0.61, hybrid_recall=0.9, hybrid_fpr=0.3, hybrid_alert_count=20)

    result = run_check(comparison, thresholds, tmp_path / "reports/measurement_quality")

    assert result["claim"]["claim_category"] == "TRADEOFF_ONLY"


def test_claim_not_supported_when_hybrid_f1_not_better(tmp_path):
    comparison = tmp_path / "results/real_comparison/metrics_comparison.csv"
    thresholds = tmp_path / "thresholds.yaml"
    write_thresholds(thresholds)
    write_comparison(comparison, hybrid_f1=0.55)

    result = run_check(comparison, thresholds, tmp_path / "reports/measurement_quality")

    assert result["claim"]["claim_category"] == "CLAIM_NOT_SUPPORTED"

