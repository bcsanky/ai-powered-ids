from __future__ import annotations

import pandas as pd
import pytest

from ml.src.hybrid_eval import build_hybrid_predictions, run_hybrid_eval


def test_hybrid_prediction_uses_or_logic():
    ae_df = pd.DataFrame(
        {
            "y_true": [0, 1, 1, 0],
            "score": [0.1, 0.2, 0.3, 0.4],
            "pred_f1_optimum": [0, 1, 0, 0],
        }
    )
    rule_df = pd.DataFrame(
        {
            "y_true": [0, 1, 1, 0],
            "score": [1.0, 2.0, 3.0, 4.0],
            "y_pred": [0, 0, 1, 0],
        }
    )

    out = build_hybrid_predictions(ae_df, rule_df)

    assert out["hybrid_pred"].tolist() == [0, 1, 1, 0]
    assert out["ae_pred"].tolist() == [0, 1, 0, 0]
    assert out["rule_pred"].tolist() == [0, 0, 1, 0]
    assert out["hybrid_score"].between(0.0, 1.0).all()


def test_hybrid_prediction_rejects_different_lengths():
    ae_df = pd.DataFrame({"y_true": [0, 1], "score": [0.1, 0.2], "pred_f1_optimum": [0, 1]})
    rule_df = pd.DataFrame({"y_true": [0], "score": [1.0], "y_pred": [0]})

    with pytest.raises(ValueError, match="hossza eltér"):
        build_hybrid_predictions(ae_df, rule_df)


def test_hybrid_prediction_rejects_different_labels():
    ae_df = pd.DataFrame({"y_true": [0, 1], "score": [0.1, 0.2], "pred_f1_optimum": [0, 1]})
    rule_df = pd.DataFrame({"y_true": [0, 0], "score": [1.0, 2.0], "y_pred": [0, 1]})

    with pytest.raises(ValueError, match="y_true"):
        build_hybrid_predictions(ae_df, rule_df)


def test_hybrid_eval_writes_metrics_summary(tmp_path):
    ae_run_dir = tmp_path / "ae_v1_test"
    rule_run_dir = tmp_path / "baseline_wazuh_test"
    output_root = tmp_path / "hybrid"
    ae_run_dir.mkdir()
    rule_run_dir.mkdir()

    pd.DataFrame(
        {
            "y_true": [0, 0, 1, 1],
            "score": [0.1, 0.4, 0.6, 0.9],
            "pred_f1_optimum": [0, 1, 0, 1],
        }
    ).to_csv(ae_run_dir / "predictions.csv", index=False)
    pd.DataFrame(
        {
            "y_true": [0, 0, 1, 1],
            "score": [0.2, 0.5, 0.7, 0.8],
            "y_pred": [0, 0, 1, 0],
        }
    ).to_csv(rule_run_dir / "predictions.csv", index=False)

    run_dir = run_hybrid_eval(ae_run_dir, rule_run_dir, output_root)

    metrics = pd.read_csv(run_dir / "metrics_summary.csv")
    required = {
        "threshold_name",
        "tn",
        "fp",
        "fn",
        "tp",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
        "true_positive_rate",
        "true_negative_rate",
        "alert_count",
        "roc_auc",
        "n_samples",
        "n_attack",
        "n_benign",
    }
    assert required.issubset(metrics.columns)
    assert metrics.loc[0, "threshold_name"] == "hybrid_union"
    assert (run_dir / "predictions.csv").exists()
    assert (run_dir / "confusion_matrix.png").exists()
