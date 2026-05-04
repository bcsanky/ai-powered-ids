from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ml.src.real_measurement_qa.generate_thesis_tables import generate_tables


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_thesis_table_generator_replaces_nan_with_no_data(tmp_path):
    comparison = tmp_path / "results/real_comparison/metrics_comparison.csv"
    wazuh = tmp_path / "results/wazuh_real/metrics_summary.csv"
    ae = tmp_path / "results/ae_lab/metrics_summary.csv"
    hybrid = tmp_path / "results/hybrid_real/metrics_summary.csv"
    answer = tmp_path / "reports/real_measurement_qa/research_question_answer.json"
    common = {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "false_positive_rate": 0.2,
        "false_negative_rate": 0.5,
        "alert_count": 2,
        "mean_ttd": np.nan,
        "n_samples": 4,
        "n_attack": 2,
        "n_benign": 2,
    }
    write_csv(
        comparison,
        [
            {"configuration": "Wazuh-only", **common},
            {"configuration": "AE-Minimal lab", **common},
            {"configuration": "Hybrid OR", **common, "f1": 0.6},
        ],
    )
    write_csv(wazuh, [common])
    write_csv(ae, [common])
    write_csv(hybrid, [{**common, "strategy": "hybrid_or"}])
    answer.parent.mkdir(parents=True, exist_ok=True)
    answer.write_text(
        json.dumps(
            {
                "best_hybrid_by_f1": "Hybrid OR",
                "f1_delta_vs_wazuh": 0.1,
                "fpr_delta_vs_wazuh": None,
                "alert_count_delta_vs_wazuh": 0,
            }
        ),
        encoding="utf-8",
    )

    outputs = generate_tables(
        comparison_path=comparison,
        wazuh_metrics_path=wazuh,
        ae_metrics_path=ae,
        hybrid_metrics_path=hybrid,
        research_answer_path=answer,
        output_dir=tmp_path / "reports/real_measurement_qa",
    )

    text = outputs["real_comparison"].read_text(encoding="utf-8")
    assert "nincs adat" in text
    assert "0.5000" in text
