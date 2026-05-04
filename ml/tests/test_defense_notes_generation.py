from __future__ import annotations

import json

import pandas as pd

from ml.src.real_measurement_qa.generate_defense_notes import QUESTIONS, generate_notes


def test_defense_notes_created_with_main_questions(tmp_path):
    comparison = tmp_path / "results/real_comparison/metrics_comparison.csv"
    comparison.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"configuration": "Wazuh-only", "f1": 0.5, "precision": 0.5, "recall": 0.5, "false_positive_rate": 0.2, "alert_count": 2},
            {"configuration": "Hybrid OR", "f1": 0.7, "precision": 0.7, "recall": 0.7, "false_positive_rate": 0.2, "alert_count": 2},
        ]
    ).to_csv(comparison, index=False)
    answer = tmp_path / "reports/real_measurement_qa/research_question_answer.json"
    answer.parent.mkdir(parents=True, exist_ok=True)
    answer.write_text(
        json.dumps({"best_hybrid_by_f1": "Hybrid OR", "wazuh_f1": 0.5, "best_hybrid_f1": 0.7, "f1_delta_vs_wazuh": 0.2}),
        encoding="utf-8",
    )
    readiness = tmp_path / "reports/real_measurement_qa/thesis_readiness.md"
    readiness.write_text("# Ready\n", encoding="utf-8")

    output = generate_notes(
        comparison_path=comparison,
        research_answer_path=answer,
        readiness_path=readiness,
        output_dir=tmp_path / "reports/real_measurement_qa",
    )

    text = output.read_text(encoding="utf-8")
    for question in QUESTIONS:
        assert question in text
    assert "A vizsgált lab mérés alapján" in text
