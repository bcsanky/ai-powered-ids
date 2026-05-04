from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.src.generate_security_report import write_report


def test_security_report_is_written_without_scored_events(tmp_path):
    comparison = tmp_path / "metrics_comparison.csv"
    pd.DataFrame(
        [
            {
                "config_name": "ae_minimal",
                "status": "ok",
                "precision": 0.8,
                "recall": 0.7,
                "f1": 0.746,
                "false_positive_rate": 0.1,
                "alert_count": 12,
            },
            {
                "config_name": "baseline_wazuh_real",
                "status": "missing",
                "precision": None,
                "recall": None,
                "f1": None,
                "false_positive_rate": None,
                "alert_count": None,
            },
        ]
    ).to_csv(comparison, index=False)

    outputs = write_report(
        comparison_path=comparison,
        scored_events_path=tmp_path / "missing_events.jsonl",
        output_dir=tmp_path / "report",
    )

    assert outputs["markdown"].exists()
    assert outputs["html"].exists()
    assert outputs["dashboard"].exists()

    text = outputs["markdown"].read_text(encoding="utf-8")
    assert "Natív Wazuh baseline: missing" in text
    assert "nem natív Wazuh teljesítménymérés" in text


def test_security_report_includes_top_risk_events(tmp_path):
    comparison = tmp_path / "metrics_comparison.csv"
    scored = tmp_path / "scored_events.jsonl"
    pd.DataFrame(
        [
            {
                "config_name": "ae_minimal",
                "status": "ok",
                "precision": 0.8,
                "recall": 0.7,
                "f1": 0.746,
                "false_positive_rate": 0.1,
                "alert_count": 12,
            }
        ]
    ).to_csv(comparison, index=False)
    scored.write_text(
        "\n".join(
            [
                '{"event_id":"e1","risk_level":"normal","anomaly_score":0.1}',
                '{"event_id":"e2","risk_level":"critical","anomaly_score":0.9}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    outputs = write_report(
        comparison_path=comparison,
        scored_events_path=scored,
        output_dir=tmp_path / "report",
    )

    text = outputs["markdown"].read_text(encoding="utf-8")
    assert "e2" in text
    assert "critical" in text


def test_report_text_has_no_forbidden_meta_terms(tmp_path):
    comparison = tmp_path / "metrics_comparison.csv"
    pd.DataFrame(
        [
            {
                "config_name": "ae_minimal",
                "status": "ok",
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "false_positive_rate": 0.0,
                "alert_count": 1,
            }
        ]
    ).to_csv(comparison, index=False)

    outputs = write_report(
        comparison_path=comparison,
        scored_events_path=None,
        output_dir=tmp_path / "report",
    )

    text = outputs["markdown"].read_text(encoding="utf-8")
    forbidden = [
        "ChatGPT",
        "Codex",
        "prompt",
        "generated",
        "AI által generált",
        "source of truth",
        "forrásigazság",
    ]
    assert not any(term in text for term in forbidden)
