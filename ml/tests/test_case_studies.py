from __future__ import annotations

import pandas as pd
import pytest

from ml.src.generate_case_studies import generate_case_studies


def test_case_studies_create_summary_files(tmp_path):
    scored = tmp_path / "scored_events.jsonl"
    scored.write_text(
        "\n".join(
            [
                '{"event_id":"e1","timestamp":"2026-05-08T09:00:00Z","scenario":"benign_activity","description":"normál","anomaly_score":0.1,"ml_alert":false,"rule_flag":false,"risk_level":"normal","reason":"ok"}',
                '{"event_id":"e2","timestamp":"2026-05-08T09:01:00Z","scenario":"port_scan","description":"scan","anomaly_score":0.8,"ml_alert":true,"rule_flag":true,"risk_level":"critical","reason":"jelzés"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    outputs = generate_case_studies(scored, tmp_path / "out")

    assert outputs["summary"].exists()
    assert outputs["scenario_summary"].exists()
    assert outputs["risk_matrix"].exists()
    assert outputs["timeline"].exists()
    assert outputs["risk_distribution"].exists()
    assert (tmp_path / "out" / "case_study_port_scan.md").exists()

    summary = pd.read_csv(outputs["scenario_summary"])
    assert {"scenario", "event_count", "critical_count", "ml_alert_count"}.issubset(summary.columns)


def test_case_studies_risk_matrix_contains_scenario_and_risk_levels(tmp_path):
    scored = tmp_path / "scored_events.csv"
    pd.DataFrame(
        [
            {
                "event_id": "e1",
                "scenario": "ssh_bruteforce",
                "anomaly_score": 0.4,
                "ml_alert": True,
                "rule_flag": True,
                "risk_level": "high",
            }
        ]
    ).to_csv(scored, index=False)

    outputs = generate_case_studies(scored, tmp_path / "out")
    matrix = pd.read_csv(outputs["risk_matrix"])

    assert "scenario" in matrix.columns
    assert "high" in matrix.columns
    assert matrix.loc[0, "scenario"] == "ssh_bruteforce"


def test_case_studies_raise_clear_error_for_missing_required_columns(tmp_path):
    scored = tmp_path / "bad.csv"
    pd.DataFrame([{"event_id": "e1", "scenario": "port_scan"}]).to_csv(scored, index=False)

    with pytest.raises(ValueError, match="Hiányzó kötelező oszlopok"):
        generate_case_studies(scored, tmp_path / "out")
