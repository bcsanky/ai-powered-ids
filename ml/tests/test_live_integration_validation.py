from __future__ import annotations

import json

import pandas as pd

from ml.src.live_integration.validate_live_integration_outputs import validate_outputs


def write_live_outputs(output_dir, *, total: int, scored: int, unmatched: int, provenance_valid: bool = True):  # noqa: ANN001
    output_dir.mkdir(parents=True)
    rows = []
    rows.extend(
        {
            "top_level_status": "scored",
            "ml_alert": True,
            "wazuh_pred": 1,
            "hybrid_or_pred": 1,
            "risk_level": "critical",
        }
        for _ in range(scored)
    )
    rows.extend(
        {
            "top_level_status": "unmatched",
            "ml_alert": False,
            "wazuh_pred": 1,
            "hybrid_or_pred": 1,
            "risk_level": "high",
        }
        for _ in range(unmatched)
    )
    pd.DataFrame(rows).to_csv(output_dir / "enriched_alerts.csv", index=False)
    pd.DataFrame(
        [
            {
                "total_alerts": total,
                "scored_alerts": scored,
                "unmatched_alerts": unmatched,
                "error_alerts": 0,
                "ml_positive_count": scored,
                "wazuh_positive_count": total,
                "hybrid_positive_count": total,
                "critical_count": scored,
                "high_count": unmatched,
                "medium_count": 0,
                "normal_count": 0,
            }
        ]
    ).to_csv(output_dir / "enrichment_summary.csv", index=False)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "wazuh_alerts": "data/wazuh/alerts.jsonl",
                "lab_features": "data/lab/lab_features.csv",
                "ground_truth": "data/lab/lab_ground_truth.csv",
                "provenance_valid": provenance_valid,
                "provenance_errors": [],
            }
        ),
        encoding="utf-8",
    )


def test_validation_not_ready_when_no_scored_alerts(tmp_path):
    output_dir = tmp_path / "reports/live_integration"
    write_live_outputs(output_dir, total=2, scored=0, unmatched=2)

    outputs = validate_outputs(output_dir)
    readiness = json.loads(outputs["json"].read_text(encoding="utf-8"))

    assert readiness["readiness"] == "NOT_READY"


def test_validation_ready_with_limitations_when_unmatched_ratio_high(tmp_path):
    output_dir = tmp_path / "reports/live_integration"
    write_live_outputs(output_dir, total=3, scored=1, unmatched=2)

    outputs = validate_outputs(output_dir)
    readiness = json.loads(outputs["json"].read_text(encoding="utf-8"))

    assert readiness["readiness"] == "READY_WITH_LIMITATIONS"

