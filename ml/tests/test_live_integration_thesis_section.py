from __future__ import annotations

import json

import pandas as pd

from ml.src.live_integration.generate_live_integration_thesis_section import generate_thesis_section


def test_thesis_section_describes_integration_not_benchmark(tmp_path):
    output_dir = tmp_path / "reports/live_integration"
    output_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "total_alerts": 3,
                "scored_alerts": 2,
                "unmatched_alerts": 1,
                "ml_positive_count": 1,
                "wazuh_positive_count": 3,
                "hybrid_positive_count": 3,
                "critical_count": 1,
                "high_count": 2,
                "medium_count": 0,
                "normal_count": 0,
            }
        ]
    ).to_csv(output_dir / "enrichment_summary.csv", index=False)
    (output_dir / "live_integration_readiness.json").write_text(
        json.dumps({"readiness": "READY_WITH_LIMITATIONS"}),
        encoding="utf-8",
    )

    outputs = generate_thesis_section(output_dir, tmp_path / "missing_provenance.json")
    text = outputs["thesis"].read_text(encoding="utf-8")

    assert "integrációs bizonyíték" in text
    assert "nem önálló benchmark" in text
    assert "detektálási javulás" not in text

