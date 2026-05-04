from __future__ import annotations

import json

from ml.src.measurement_quality.generate_thesis_measurement_quality_notes import run_notes


def test_thesis_notes_do_not_overclaim_improvement(tmp_path):
    out = tmp_path / "reports/measurement_quality"
    out.mkdir(parents=True)
    (out / "measurement_quality_summary.json").write_text(
        json.dumps({"measurement_quality_status": "MEASUREMENT_WEAK"}),
        encoding="utf-8",
    )
    (out / "research_claim_strength.json").write_text(
        json.dumps({"claim_category": "CLAIM_NOT_SUPPORTED"}),
        encoding="utf-8",
    )

    result = run_notes(out)
    text = result["markdown"].read_text(encoding="utf-8")

    assert "nem igazolható egyértelmű hibrid javulás" in text
    assert "garantált" not in text.lower()
