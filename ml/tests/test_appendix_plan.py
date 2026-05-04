from __future__ import annotations

import pandas as pd

from ml.src.thesis_integration.generate_appendix_plan import generate_appendix_plan


def test_appendix_plan_does_not_auto_include_raw_alert(tmp_path):
    outputs = generate_appendix_plan(tmp_path / "reports/thesis_integration", tmp_path / "reports/real_measurement/measurement_provenance.json")

    manifest = pd.read_csv(outputs["csv"])
    raw = manifest[manifest["source_file"].astype(str).str.contains("alerts.jsonl")]
    assert raw.iloc[0]["include"] == "with_redaction"

