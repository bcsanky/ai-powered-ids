from __future__ import annotations

import pandas as pd

from ml.src.thesis_integration.generate_figures_and_tables_plan import generate_plan


def test_figures_tables_plan_does_not_mark_yes_without_provenance(tmp_path):
    (tmp_path / "results/real_comparison").mkdir(parents=True)
    (tmp_path / "results/real_comparison/metrics_comparison.csv").write_text("configuration,f1\nWazuh-only,0.5\n", encoding="utf-8")

    outputs = generate_plan(tmp_path / "reports/thesis_integration", tmp_path / "reports/real_measurement/measurement_provenance.json")

    manifest = pd.read_csv(outputs["manifest"])
    assert "yes" not in set(manifest["can_include"])

