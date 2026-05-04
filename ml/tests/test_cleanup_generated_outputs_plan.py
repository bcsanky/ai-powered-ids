from __future__ import annotations

import pandas as pd

from ml.src.repo_hygiene.cleanup_generated_outputs_plan import create_cleanup_plan


def test_cleanup_plan_marks_reports_lab_as_demo_or_remove(tmp_path):
    path = tmp_path / "reports/lab/case_study_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("demo\n", encoding="utf-8")

    outputs = create_cleanup_plan(tmp_path, tmp_path / "reports/repo_hygiene")
    df = pd.read_csv(outputs["csv"])
    row = df[df["relative_path"] == "reports/lab/case_study_summary.md"].iloc[0]

    assert row["suggested_action"] in {"keep_but_mark_demo", "remove_from_git"}
