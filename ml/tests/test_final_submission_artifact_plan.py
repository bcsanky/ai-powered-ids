from __future__ import annotations

import pandas as pd

from ml.src.final_submission_check.check_submission_artifact_plan import run_check


def test_submission_artifact_plan_does_not_auto_include_raw_wazuh(tmp_path):
    manifest = tmp_path / "reports/thesis_integration/appendix_manifest.csv"
    manifest.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"source_file": "data/wazuh/alerts.jsonl", "include": "with_redaction"},
            {"source_file": "templates/lab/lab_ground_truth_template.csv", "include": "true"},
        ]
    ).to_csv(manifest, index=False)
    docs = tmp_path / "docs/thesis_result_artifacts.md"
    docs.parent.mkdir()
    docs.write_text("examples/lab demonstrációs bemenet. runbook dokumentáció.\n", encoding="utf-8")

    result = run_check(tmp_path / "reports/final_submission_check", tmp_path)

    statuses = {row["check_id"]: row["status"] for row in result["rows"]}
    assert statuses["raw_inputs_not_auto_included"] == "PASS"
    assert statuses["examples_lab_not_measurement"] == "PASS"

