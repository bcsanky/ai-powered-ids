from __future__ import annotations

import pandas as pd

from ml.src.repo_hygiene.audit_generated_artifacts import audit


def write(path, content="x\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_repo_hygiene_audit_classifies_demo_and_sensitive_files(tmp_path):
    write(tmp_path / "examples/lab/lab_events.jsonl")
    write(tmp_path / "reports/lab/case_study_summary.md")
    write(tmp_path / "data/wazuh/alerts.jsonl")

    outputs = audit(tmp_path, tmp_path / "reports/repo_hygiene")
    df = pd.read_csv(outputs["csv"])

    categories = dict(zip(df["relative_path"], df["category"]))
    assert categories["examples/lab/lab_events.jsonl"] == "demo_input"
    assert categories["reports/lab/case_study_summary.md"] == "demo_output"
    assert categories["data/wazuh/alerts.jsonl"] == "sensitive_input"
