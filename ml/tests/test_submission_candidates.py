from __future__ import annotations

from pathlib import Path

from ml.src.submission_bundle.collect_submission_candidates import collect_candidates


def write_policy(path: Path) -> None:
    path.write_text(
        """
include_groups:
  source_code:
    - ml/src/**
  examples_demo:
    - examples/lab/**
runtime_outputs_allowed_if_verified:
  - reports/real_measurement/**
always_exclude:
  - data/**
  - raw/**
  - .env
sensitive_patterns:
  - alerts.jsonl
rules:
  require_provenance_for_runtime_results: true
""",
        encoding="utf-8",
    )


def test_source_code_candidate_bekerul(tmp_path: Path) -> None:
    (tmp_path / "ml/src").mkdir(parents=True)
    (tmp_path / "ml/src/app.py").write_text("print('ok')\n", encoding="utf-8")
    policy = tmp_path / "policy.yaml"
    write_policy(policy)

    rows, _status, _errors = collect_candidates(tmp_path, policy)

    app = [row for row in rows if row["relative_path"] == "ml/src/app.py"][0]
    assert app["include_candidate"] is True
    assert app["group"] == "source_code"


def test_wazuh_alerts_soha_nem_candidate(tmp_path: Path) -> None:
    (tmp_path / "data/wazuh").mkdir(parents=True)
    (tmp_path / "data/wazuh/alerts.jsonl").write_text("{}\n", encoding="utf-8")
    policy = tmp_path / "policy.yaml"
    write_policy(policy)

    rows, _status, _errors = collect_candidates(tmp_path, policy)

    assert not any(row["relative_path"] == "data/wazuh/alerts.jsonl" and row["include_candidate"] for row in rows)


def test_examples_lab_demo_groupba_kerul(tmp_path: Path) -> None:
    (tmp_path / "examples/lab").mkdir(parents=True)
    (tmp_path / "examples/lab/README.md").write_text("demo input\n", encoding="utf-8")
    (tmp_path / "examples/lab/lab_events.jsonl").write_text("{}\n", encoding="utf-8")
    policy = tmp_path / "policy.yaml"
    write_policy(policy)

    rows, _status, _errors = collect_candidates(tmp_path, policy)

    event = [row for row in rows if row["relative_path"] == "examples/lab/lab_events.jsonl"][0]
    assert event["group"] == "demo_example"
    assert event["include_candidate"] is True

