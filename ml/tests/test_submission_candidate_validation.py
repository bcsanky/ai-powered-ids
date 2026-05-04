from __future__ import annotations

from pathlib import Path

from ml.src.submission_bundle.common import write_csv
from ml.src.submission_bundle.validate_submission_candidates import validate_candidates


def policy(path: Path) -> None:
    path.write_text(
        """
include_groups: {}
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


def test_raw_wazuh_export_soha_nem_valid_candidate(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw/wazuh_export.json").write_text("[]\n", encoding="utf-8")
    candidates = tmp_path / "submission_candidates.csv"
    write_csv(
        candidates,
        [
            {
                "relative_path": "raw/wazuh_export.json",
                "group": "source_code",
                "exists": True,
                "include_candidate": True,
                "requires_provenance": False,
                "reason": "",
                "warning": "",
            }
        ],
        ["relative_path", "group", "exists", "include_candidate", "requires_provenance", "reason", "warning"],
    )
    policy_path = tmp_path / "policy.yaml"
    policy(policy_path)

    rows, _status = validate_candidates(tmp_path, policy_path, candidates)

    assert any(row["status"] == "FAIL" and "Tiltott" in row["message"] for row in rows)


def test_env_soha_nem_valid_candidate(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("TOKEN=x\n", encoding="utf-8")
    candidates = tmp_path / "submission_candidates.csv"
    write_csv(
        candidates,
        [
            {
                "relative_path": ".env",
                "group": "configs",
                "exists": True,
                "include_candidate": True,
                "requires_provenance": False,
                "reason": "",
                "warning": "",
            }
        ],
        ["relative_path", "group", "exists", "include_candidate", "requires_provenance", "reason", "warning"],
    )
    policy_path = tmp_path / "policy.yaml"
    policy(policy_path)

    rows, _status = validate_candidates(tmp_path, policy_path, candidates)

    assert any(row["status"] == "FAIL" for row in rows)


def test_runtime_result_provenance_nelkul_fail(tmp_path: Path) -> None:
    (tmp_path / "reports/real_measurement").mkdir(parents=True)
    (tmp_path / "reports/real_measurement/real_lab_results_report.md").write_text("report\n", encoding="utf-8")
    candidates = tmp_path / "submission_candidates.csv"
    write_csv(
        candidates,
        [
            {
                "relative_path": "reports/real_measurement/real_lab_results_report.md",
                "group": "runtime_output",
                "exists": True,
                "include_candidate": True,
                "requires_provenance": True,
                "reason": "",
                "warning": "",
            }
        ],
        ["relative_path", "group", "exists", "include_candidate", "requires_provenance", "reason", "warning"],
    )
    policy_path = tmp_path / "policy.yaml"
    policy(policy_path)

    rows, _status = validate_candidates(tmp_path, policy_path, candidates)

    assert any(row["status"] == "FAIL" and "provenance" in row["message"] for row in rows)

