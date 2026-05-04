from __future__ import annotations

import json

from ml.src.measurement_quality.generate_measurement_quality_summary import run_summary


def write_check(path, status):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "check_id,category,status,message,recommendation,value\n"
        f"x,test,{status},message,,\n",
        encoding="utf-8",
    )


def test_summary_not_ready_on_critical_fail(tmp_path):
    out = tmp_path / "reports/measurement_quality"
    for name in ["scenario_coverage.csv", "feature_alert_alignment.csv", "metric_consistency.csv", "ttd_quality.csv"]:
        write_check(out / name, "PASS")
    write_check(out / "metric_consistency.csv", "FAIL")
    (out / "research_claim_strength.json").write_text(json.dumps({"claim_category": "CLAIM_SUPPORTED_WITH_LIMITATIONS"}), encoding="utf-8")

    result = run_summary(out)

    assert result["status"] == "MEASUREMENT_NOT_READY"


def test_summary_weak_when_claim_not_supported(tmp_path):
    out = tmp_path / "reports/measurement_quality"
    for name in ["scenario_coverage.csv", "feature_alert_alignment.csv", "metric_consistency.csv", "ttd_quality.csv"]:
        write_check(out / name, "PASS")
    (out / "research_claim_strength.json").write_text(json.dumps({"claim_category": "CLAIM_NOT_SUPPORTED"}), encoding="utf-8")

    result = run_summary(out)

    assert result["status"] == "MEASUREMENT_WEAK"

