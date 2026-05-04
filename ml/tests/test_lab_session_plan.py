from __future__ import annotations

import yaml

from ml.src.lab_session.create_session_plan import create_session_plan


def test_create_session_plan_writes_only_plan_and_log_template(tmp_path):
    output_dir = tmp_path / "reports/lab_session"

    outputs = create_session_plan(
        session_id="real-lab-test",
        attacker_ip="192.168.56.20",
        target_ip="192.168.56.10",
        wazuh_manager="192.168.56.5",
        output_dir=output_dir,
    )
    plan = yaml.safe_load(outputs["yaml"].read_text(encoding="utf-8"))

    assert plan["session_id"] == "real-lab-test"
    assert "port_scan" in plan["planned_scenarios"]
    assert outputs["markdown"].exists()
    assert outputs["operator_log_template"].exists()
    assert not (tmp_path / "data/lab/lab_ground_truth.csv").exists()
    assert not (tmp_path / "data/wazuh/alerts.jsonl").exists()

