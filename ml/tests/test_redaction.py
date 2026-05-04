from __future__ import annotations

import json

from ml.src.real_measurement.redact_measurement_outputs import redact_outputs


def test_redaction_replaces_ips_and_preserves_input(tmp_path):
    input_dir = tmp_path / "reports/real_measurement"
    output_dir = tmp_path / "reports/real_measurement_redacted"
    mapping = output_dir / "redaction_mapping.json"
    input_dir.mkdir(parents=True)
    source = input_dir / "real_lab_results_report.md"
    source.write_text("Alert from 192.168.56.20 to 192.168.56.10\n", encoding="utf-8")
    csv_file = input_dir / "events.csv"
    csv_file.write_text("agent_name,source_ip\nlab-target,192.168.56.20\n", encoding="utf-8")

    result = redact_outputs(input_dir, output_dir, mapping)

    redacted = (output_dir / "real_lab_results_report.md").read_text(encoding="utf-8")
    assert "192.168.56.20" not in redacted
    assert "IP_001" in redacted
    assert source.read_text(encoding="utf-8") == "Alert from 192.168.56.20 to 192.168.56.10\n"
    mapping_data = json.loads(mapping.read_text(encoding="utf-8"))
    assert mapping_data["ip"]["192.168.56.20"] == "IP_001"
    assert result["files"]
    redacted_csv = (output_dir / "events.csv").read_text(encoding="utf-8")
    assert "HOST_001" in redacted_csv
