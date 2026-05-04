from __future__ import annotations

import hashlib

import pytest

from ml.src.repo_hygiene.create_measurement_provenance import create_provenance


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_create_measurement_provenance_rejects_examples_input(tmp_path):
    ground_truth = tmp_path / "examples/lab/lab_ground_truth.csv"
    lab_features = tmp_path / "data/lab/lab_features.csv"
    wazuh_alerts = tmp_path / "data/wazuh/alerts.jsonl"
    for path in [ground_truth, lab_features, wazuh_alerts]:
        write(path, "x\n")

    with pytest.raises(ValueError):
        create_provenance(
            ground_truth=ground_truth,
            lab_features=lab_features,
            wazuh_alerts=wazuh_alerts,
            output=tmp_path / "reports/real_measurement/measurement_provenance.json",
        )


def test_create_measurement_provenance_hashes_inputs(tmp_path):
    ground_truth = tmp_path / "data/lab/lab_ground_truth.csv"
    lab_features = tmp_path / "data/lab/lab_features.csv"
    wazuh_alerts = tmp_path / "data/wazuh/alerts.jsonl"
    write(ground_truth, "gt\n")
    write(lab_features, "features\n")
    write(wazuh_alerts, "{}\n")

    payload = create_provenance(
        ground_truth=ground_truth,
        lab_features=lab_features,
        wazuh_alerts=wazuh_alerts,
        output=tmp_path / "reports/real_measurement/measurement_provenance.json",
    )

    assert payload["measurement_source"] == "real_lab"
    assert payload["ground_truth_sha256"] == hashlib.sha256(ground_truth.read_bytes()).hexdigest()
