from __future__ import annotations

import json

import pandas as pd

from ml.src.measurement_quality.check_feature_alert_alignment import run_check


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_thresholds(path):
    path.write_text("minimums:\n  max_unmatched_alert_ratio: 0.5\n  max_missing_ae_score_ratio: 0.1\n", encoding="utf-8")


def write_provenance(path, gt, features):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "measurement_source": "real_lab",
                "ground_truth_path": gt.as_posix(),
                "lab_features_path": features.as_posix(),
                "wazuh_alerts_path": "data/wazuh/alerts.jsonl",
                "ground_truth_sha256": "a",
                "lab_features_sha256": "b",
                "wazuh_alerts_sha256": "c",
            }
        ),
        encoding="utf-8",
    )


def test_alignment_event_id_mismatch_fails(tmp_path):
    gt = tmp_path / "data/lab/lab_ground_truth.csv"
    features = tmp_path / "data/lab/lab_features.csv"
    wazuh = tmp_path / "results/wazuh_real/predictions.csv"
    ae = tmp_path / "results/ae_lab/predictions.csv"
    hybrid = tmp_path / "results/hybrid_real/predictions.csv"
    thresholds = tmp_path / "thresholds.yaml"
    provenance = tmp_path / "reports/real_measurement/measurement_provenance.json"
    write_thresholds(thresholds)
    write_csv(gt, [{"event_id": "e1", "label": "attack", "scenario": "port_scan", "source_ip": "1", "target_ip": "2"}])
    write_csv(features, [{"event_id": "other", "source_ip": "1", "target_ip": "2"}])
    write_csv(wazuh, [{"event_id": "e1", "y_true": 1, "wazuh_pred": 1}])
    write_csv(ae, [{"event_id": "e1", "y_true": 1, "ae_pred": 1, "anomaly_score": 0.9}])
    write_csv(hybrid, [{"event_id": "e1", "hybrid_or_pred": 1, "hybrid_weighted_pred": 1, "hybrid_priority_pred": 1}])
    write_provenance(provenance, gt, features)

    result = run_check(
        ground_truth=gt,
        lab_features=features,
        wazuh_predictions=wazuh,
        ae_predictions=ae,
        hybrid_predictions=hybrid,
        thresholds_path=thresholds,
        provenance_path=provenance,
        output_dir=tmp_path / "reports/measurement_quality",
    )

    assert any(row["check_id"] == "features_event_set" and row["status"] == "FAIL" for row in result["rows"])


def test_alignment_missing_ae_score_ratio_fails(tmp_path):
    gt = tmp_path / "data/lab/lab_ground_truth.csv"
    features = tmp_path / "data/lab/lab_features.csv"
    wazuh = tmp_path / "results/wazuh_real/predictions.csv"
    ae = tmp_path / "results/ae_lab/predictions.csv"
    hybrid = tmp_path / "results/hybrid_real/predictions.csv"
    thresholds = tmp_path / "thresholds.yaml"
    provenance = tmp_path / "reports/real_measurement/measurement_provenance.json"
    write_thresholds(thresholds)
    write_csv(gt, [{"event_id": "e1", "label": "attack", "scenario": "port_scan", "source_ip": "1", "target_ip": "2"}])
    write_csv(features, [{"event_id": "e1", "source_ip": "1", "target_ip": "2"}])
    write_csv(wazuh, [{"event_id": "e1", "y_true": 1, "wazuh_pred": 1}])
    write_csv(ae, [{"event_id": "e1", "y_true": 1, "ae_pred": 1, "anomaly_score": ""}])
    write_csv(hybrid, [{"event_id": "e1", "hybrid_or_pred": 1, "hybrid_weighted_pred": 1, "hybrid_priority_pred": 1}])
    write_provenance(provenance, gt, features)

    result = run_check(
        ground_truth=gt,
        lab_features=features,
        wazuh_predictions=wazuh,
        ae_predictions=ae,
        hybrid_predictions=hybrid,
        thresholds_path=thresholds,
        provenance_path=provenance,
        output_dir=tmp_path / "reports/measurement_quality",
    )

    assert any(row["check_id"] == "missing_ae_score_ratio" and row["status"] == "FAIL" for row in result["rows"])

