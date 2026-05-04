from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ml.src.live_integration.enrich_wazuh_alerts import enrich_wazuh_alerts, hybrid_priority
from ml.src.scoring_runtime import ScoringResult


class FakeScorer:
    def __init__(self, *, ml_alert: bool = True, score: float = 0.9) -> None:
        self.ml_alert = ml_alert
        self.score = score

    def score_event(self, *, event_id, features, rule_flag=False, rule_level=0):  # noqa: ANN001
        return ScoringResult(
            event_id=event_id,
            model_loaded=True,
            model_version="test",
            anomaly_score=self.score,
            threshold_name="fixed",
            threshold_value=0.5,
            ml_alert=self.ml_alert,
            rule_flag=bool(rule_flag),
            rule_level=int(rule_level),
            risk_level="medium",
            reason="teszt scoring",
        )


def write_common_inputs(tmp_path: Path) -> dict[str, Path]:
    data_dir = tmp_path / "data/lab"
    data_dir.mkdir(parents=True)
    ground_truth = data_dir / "lab_ground_truth.csv"
    features = data_dir / "lab_features.csv"
    alerts = tmp_path / "data/wazuh/alerts.jsonl"
    alerts.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "event_id": "lab-001",
                "timestamp_start": "2026-05-04T10:00:00Z",
                "timestamp_end": "2026-05-04T10:05:00Z",
                "scenario": "port_scan",
                "label": "attack",
                "attack_type": "port_scan",
                "source_ip": "10.0.0.2",
                "target_ip": "10.0.0.10",
            }
        ]
    ).to_csv(ground_truth, index=False)
    pd.DataFrame(
        [
            {
                "event_id": "lab-001",
                "timestamp": "2026-05-04T10:00:10Z",
                "destination_port": 22,
                "flow_duration": 1.0,
                "total_fwd_packets": 10,
                "total_backward_packets": 4,
                "flow_bytes_per_sec": 1200.0,
                "flow_packets_per_sec": 14.0,
                "protocol": "tcp",
                "source_ip": "10.0.0.2",
                "target_ip": "10.0.0.10",
                "scenario": "port_scan",
            }
        ]
    ).to_csv(features, index=False)
    return {"ground_truth": ground_truth, "features": features, "alerts": alerts}


def write_provenance(tmp_path: Path, paths: dict[str, Path]) -> Path:
    provenance = tmp_path / "reports/real_measurement/measurement_provenance.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text(
        json.dumps(
            {
                "measurement_source": "real_lab",
                "ground_truth_path": paths["ground_truth"].as_posix(),
                "lab_features_path": paths["features"].as_posix(),
                "wazuh_alerts_path": paths["alerts"].as_posix(),
                "ground_truth_sha256": "a",
                "lab_features_sha256": "b",
                "wazuh_alerts_sha256": "c",
            }
        ),
        encoding="utf-8",
    )
    return provenance


def run_enrichment(tmp_path: Path, paths: dict[str, Path], provenance: Path, scorer: FakeScorer) -> pd.DataFrame:
    output_dir = tmp_path / "reports/live_integration"
    enrich_wazuh_alerts(
        wazuh_alerts_path=paths["alerts"],
        lab_features_path=paths["features"],
        ground_truth_path=paths["ground_truth"],
        output_jsonl_path=output_dir / "enriched_alerts.jsonl",
        output_csv_path=output_dir / "enriched_alerts.csv",
        model_root=tmp_path / "model-root",
        preprocess_path=tmp_path / "preprocess.pkl",
        provenance_path=provenance,
        require_provenance=True,
        allow_time_only_match=False,
        scorer=scorer,
    )
    return pd.read_csv(output_dir / "enriched_alerts.csv")


def test_demo_path_input_fails_in_real_mode(tmp_path):
    with pytest.raises(ValueError, match="demo/sample/fixture"):
        enrich_wazuh_alerts(
            wazuh_alerts_path=tmp_path / "examples/lab/alerts.jsonl",
            lab_features_path=tmp_path / "data/lab/lab_features.csv",
            ground_truth_path=tmp_path / "data/lab/lab_ground_truth.csv",
            output_jsonl_path=tmp_path / "out/enriched.jsonl",
            output_csv_path=tmp_path / "out/enriched.csv",
            model_root=tmp_path / "model-root",
            preprocess_path=tmp_path / "preprocess.pkl",
            provenance_path=None,
            require_provenance=False,
            allow_time_only_match=False,
            scorer=FakeScorer(),
        )


def test_require_provenance_fails_when_missing(tmp_path):
    paths = write_common_inputs(tmp_path)
    with pytest.raises(ValueError, match="provenance"):
        enrich_wazuh_alerts(
            wazuh_alerts_path=paths["alerts"],
            lab_features_path=paths["features"],
            ground_truth_path=paths["ground_truth"],
            output_jsonl_path=tmp_path / "out/enriched.jsonl",
            output_csv_path=tmp_path / "out/enriched.csv",
            model_root=tmp_path / "model-root",
            preprocess_path=tmp_path / "preprocess.pkl",
            provenance_path=tmp_path / "missing_provenance.json",
            require_provenance=True,
            allow_time_only_match=False,
            scorer=FakeScorer(),
        )


def test_event_id_match_scores_alert_and_sets_critical_priority(tmp_path):
    paths = write_common_inputs(tmp_path)
    paths["alerts"].write_text(
        json.dumps(
            {
                "timestamp": "2026-05-04T10:01:00Z",
                "rule": {"id": "5710", "level": 7, "description": "ssh alert"},
                "data": {"event_id": "lab-001", "srcip": "10.0.0.2", "dstip": "10.0.0.10"},
                "agent": {"name": "target"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    provenance = write_provenance(tmp_path, paths)

    df = run_enrichment(tmp_path, paths, provenance, FakeScorer(ml_alert=True))

    assert df.loc[0, "event_id"] == "lab-001"
    assert df.loc[0, "match_method"] == "event_id"
    assert df.loc[0, "top_level_status"] == "scored"
    assert df.loc[0, "hybrid_priority_level"] == "critical"


def test_timestamp_and_ip_match_scores_alert(tmp_path):
    paths = write_common_inputs(tmp_path)
    paths["alerts"].write_text(
        json.dumps(
            {
                "timestamp": "2026-05-04T10:01:00Z",
                "rule": {"id": "5710", "level": 7},
                "data": {"srcip": "10.0.0.2", "dstip": "10.0.0.10"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    provenance = write_provenance(tmp_path, paths)

    df = run_enrichment(tmp_path, paths, provenance, FakeScorer(ml_alert=False))

    assert df.loc[0, "event_id"] == "lab-001"
    assert df.loc[0, "match_method"] == "time_ip"
    assert df.loc[0, "hybrid_priority_level"] == "high"


def test_unmatched_alert_is_written_to_unmatched_csv(tmp_path):
    paths = write_common_inputs(tmp_path)
    paths["alerts"].write_text(
        json.dumps({"timestamp": "2026-05-04T11:01:00Z", "rule": {"id": "1", "level": 3}}) + "\n",
        encoding="utf-8",
    )
    provenance = write_provenance(tmp_path, paths)
    output_dir = tmp_path / "reports/live_integration"

    enrich_wazuh_alerts(
        wazuh_alerts_path=paths["alerts"],
        lab_features_path=paths["features"],
        ground_truth_path=paths["ground_truth"],
        output_jsonl_path=output_dir / "enriched_alerts.jsonl",
        output_csv_path=output_dir / "enriched_alerts.csv",
        model_root=tmp_path / "model-root",
        preprocess_path=tmp_path / "preprocess.pkl",
        provenance_path=provenance,
        require_provenance=True,
        allow_time_only_match=False,
        scorer=FakeScorer(),
    )

    unmatched = pd.read_csv(output_dir / "unmatched_alerts.csv")
    assert len(unmatched) == 1
    assert unmatched.loc[0, "top_level_status"] == "unmatched"


def test_hybrid_priority_levels_are_deterministic():
    assert hybrid_priority(1, True)[:2] == ("critical", 1)
    assert hybrid_priority(1, False)[:2] == ("high", 1)
    assert hybrid_priority(0, True)[:2] == ("medium", 1)
    assert hybrid_priority(0, False)[:2] == ("normal", 0)

