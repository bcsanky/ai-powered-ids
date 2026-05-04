from __future__ import annotations

import json

import pandas as pd

from ml.src.real_measurement_qa.postrun_quality_gate import run_quality_gate
from ml.src.repo_hygiene.create_measurement_provenance import create_provenance


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def create_postrun_bundle(root, *, hybrid_f1=0.7, include_comparison=True, with_provenance=True):
    ground_truth = root / "data/lab/lab_ground_truth.csv"
    lab_features = root / "data/lab/lab_features.csv"
    wazuh_alerts = root / "data/wazuh/alerts.jsonl"
    write_csv(ground_truth, [{"event_id": "e1", "label": "benign"}, {"event_id": "e2", "label": "attack"}])
    write_csv(lab_features, [{"event_id": "e1"}, {"event_id": "e2"}])
    wazuh_alerts.parent.mkdir(parents=True, exist_ok=True)
    wazuh_alerts.write_text('{"timestamp":"2026-05-04T10:00:00Z"}\n', encoding="utf-8")
    metrics = {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "false_positive_rate": 0.2,
        "false_negative_rate": 0.5,
        "alert_count": 2,
        "n_samples": 4,
        "n_attack": 2,
        "n_benign": 2,
    }
    write_csv(root / "results/wazuh_real/metrics_summary.csv", [metrics])
    write_csv(root / "results/ae_lab/metrics_summary.csv", [{**metrics, "f1": 0.4}])
    write_csv(
        root / "results/hybrid_real/metrics_summary.csv",
        [
            {**metrics, "strategy": "hybrid_or", "f1": hybrid_f1, "recall": hybrid_f1},
            {**metrics, "strategy": "hybrid_weighted", "f1": 0.45},
            {**metrics, "strategy": "hybrid_priority", "f1": min(0.45, hybrid_f1)},
        ],
    )
    if include_comparison:
        write_csv(
            root / "results/real_comparison/metrics_comparison.csv",
            [
                {"configuration": "Wazuh-only", **metrics},
                {"configuration": "AE-Minimal lab", **metrics, "f1": 0.4},
                {"configuration": "Hybrid OR", **metrics, "f1": hybrid_f1, "recall": hybrid_f1},
                {"configuration": "Hybrid weighted", **metrics, "f1": 0.45},
                {"configuration": "Hybrid priority", **metrics, "f1": min(0.45, hybrid_f1)},
            ],
        )
    md = root / "results/real_comparison/metrics_comparison.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("| ok |\n", encoding="utf-8")
    for rel_path in [
        "reports/real_measurement/real_lab_results_report.md",
        "reports/real_measurement/thesis_real_lab_section.md",
    ]:
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n", encoding="utf-8")
    write_csv(root / "reports/real_measurement/measurement_manifest.csv", [{"relative_path": "x", "sha256": "abc"}])
    if with_provenance:
        create_provenance(
            ground_truth=ground_truth,
            lab_features=lab_features,
            wazuh_alerts=wazuh_alerts,
            output=root / "reports/real_measurement/measurement_provenance.json",
        )


def test_postrun_quality_gate_ready_for_consistent_metrics(tmp_path):
    create_postrun_bundle(tmp_path)

    result = run_quality_gate(tmp_path, tmp_path / "reports/real_measurement_qa")

    assert result["status"] == "READY"
    answer = json.loads(result["answer"].read_text(encoding="utf-8"))
    assert answer["best_hybrid_by_f1"] == "Hybrid OR"


def test_postrun_quality_gate_not_ready_without_comparison(tmp_path):
    create_postrun_bundle(tmp_path, include_comparison=False)

    result = run_quality_gate(tmp_path, tmp_path / "reports/real_measurement_qa")

    assert result["status"] == "NOT_READY"


def test_postrun_quality_gate_not_ready_without_provenance(tmp_path):
    create_postrun_bundle(tmp_path, with_provenance=False)

    result = run_quality_gate(tmp_path, tmp_path / "reports/real_measurement_qa")

    assert result["status"] == "NOT_READY"


def test_postrun_quality_gate_does_not_claim_improvement_when_f1_not_higher(tmp_path):
    create_postrun_bundle(tmp_path, hybrid_f1=0.4)

    result = run_quality_gate(tmp_path, tmp_path / "reports/real_measurement_qa")

    readiness = result["readiness"].read_text(encoding="utf-8")
    assert "nem igazolható egyértelmű F1 javulás" in readiness


def test_postrun_quality_gate_uses_cautious_improvement_when_f1_higher(tmp_path):
    create_postrun_bundle(tmp_path, hybrid_f1=0.7)

    result = run_quality_gate(tmp_path, tmp_path / "reports/real_measurement_qa")

    readiness = result["readiness"].read_text(encoding="utf-8")
    assert "A vizsgált lab mérés alapján" in readiness
    assert "F1 javulás figyelhető meg" in readiness
