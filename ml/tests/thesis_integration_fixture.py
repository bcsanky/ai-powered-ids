from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.src.repo_hygiene.common import sha256_file


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def create_thesis_measurement(root: Path, *, hybrid_f1: float = 0.7, with_provenance: bool = True) -> None:
    ground_truth = root / "data/lab/lab_ground_truth.csv"
    lab_features = root / "data/lab/lab_features.csv"
    wazuh_alerts = root / "data/wazuh/alerts.jsonl"
    write_csv(
        ground_truth,
        [
            {"event_id": "e1", "timestamp_start": "2026-05-04T10:00:00Z", "timestamp_end": "2026-05-04T10:01:00Z", "label": "benign"},
            {"event_id": "e2", "timestamp_start": "2026-05-04T10:02:00Z", "timestamp_end": "2026-05-04T10:03:00Z", "label": "attack"},
        ],
    )
    write_csv(lab_features, [{"event_id": "e1"}, {"event_id": "e2"}])
    wazuh_alerts.parent.mkdir(parents=True, exist_ok=True)
    wazuh_alerts.write_text('{"timestamp":"2026-05-04T10:02:10Z","rule":{"id":"1001"}}\n', encoding="utf-8")

    base = {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "false_positive_rate": 0.2,
        "false_negative_rate": 0.5,
        "alert_count": 2,
        "mean_ttd": 10.0,
        "median_ttd": 10.0,
        "n_samples": 4,
        "n_attack": 2,
        "n_benign": 2,
    }
    write_csv(root / "results/wazuh_real/metrics_summary.csv", [base])
    write_csv(root / "results/ae_lab/metrics_summary.csv", [{**base, "f1": 0.4}])
    write_csv(
        root / "results/hybrid_real/metrics_summary.csv",
        [
            {**base, "strategy": "hybrid_or", "f1": hybrid_f1, "recall": hybrid_f1},
            {**base, "strategy": "hybrid_weighted", "f1": 0.45},
            {**base, "strategy": "hybrid_priority", "f1": min(0.45, hybrid_f1)},
        ],
    )
    write_csv(
        root / "results/real_comparison/metrics_comparison.csv",
        [
            {"configuration": "Wazuh-only", **base},
            {"configuration": "AE-Minimal lab", **base, "f1": 0.4},
            {"configuration": "Hybrid OR", **base, "f1": hybrid_f1, "recall": hybrid_f1},
            {"configuration": "Hybrid weighted", **base, "f1": 0.45},
            {"configuration": "Hybrid priority", **base, "f1": min(0.45, hybrid_f1)},
        ],
    )
    if with_provenance:
        provenance = {
            "measurement_source": "real_lab",
            "created_at": "2026-05-04T10:10:00Z",
            "ground_truth_path": ground_truth.as_posix(),
            "lab_features_path": lab_features.as_posix(),
            "wazuh_alerts_path": wazuh_alerts.as_posix(),
            "ground_truth_sha256": sha256_file(ground_truth),
            "lab_features_sha256": sha256_file(lab_features),
            "wazuh_alerts_sha256": sha256_file(wazuh_alerts),
            "result_files": [
                "results/wazuh_real/metrics_summary.csv",
                "results/ae_lab/metrics_summary.csv",
                "results/hybrid_real/metrics_summary.csv",
                "results/real_comparison/metrics_comparison.csv",
            ],
        }
        path = root / "reports/real_measurement/measurement_provenance.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")

