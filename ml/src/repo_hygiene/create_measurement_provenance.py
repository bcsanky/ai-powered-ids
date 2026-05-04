from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ml.src.repo_hygiene.common import is_demo_or_fixture_path, sha256_file


RESULT_FILES = [
    "results/wazuh_real/metrics_summary.csv",
    "results/ae_lab/metrics_summary.csv",
    "results/hybrid_real/metrics_summary.csv",
    "results/real_comparison/metrics_comparison.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--lab-features", required=True)
    parser.add_argument("--wazuh-alerts", required=True)
    parser.add_argument("--output", default="reports/real_measurement/measurement_provenance.json")
    return parser.parse_args()


def validate_input_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó {label}: {path}")
    if is_demo_or_fixture_path(path):
        raise ValueError(f"A(z) {label} nem lehet demo/sample/fixture eredetű: {path}")


def result_hashes(result_files: list[str]) -> tuple[dict[str, str], list[str]]:
    hashes = {}
    warnings = []
    for rel_path in result_files:
        path = Path(rel_path)
        if path.exists():
            hashes[rel_path] = sha256_file(path)
        else:
            warnings.append(f"Hiányzó eredményfájl: {rel_path}")
    return hashes, warnings


def create_provenance(
    *,
    ground_truth: Path,
    lab_features: Path,
    wazuh_alerts: Path,
    output: Path,
) -> dict[str, Any]:
    validate_input_path(ground_truth, "ground truth")
    validate_input_path(lab_features, "lab feature")
    validate_input_path(wazuh_alerts, "Wazuh alert export")
    hashes, warnings = result_hashes(RESULT_FILES)
    payload: dict[str, Any] = {
        "measurement_source": "real_lab",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ground_truth_path": ground_truth.as_posix(),
        "lab_features_path": lab_features.as_posix(),
        "wazuh_alerts_path": wazuh_alerts.as_posix(),
        "ground_truth_sha256": sha256_file(ground_truth),
        "lab_features_sha256": sha256_file(lab_features),
        "wazuh_alerts_sha256": sha256_file(wazuh_alerts),
        "result_files": RESULT_FILES,
        "result_file_sha256": hashes,
        "warning": warnings,
        "notes": [
            "This provenance file does not certify correctness; it records input/output identity for traceability.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    create_provenance(
        ground_truth=Path(args.ground_truth),
        lab_features=Path(args.lab_features),
        wazuh_alerts=Path(args.wazuh_alerts),
        output=Path(args.output),
    )
    print(f"[OK] Provenance: {args.output}")


if __name__ == "__main__":
    main()
