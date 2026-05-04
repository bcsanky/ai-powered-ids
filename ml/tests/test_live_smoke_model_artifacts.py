from __future__ import annotations

import json

import joblib
import pandas as pd

from ml.src.live_smoke.check_model_artifacts import run_check


def test_model_artifact_hash_calculation_with_tmp_files(tmp_path):
    model_root = tmp_path / "artifacts/final/final-ae-minimal-v1"
    run_dir = model_root / "ae_v1_001"
    run_dir.mkdir(parents=True)
    joblib.dump({"model": "ok"}, run_dir / "model.joblib")
    (run_dir / "thresholds.json").write_text(json.dumps({"fixed": 0.5}), encoding="utf-8")
    preprocess = tmp_path / "data/processed/final/ae_minimal/preprocess.pkl"
    preprocess.parent.mkdir(parents=True)
    joblib.dump({"preprocess": "ok"}, preprocess)

    result = run_check(tmp_path / "reports/live_smoke", model_root, preprocess)
    summary = pd.read_csv(result["csv"])
    payload = json.loads(result["json"].read_text(encoding="utf-8"))

    assert "FAIL" not in set(summary["status"])
    assert payload["hashes"]["model_joblib_sha256"]
    assert payload["hashes"]["preprocess_sha256"]
    assert payload["hashes"]["thresholds_sha256"]

