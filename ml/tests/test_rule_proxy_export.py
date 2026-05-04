from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ml.src.create_rule_proxy_export import create_rule_proxy_export


def write_split(path, rows):
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_rule_proxy_export_uses_validation_threshold_and_writes_schema(tmp_path):
    data_dir = tmp_path / "processed"
    output_dir = tmp_path / "rule_proxy"
    data_dir.mkdir()

    write_split(
        data_dir / "train.parquet",
        [
            {"feature_a": 0.1, "feature_b": -0.2, "label": "BENIGN", "is_benign": 1},
            {"feature_a": 0.2, "feature_b": -0.3, "label": "BENIGN", "is_benign": 1},
        ],
    )
    write_split(
        data_dir / "val.parquet",
        [
            {"feature_a": 1.0, "feature_b": 0.0, "label": "BENIGN", "is_benign": 1},
            {"feature_a": 2.0, "feature_b": 0.0, "label": "BENIGN", "is_benign": 1},
            {"feature_a": 3.0, "feature_b": 0.0, "label": "BENIGN", "is_benign": 1},
            {"feature_a": 4.0, "feature_b": 0.0, "label": "BENIGN", "is_benign": 1},
        ],
    )
    write_split(
        data_dir / "test.parquet",
        [
            {"feature_a": 0.5, "feature_b": 0.0, "label": "BENIGN", "is_benign": 1},
            {"feature_a": 4.0, "feature_b": 0.0, "label": "DoS", "is_benign": 0},
            {"feature_a": 7.0, "feature_b": 0.0, "label": "PortScan", "is_benign": 0},
        ],
    )

    metadata = create_rule_proxy_export(data_dir, output_dir, threshold_quantile=0.5)

    expected_threshold = float(np.quantile(np.array([1.0, 2.0, 3.0, 4.0]), 0.5))
    assert metadata["threshold_value"] == expected_threshold
    assert metadata["fitted_on"] == "validation split"
    assert metadata["evaluated_on"] == "test split"

    metadata_path = output_dir / "rule_proxy_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        saved_metadata = json.load(f)
    assert saved_metadata["threshold_value"] == expected_threshold

    out = pd.read_csv(output_dir / "wazuh_like_rule_eval.csv")
    required_columns = {
        "row_id",
        "label",
        "is_benign",
        "y_pred",
        "prediction",
        "wazuh_alert",
        "is_alert",
        "rule_level",
        "alert_score",
        "score",
        "rule_threshold",
    }
    assert required_columns.issubset(out.columns)
    assert set(out["rule_level"]).issubset({0, 5, 10})
    assert set(out["y_pred"]).issubset({0, 1})
    assert out["rule_threshold"].nunique() == 1
    assert out["rule_threshold"].iloc[0] == expected_threshold
