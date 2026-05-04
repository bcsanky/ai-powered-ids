from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml

TEST_CONFIG = Path("experiments/experiment_test.yaml")


@pytest.fixture(scope="session")
def processed_dir(tmp_path_factory):
    work_dir = tmp_path_factory.mktemp("dataset_outputs")
    output_dir = work_dir / "processed"
    config_path = work_dir / "experiment_test_tmp.yaml"

    with open(TEST_CONFIG, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["dataset"]["output_dir"] = str(output_dir)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    subprocess.run(
        [sys.executable, "ml/src/build_dataset.py", "--config", str(config_path)],
        check=True,
    )

    return output_dir


def test_processed_files_exist(processed_dir):
    assert (processed_dir / "train.parquet").exists()
    assert (processed_dir / "val.parquet").exists()
    assert (processed_dir / "calib.parquet").exists()
    assert (processed_dir / "test.parquet").exists()
    assert (processed_dir / "preprocess.pkl").exists()
    assert (processed_dir / "dataset_metadata.json").exists()


def test_train_val_calib_test_can_be_loaded(processed_dir):
    train = pd.read_parquet(processed_dir / "train.parquet")
    val = pd.read_parquet(processed_dir / "val.parquet")
    calib = pd.read_parquet(processed_dir / "calib.parquet")
    test = pd.read_parquet(processed_dir / "test.parquet")

    assert len(train) > 0
    assert len(val) > 0
    assert len(calib) > 0
    assert len(test) > 0


def test_train_and_val_are_benign_only(processed_dir):
    train = pd.read_parquet(processed_dir / "train.parquet")
    val = pd.read_parquet(processed_dir / "val.parquet")

    assert set(train["is_benign"].unique()) == {1}
    assert set(val["is_benign"].unique()) == {1}


def test_calib_contains_both_classes(processed_dir):
    calib = pd.read_parquet(processed_dir / "calib.parquet")

    classes = set(calib["is_benign"].unique())
    assert 0 in classes
    assert 1 in classes


def test_test_contains_both_classes(processed_dir):
    test = pd.read_parquet(processed_dir / "test.parquet")

    classes = set(test["is_benign"].unique())
    assert 0 in classes
    assert 1 in classes


def test_preprocessor_can_be_loaded(processed_dir):
    pp = joblib.load(processed_dir / "preprocess.pkl")
    assert pp is not None
    assert type(pp).__name__ == "ColumnTransformer"


def test_versioned_preprocess_file_exists(processed_dir):
    metadata_path = processed_dir / "dataset_metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    versioned_file = Path(metadata["versioned_preprocess_file"])
    assert versioned_file.exists()


def test_metadata_matches_real_row_counts(processed_dir):
    with open(processed_dir / "dataset_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    train = pd.read_parquet(processed_dir / "train.parquet")
    val = pd.read_parquet(processed_dir / "val.parquet")
    calib = pd.read_parquet(processed_dir / "calib.parquet")
    test = pd.read_parquet(processed_dir / "test.parquet")

    assert metadata["rows_train"] == len(train)
    assert metadata["rows_val"] == len(val)
    assert metadata["rows_calib"] == len(calib)
    assert metadata["rows_test"] == len(test)

    assert metadata["rows_calib_attacks"] == int((calib["is_benign"] == 0).sum())
    assert metadata["rows_calib_benign"] == int((calib["is_benign"] == 1).sum())
    assert metadata["rows_test_attacks"] == int((test["is_benign"] == 0).sum())
    assert metadata["rows_test_benign"] == int((test["is_benign"] == 1).sum())

    assert metadata["context_enabled"] is False
    assert metadata["context_features"] == []
    assert metadata["context_fit_split"] is None
    assert metadata["unknown_context_frequency"] == 0.0
    assert metadata["dev_sample"]["enabled"] is False
    assert metadata["dev_sample"]["used"] is False
    assert metadata["dev_sample"]["max_rows_total"] is None
    assert metadata["dev_sample"]["rows_before"] == metadata["dev_sample"]["rows_after"]


def test_train_val_calib_test_have_same_feature_columns(processed_dir):
    train = pd.read_parquet(processed_dir / "train.parquet")
    val = pd.read_parquet(processed_dir / "val.parquet")
    calib = pd.read_parquet(processed_dir / "calib.parquet")
    test = pd.read_parquet(processed_dir / "test.parquet")

    assert list(train.columns) == list(val.columns) == list(calib.columns) == list(test.columns)
