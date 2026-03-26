from __future__ import annotations

from pathlib import Path


import joblib
import pandas as pd
import json

PROCESSED_DIR = Path("data/processed")


def test_processed_files_exist():
    assert (PROCESSED_DIR / "train.parquet").exists()
    assert (PROCESSED_DIR / "val.parquet").exists()
    assert (PROCESSED_DIR / "test.parquet").exists()
    assert (PROCESSED_DIR / "preprocess.pkl").exists()
    assert (PROCESSED_DIR / "dataset_metadata.json").exists()


def test_train_val_test_can_be_loaded():
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0


def test_train_and_val_are_benign_only():
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")

    assert set(train["is_benign"].unique()) == {1}
    assert set(val["is_benign"].unique()) == {1}


def test_test_contains_both_classes():
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    classes = set(test["is_benign"].unique())
    assert 0 in classes
    assert 1 in classes


def test_preprocessor_can_be_loaded():
    pp = joblib.load(PROCESSED_DIR / "preprocess.pkl")
    assert pp is not None
    assert type(pp).__name__ == "ColumnTransformer"

PROCESSED_DIR = Path("data/processed")

def test_versioned_preprocess_file_exists():
    metadata_path = PROCESSED_DIR / "dataset_metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    versioned_file = Path(metadata["versioned_preprocess_file"])
    assert versioned_file.exists()


def test_metadata_matches_real_row_counts():
    with open(PROCESSED_DIR / "dataset_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    assert metadata["rows_train"] == len(train)
    assert metadata["rows_val"] == len(val)
    assert metadata["rows_test"] == len(test)
    assert metadata["rows_test_attacks"] == int((test["is_benign"] == 0).sum())
    assert metadata["rows_test_benign"] == int((test["is_benign"] == 1).sum())


def test_train_val_test_have_same_feature_columns():
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    assert list(train.columns) == list(val.columns) == list(test.columns)
