from __future__ import annotations

import argparse
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from schema import canonicalize_columns


BENIGN_LABELS = {
    "benign",
    "normal",
}


def read_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_csvs(raw_dir: Path) -> pd.DataFrame:
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nincs CSV a könyvtárban: {raw_dir}")

    frames = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df["source_file"] = csv_file.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    mapping = canonicalize_columns(df.columns.tolist())

    renamed = {}
    for canonical, original in mapping.items():
        renamed[original] = canonical

    df = df.rename(columns=renamed)

    if "label" not in df.columns:
        raise ValueError("Nem található label oszlop.")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["is_benign"] = df["label"].isin(BENIGN_LABELS).astype(int)

    return df


def clean_numeric_columns(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    for col in numeric_cols:
        if col not in df.columns:
            raise ValueError(f"Hiányzó numerikus oszlop: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=numeric_cols)

    return df


def fill_categorical(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str)
    return df

def validate_ratio_sum(name: str, values: list[float], expected: float = 1.0, tol: float = 1e-9) -> None:
    total = sum(values)
    if abs(total - expected) > tol:
        raise ValueError(
            f"Hibás {name} arányok: összegük {total}, de {expected} kellene legyen."
        )
    
def allocate_counts(
    total: int,
    ratios: list[float],
    min_counts: list[int],
    split_name: str,
) -> list[int]:
    if len(ratios) != len(min_counts):
        raise ValueError(
            f"{split_name}: a ratios és min_counts hossza nem egyezik."
        )

    validate_ratio_sum(split_name, ratios)

    min_total = sum(min_counts)
    if total < min_total:
        raise ValueError(
            f"Nincs elég minta a(z) {split_name} felosztáshoz: "
            f"összesen {total}, de minimum {min_total} kellene."
        )

    remaining = total - min_total

    raw = [r * remaining for r in ratios]
    extra = [int(np.floor(x)) for x in raw]

    leftover = remaining - sum(extra)
    order = sorted(
        range(len(ratios)),
        key=lambda i: raw[i] - extra[i],
        reverse=True,
    )

    for i in order[:leftover]:
        extra[i] += 1

    counts = [m + e for m, e in zip(min_counts, extra)]

    diff = total - sum(counts)
    if diff != 0:
        counts[0] += diff

    return counts   

def split_by_counts(df: pd.DataFrame, counts: list[int], seed: int) -> list[pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    parts = []
    start = 0
    for count in counts:
        end = start + count
        parts.append(shuffled.iloc[start:end].reset_index(drop=True))
        start = end

    return parts

def split_for_autoencoder(df: pd.DataFrame, split_cfg: dict, seed: int):
    benign = df[df["is_benign"] == 1].copy()
    attack = df[df["is_benign"] == 0].copy()

    if benign.empty:
        raise ValueError("Nincs benign minta, AE tanításhoz ez kötelező.")
    if attack.empty:
        raise ValueError("Nincs attack minta, calib/test felosztáshoz ez kötelező.")

    benign_train_ratio = float(split_cfg["benign_train_ratio"])
    benign_val_ratio = float(split_cfg["benign_val_ratio"])
    benign_calib_ratio = float(split_cfg["benign_calib_ratio"])
    benign_test_ratio = float(split_cfg["benign_test_ratio"])

    attack_calib_ratio = float(split_cfg["attack_calib_ratio"])
    attack_test_ratio = float(split_cfg["attack_test_ratio"])

    benign_counts = allocate_counts(
        total=len(benign),
        ratios=[
            benign_train_ratio,
            benign_val_ratio,
            benign_calib_ratio,
            benign_test_ratio,
        ],
        min_counts=[1, 1, 1, 1],
        split_name="benign split",
    )

    attack_counts = allocate_counts(
        total=len(attack),
        ratios=[
            attack_calib_ratio,
            attack_test_ratio,
        ],
        min_counts=[1, 1],
        split_name="attack split",
    )

    benign_train, benign_val, benign_calib, benign_test = split_by_counts(
        benign, benign_counts, seed
    )
    attack_calib, attack_test = split_by_counts(
        attack, attack_counts, seed
    )

    calib_df = pd.concat([benign_calib, attack_calib], ignore_index=True)
    calib_df = calib_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    test_df = pd.concat([benign_test, attack_test], ignore_index=True)
    test_df = test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return (
        benign_train.reset_index(drop=True),
        benign_val.reset_index(drop=True),
        calib_df,
        test_df,
    )

def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    transformers = []

    if numeric_cols:
        transformers.append(
            ("num", StandardScaler(), numeric_cols)
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transform_to_dataframe(preprocessor: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    transformed = preprocessor.transform(df)
    feature_names = preprocessor.get_feature_names_out()

    out = pd.DataFrame(transformed, columns=feature_names, index=df.index)
    out["label"] = df["label"].values
    out["is_benign"] = df["is_benign"].values
    out["source_file"] = df["source_file"].values if "source_file" in df.columns else ""

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = read_config(args.config)

    raw_dir = Path(cfg["dataset"]["raw_dir"])
    output_dir = Path(cfg["dataset"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols = cfg["features"]["numeric"]
    categorical_cols = cfg["features"]["categorical"]
    seed = cfg["random_seed"]

    print("Nyers CSV-k beolvasása...")
    df = load_all_csvs(raw_dir)

    print("Oszlopok normalizálása...")
    df = standardize_dataframe(df)

    print("Tisztítás...")
    df = clean_numeric_columns(df, numeric_cols)
    df = fill_categorical(df, categorical_cols)

    used_cols = numeric_cols + categorical_cols + ["label", "is_benign", "source_file"]
    df = df[used_cols].copy()

    print("Split készítése...")
    split_cfg = cfg["split"]
    train_df, val_df, calib_df, test_df = split_for_autoencoder(df, split_cfg, seed)

    print("Preprocess fit csak train-en...")
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(train_df)

    print("Transform...")
    train_out = transform_to_dataframe(preprocessor, train_df)
    val_out = transform_to_dataframe(preprocessor, val_df)
    calib_out = transform_to_dataframe(preprocessor, calib_df)
    test_out = transform_to_dataframe(preprocessor, test_df)

    train_path = output_dir / cfg["output"]["train_file"]
    val_path = output_dir / cfg["output"]["val_file"]
    calib_path = output_dir / cfg["output"]["calib_file"]
    test_path = output_dir / cfg["output"]["test_file"]
    experiment_id = cfg["experiment_id"]
    preprocess_path = output_dir / cfg["output"]["preprocess_file"]
    versioned_preprocess_path = output_dir / f"preprocess_{experiment_id}.pkl"

    print("Parquet mentés...")
    train_out.to_parquet(train_path, index=False)
    val_out.to_parquet(val_path, index=False)
    calib_out.to_parquet(calib_path, index=False)
    test_out.to_parquet(test_path, index=False)

    print("Preprocess objektum mentése...")
    joblib.dump(preprocessor, preprocess_path)
    joblib.dump(preprocessor, versioned_preprocess_path)

    metadata = {
        "experiment_id": experiment_id,
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "rows_calib": int(len(calib_df)),
        "rows_test": int(len(test_df)),
        "rows_calib_attacks": int((calib_df["is_benign"] == 0).sum()),
        "rows_calib_benign": int((calib_df["is_benign"] == 1).sum()),
        "rows_test_attacks": int((test_df["is_benign"] == 0).sum()),
        "rows_test_benign": int((test_df["is_benign"] == 1).sum()),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "seed": seed,
        "preprocess_file": str(preprocess_path),
        "versioned_preprocess_file": str(versioned_preprocess_path),
    }

    with open(output_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Kész.")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
