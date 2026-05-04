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

CONTEXT_FEATURES = [
    "destination_port_frequency",
    "protocol_frequency",
    "is_rare_destination_port",
    "packet_ratio",
    "bytes_packets_ratio",
]


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


def dev_sample_metadata(
    enabled: bool,
    max_rows_total: int | None,
    rows_before: int,
    rows_after: int,
    class_counts_before: dict[int, int],
    class_counts_after: dict[int, int],
) -> dict:
    return {
        "enabled": enabled,
        "max_rows_total": max_rows_total,
        "used": enabled and rows_after < rows_before,
        "sample_stage": "after_cleaning_before_split",
        "class_column": "is_benign",
        "preserve_classes_if_possible": True,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "class_counts_before": {
            str(k): int(v) for k, v in sorted(class_counts_before.items())
        },
        "class_counts_after": {
            str(k): int(v) for k, v in sorted(class_counts_after.items())
        },
    }


def apply_dev_sample(
    df: pd.DataFrame,
    dev_sample_cfg: dict | None,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    cfg = dev_sample_cfg or {}
    enabled = bool(cfg.get("enabled", False))
    raw_max_rows = cfg.get("max_rows_total")
    max_rows_total = int(raw_max_rows) if raw_max_rows is not None else None

    rows_before = int(len(df))
    class_counts_before = df["is_benign"].value_counts().to_dict()

    if not enabled or max_rows_total is None or max_rows_total >= rows_before:
        metadata = dev_sample_metadata(
            enabled=enabled,
            max_rows_total=max_rows_total,
            rows_before=rows_before,
            rows_after=rows_before,
            class_counts_before=class_counts_before,
            class_counts_after=class_counts_before,
        )
        return df, metadata

    if max_rows_total <= 0:
        raise ValueError("A dev_sample.max_rows_total pozitív egész kell legyen.")

    class_values = sorted(df["is_benign"].unique().tolist())
    if len(class_values) > max_rows_total:
        raise ValueError(
            "A dev_sample.max_rows_total túl kicsi ahhoz, hogy minden osztályból "
            "legalább egy minta megmaradjon."
        )

    sampled_parts = []
    remaining = max_rows_total
    remaining_classes = len(class_values)

    for class_value in class_values:
        class_df = df[df["is_benign"] == class_value]
        class_count = len(class_df)
        proportional_count = int(round(max_rows_total * class_count / rows_before))
        requested = max(1, proportional_count)
        requested = min(requested, class_count)
        requested = min(requested, remaining - (remaining_classes - 1))

        sampled_parts.append(
            class_df.sample(n=requested, random_state=seed + int(class_value))
        )
        remaining -= requested
        remaining_classes -= 1

    sampled_df = pd.concat(sampled_parts, ignore_index=True)

    if len(sampled_df) < max_rows_total:
        extra_needed = max_rows_total - len(sampled_df)
        rest = df.drop(index=pd.concat(sampled_parts).index)
        if not rest.empty:
            extra = rest.sample(
                n=min(extra_needed, len(rest)),
                random_state=seed + 17,
            )
            sampled_df = pd.concat([sampled_df, extra], ignore_index=True)

    sampled_df = sampled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    class_counts_after = sampled_df["is_benign"].value_counts().to_dict()

    metadata = dev_sample_metadata(
        enabled=enabled,
        max_rows_total=max_rows_total,
        rows_before=rows_before,
        rows_after=int(len(sampled_df)),
        class_counts_before=class_counts_before,
        class_counts_after=class_counts_after,
    )

    return sampled_df, metadata


def validate_context_columns(df: pd.DataFrame) -> None:
    required = {
        "destination_port",
        "protocol",
        "total_fwd_packets",
        "total_backward_packets",
        "flow_bytes_per_sec",
        "flow_packets_per_sec",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Context feature készítéshez hiányzó oszlopok: "
            + ", ".join(missing)
        )


def fit_context_statistics(
    train_df: pd.DataFrame,
    rare_destination_port_threshold: float,
) -> dict:
    validate_context_columns(train_df)

    if rare_destination_port_threshold < 0:
        raise ValueError("A rare_destination_port_threshold nem lehet negatív.")

    return {
        "destination_port_frequency": train_df["destination_port"]
        .value_counts(normalize=True)
        .to_dict(),
        "protocol_frequency": train_df["protocol"]
        .value_counts(normalize=True)
        .to_dict(),
        "rare_destination_port_threshold": rare_destination_port_threshold,
    }


def apply_context_features(
    df: pd.DataFrame,
    context_statistics: dict,
    bytes_packets_epsilon: float,
    unknown_frequency: float = 0.0,
) -> pd.DataFrame:
    validate_context_columns(df)

    if bytes_packets_epsilon <= 0:
        raise ValueError("A bytes_packets_epsilon pozitív kell legyen.")

    out = df.copy()

    destination_port_frequency = out["destination_port"].map(
        context_statistics["destination_port_frequency"]
    )
    protocol_frequency = out["protocol"].map(
        context_statistics["protocol_frequency"]
    )

    backward_packets = np.maximum(
        out["total_backward_packets"].to_numpy(dtype=np.float64),
        1.0,
    )
    packets_per_sec = np.maximum(
        out["flow_packets_per_sec"].to_numpy(dtype=np.float64),
        bytes_packets_epsilon,
    )

    out["destination_port_frequency"] = (
        destination_port_frequency.fillna(unknown_frequency).astype(float)
    )
    out["protocol_frequency"] = (
        protocol_frequency.fillna(unknown_frequency).astype(float)
    )
    out["is_rare_destination_port"] = (
        out["destination_port_frequency"]
        < float(context_statistics["rare_destination_port_threshold"])
    ).astype(int)
    out["packet_ratio"] = (
        out["total_fwd_packets"].to_numpy(dtype=np.float64) / backward_packets
    )
    out["bytes_packets_ratio"] = (
        out["flow_bytes_per_sec"].to_numpy(dtype=np.float64) / packets_per_sec
    )

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=CONTEXT_FEATURES)

    return out

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

    features_cfg = cfg["features"]
    numeric_cols = list(features_cfg["numeric"])
    categorical_cols = list(features_cfg["categorical"])
    context_enabled = bool(features_cfg.get("context_enabled", False))
    context_cfg = features_cfg.get("context", {})
    context_features = list(CONTEXT_FEATURES) if context_enabled else []
    context_fit_split = "train" if context_enabled else None
    unknown_context_frequency = 0.0
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

    print("Development mintavétel ellenőrzése...")
    dev_sample_cfg = cfg.get("dev_sample", cfg["dataset"].get("dev_sample", {}))
    df, dev_sample_info = apply_dev_sample(df, dev_sample_cfg, seed)

    print("Split készítése...")
    split_cfg = cfg["split"]
    train_df, val_df, calib_df, test_df = split_for_autoencoder(df, split_cfg, seed)

    if context_enabled:
        print("Context feature statisztikák illesztése train split-en...")
        context_statistics = fit_context_statistics(
            train_df,
            rare_destination_port_threshold=float(
                context_cfg.get("rare_destination_port_threshold", 0.001)
            ),
        )

        print("Context feature-ök alkalmazása...")
        apply_kwargs = {
            "context_statistics": context_statistics,
            "bytes_packets_epsilon": float(
                context_cfg.get("bytes_packets_epsilon", 1e-9)
            ),
            "unknown_frequency": unknown_context_frequency,
        }
        train_df = apply_context_features(train_df, **apply_kwargs)
        val_df = apply_context_features(val_df, **apply_kwargs)
        calib_df = apply_context_features(calib_df, **apply_kwargs)
        test_df = apply_context_features(test_df, **apply_kwargs)
        numeric_cols = numeric_cols + context_features

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
        "context_enabled": context_enabled,
        "context_features": context_features,
        "context_fit_split": context_fit_split,
        "unknown_context_frequency": unknown_context_frequency,
        "dev_sample": dev_sample_info,
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
