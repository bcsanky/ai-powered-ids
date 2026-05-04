from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


AUXILIARY_COLUMNS = {
    "label",
    "is_benign",
    "source_file",
    "split",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/final/ae_minimal")
    parser.add_argument("--output-dir", default="data/processed/final/rule_proxy")
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    return parser.parse_args()


def load_split(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó bemeneti fájl: {path}")
    return pd.read_parquet(path)


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    columns = [
        col
        for col in df.columns
        if col not in AUXILIARY_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not columns:
        raise ValueError("Nem található numerikus feature oszlop a szabályproxyhoz.")
    return columns


def validate_feature_columns(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    train_cols = select_feature_columns(train_df)
    val_cols = select_feature_columns(val_df)
    test_cols = select_feature_columns(test_df)
    if train_cols != val_cols or train_cols != test_cols:
        raise ValueError(
            "A train, val és test feature oszlopai nem egyeznek. "
            f"train={train_cols}, val={val_cols}, test={test_cols}"
        )
    return train_cols


def compute_rule_score(df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    values = df[feature_columns].to_numpy(dtype=float)
    return np.nanmax(np.abs(values), axis=1)


def build_rule_proxy_export(
    test_df: pd.DataFrame,
    test_scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    alerts = (test_scores >= threshold).astype(int)
    high_alerts = test_scores >= (1.5 * threshold)
    rule_level = np.where(alerts == 0, 0, np.where(high_alerts, 10, 5)).astype(int)

    if "label" in test_df.columns:
        label = test_df["label"].values
    else:
        label = np.where(test_df["is_benign"].astype(int).to_numpy() == 1, "BENIGN", "ATTACK")

    out = pd.DataFrame(
        {
            "row_id": np.arange(len(test_df), dtype=int),
            "label": label,
            "is_benign": test_df["is_benign"].astype(int).to_numpy(),
            "y_pred": alerts,
            "prediction": alerts,
            "wazuh_alert": alerts,
            "is_alert": alerts,
            "rule_level": rule_level,
            "alert_score": test_scores,
            "score": test_scores,
            "rule_threshold": float(threshold),
        }
    )
    if "source_file" in test_df.columns:
        out["source_file"] = test_df["source_file"].values
    return out


def create_rule_proxy_export(
    data_dir: Path,
    output_dir: Path,
    threshold_quantile: float,
) -> dict:
    if not 0.0 <= threshold_quantile <= 1.0:
        raise ValueError("--threshold-quantile értéke 0 és 1 közé essen.")

    train_df = load_split(data_dir, "train")
    val_df = load_split(data_dir, "val")
    test_df = load_split(data_dir, "test")

    feature_columns = validate_feature_columns(train_df, val_df, test_df)
    val_scores = compute_rule_score(val_df, feature_columns)
    test_scores = compute_rule_score(test_df, feature_columns)
    threshold = float(np.quantile(val_scores, threshold_quantile))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "wazuh_like_rule_eval.csv"
    export_df = build_rule_proxy_export(test_df, test_scores, threshold)
    export_df.to_csv(output_file, index=False)

    metadata = {
        "method": "max_abs_standardized_feature_rule",
        "threshold_quantile": float(threshold_quantile),
        "threshold_value": threshold,
        "fitted_on": "validation split",
        "evaluated_on": "test split",
        "input_data_dir": str(data_dir),
        "output_file": str(output_file),
        "feature_columns": feature_columns,
        "rows_val": int(len(val_df)),
        "rows_test": int(len(test_df)),
        "note": "Ez kontrollált szabályalapú proxy baseline, nem natív Wazuh export.",
    }
    with (output_dir / "rule_proxy_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def main() -> None:
    args = parse_args()
    metadata = create_rule_proxy_export(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        threshold_quantile=float(args.threshold_quantile),
    )
    print(f"[OK] Szabályproxy export mentve ide: {metadata['output_file']}")


if __name__ == "__main__":
    main()
