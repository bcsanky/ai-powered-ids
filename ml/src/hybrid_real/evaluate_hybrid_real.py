from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.src.lab_ae_eval.evaluate_ae_lab import compute_binary_metrics


STRATEGIES = ["hybrid_or", "hybrid_weighted", "hybrid_priority"]
PREDICTION_COLUMNS = [
    "event_id",
    "label",
    "y_true",
    "wazuh_pred",
    "ae_pred",
    "anomaly_score",
    "max_rule_level",
    "time_to_detection_sec",
    "hybrid_or_pred",
    "hybrid_weighted_score",
    "hybrid_weighted_pred",
    "hybrid_priority_level",
    "hybrid_priority_pred",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wazuh-predictions", default="results/wazuh_real/predictions.csv")
    parser.add_argument("--ae-predictions", default="results/ae_lab/predictions.csv")
    parser.add_argument("--results-dir", default="results/hybrid_real")
    parser.add_argument("--weighted-threshold", type=float, default=0.5)
    return parser.parse_args()


def minmax(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if max_value <= min_value:
        return pd.Series(np.zeros(len(numeric), dtype=float), index=numeric.index)
    return (numeric - min_value) / (max_value - min_value)


def numeric_column(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def priority_level(wazuh_pred: int, ae_pred: int) -> str:
    if wazuh_pred == 1 and ae_pred == 1:
        return "critical"
    if wazuh_pred == 1 and ae_pred == 0:
        return "high"
    if wazuh_pred == 0 and ae_pred == 1:
        return "medium"
    return "normal"


def load_and_join_predictions(wazuh_path: Path, ae_path: Path) -> pd.DataFrame:
    if not wazuh_path.exists():
        raise FileNotFoundError(f"Hiányzó Wazuh predikciós CSV: {wazuh_path}")
    if not ae_path.exists():
        raise FileNotFoundError(f"Hiányzó AE lab predikciós CSV: {ae_path}")

    wazuh = pd.read_csv(wazuh_path)
    ae = pd.read_csv(ae_path)
    for name, df, required in [
        ("Wazuh", wazuh, {"event_id", "label", "y_true", "wazuh_pred"}),
        ("AE", ae, {"event_id", "label", "y_true", "ae_pred", "anomaly_score"}),
    ]:
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Hiányzó {name} predikciós oszlopok: {', '.join(missing)}")

    wazuh_ids = set(wazuh["event_id"].astype(str))
    ae_ids = set(ae["event_id"].astype(str))
    if wazuh_ids != ae_ids:
        raise ValueError("A Wazuh és AE predikciók event_id készlete nem egyezik.")

    merged = wazuh.merge(
        ae[["event_id", "y_true", "ae_pred", "anomaly_score"]],
        on="event_id",
        how="inner",
        suffixes=("_wazuh", "_ae"),
        validate="one_to_one",
    )
    if not (merged["y_true_wazuh"].astype(int) == merged["y_true_ae"].astype(int)).all():
        raise ValueError("A Wazuh és AE y_true értékek nem egyeznek event_id alapján.")

    merged["label"] = merged["label"].astype(str).str.strip().str.lower()
    merged["y_true"] = merged["y_true_wazuh"].astype(int)
    merged["wazuh_pred"] = pd.to_numeric(merged["wazuh_pred"], errors="coerce").fillna(0).astype(int)
    merged["ae_pred"] = pd.to_numeric(merged["ae_pred"], errors="coerce").fillna(0).astype(int)
    merged["max_rule_level"] = numeric_column(merged, "max_rule_level", 0.0)
    merged["time_to_detection_sec"] = numeric_column(merged, "time_to_detection_sec", np.nan)
    merged["anomaly_score"] = pd.to_numeric(merged["anomaly_score"], errors="coerce").fillna(0.0)
    return merged.sort_values("event_id", kind="mergesort").reset_index(drop=True)


def build_hybrid_predictions(
    wazuh_df: pd.DataFrame,
    ae_df: pd.DataFrame,
    *,
    weighted_threshold: float = 0.5,
) -> pd.DataFrame:
    joined = load_and_join_dataframes(wazuh_df, ae_df)
    return add_hybrid_columns(joined, weighted_threshold=weighted_threshold)


def load_and_join_dataframes(wazuh: pd.DataFrame, ae: pd.DataFrame) -> pd.DataFrame:
    temp_wazuh = wazuh.copy()
    temp_ae = ae.copy()
    wazuh_ids = set(temp_wazuh["event_id"].astype(str))
    ae_ids = set(temp_ae["event_id"].astype(str))
    if wazuh_ids != ae_ids:
        raise ValueError("A Wazuh és AE predikciók event_id készlete nem egyezik.")
    merged = temp_wazuh.merge(
        temp_ae[["event_id", "y_true", "ae_pred", "anomaly_score"]],
        on="event_id",
        how="inner",
        suffixes=("_wazuh", "_ae"),
        validate="one_to_one",
    )
    if not (merged["y_true_wazuh"].astype(int) == merged["y_true_ae"].astype(int)).all():
        raise ValueError("A Wazuh és AE y_true értékek nem egyeznek event_id alapján.")
    merged["label"] = merged["label"].astype(str).str.strip().str.lower()
    merged["y_true"] = merged["y_true_wazuh"].astype(int)
    merged["wazuh_pred"] = pd.to_numeric(merged["wazuh_pred"], errors="coerce").fillna(0).astype(int)
    merged["ae_pred"] = pd.to_numeric(merged["ae_pred"], errors="coerce").fillna(0).astype(int)
    merged["max_rule_level"] = numeric_column(merged, "max_rule_level", 0.0)
    merged["time_to_detection_sec"] = numeric_column(merged, "time_to_detection_sec", np.nan)
    merged["anomaly_score"] = pd.to_numeric(merged["anomaly_score"], errors="coerce").fillna(0.0)
    return merged.sort_values("event_id", kind="mergesort").reset_index(drop=True)


def add_hybrid_columns(joined: pd.DataFrame, *, weighted_threshold: float) -> pd.DataFrame:
    out = joined.copy()
    normalized_ae_score = minmax(out["anomaly_score"])
    normalized_wazuh_level = (pd.to_numeric(out["max_rule_level"], errors="coerce").fillna(0.0) / 15.0).clip(0.0, 1.0)
    out["hybrid_or_pred"] = ((out["wazuh_pred"] == 1) | (out["ae_pred"] == 1)).astype(int)
    out["hybrid_weighted_score"] = 0.6 * normalized_ae_score + 0.4 * normalized_wazuh_level
    out["hybrid_weighted_pred"] = (out["hybrid_weighted_score"] >= weighted_threshold).astype(int)
    out["hybrid_priority_level"] = [
        priority_level(wazuh_pred, ae_pred)
        for wazuh_pred, ae_pred in zip(out["wazuh_pred"], out["ae_pred"])
    ]
    out["hybrid_priority_pred"] = (out["hybrid_priority_level"] != "normal").astype(int)
    return out[PREDICTION_COLUMNS]


def compute_strategy_metrics(df: pd.DataFrame, strategy: str, prediction_column: str) -> dict[str, Any]:
    metrics = compute_binary_metrics(df["y_true"], df[prediction_column])
    detected = df[
        (df["y_true"].astype(int) == 1)
        & (df[prediction_column].astype(int) == 1)
    ]
    ttd = pd.to_numeric(detected["time_to_detection_sec"], errors="coerce").dropna()
    metrics["strategy"] = strategy
    metrics["mean_ttd"] = float(ttd.mean()) if not ttd.empty else np.nan
    metrics["median_ttd"] = float(ttd.median()) if not ttd.empty else np.nan
    return metrics


def evaluate_hybrid_real(
    *,
    wazuh_predictions_path: Path,
    ae_predictions_path: Path,
    results_dir: Path,
    weighted_threshold: float = 0.5,
) -> dict[str, Path]:
    joined = load_and_join_predictions(wazuh_predictions_path, ae_predictions_path)
    predictions = add_hybrid_columns(joined, weighted_threshold=weighted_threshold)
    metrics_rows = [
        compute_strategy_metrics(predictions, "hybrid_or", "hybrid_or_pred"),
        compute_strategy_metrics(predictions, "hybrid_weighted", "hybrid_weighted_pred"),
        compute_strategy_metrics(predictions, "hybrid_priority", "hybrid_priority_pred"),
    ]

    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics_summary.csv"
    predictions_path = results_dir / "predictions.csv"
    confusion_path = results_dir / "confusion_matrix_by_strategy.csv"
    metadata_path = results_dir / "run_metadata.json"

    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    confusion_rows = []
    for row in metrics_rows:
        strategy = row["strategy"]
        confusion_rows.extend(
            [
                {
                    "strategy": strategy,
                    "actual": "benign",
                    "predicted_benign": row["TN"],
                    "predicted_attack": row["FP"],
                },
                {
                    "strategy": strategy,
                    "actual": "attack",
                    "predicted_benign": row["FN"],
                    "predicted_attack": row["TP"],
                },
            ]
        )
    pd.DataFrame(confusion_rows).to_csv(confusion_path, index=False)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "wazuh_predictions": str(wazuh_predictions_path),
        "ae_predictions": str(ae_predictions_path),
        "results_dir": str(results_dir),
        "weighted_threshold": weighted_threshold,
        "strategies": STRATEGIES,
        "note": "Natív Wazuh és AE-Minimal lab predikciók kontrollált hibrid kiértékelése.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "confusion_matrix_by_strategy": confusion_path,
        "metadata": metadata_path,
    }


def main() -> None:
    args = parse_args()
    outputs = evaluate_hybrid_real(
        wazuh_predictions_path=Path(args.wazuh_predictions),
        ae_predictions_path=Path(args.ae_predictions),
        results_dir=Path(args.results_dir),
        weighted_threshold=args.weighted_threshold,
    )
    for path in outputs.values():
        print(f"[OK] Kimenet: {path}")


if __name__ == "__main__":
    main()
