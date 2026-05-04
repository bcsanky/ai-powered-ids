from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--results-dir", default="results/wazuh_real")
    return parser.parse_args()


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def compute_wazuh_metrics(df: pd.DataFrame) -> dict[str, Any]:
    y_true = df["y_true"].astype(int).to_numpy()
    y_pred = df["wazuh_pred"].astype(int).to_numpy()

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = safe_rate(tp, tp + fp)
    recall = safe_rate(tp, tp + fn)
    f1 = safe_rate(2 * precision * recall, precision + recall)
    false_positive_rate = safe_rate(fp, fp + tn)
    false_negative_rate = safe_rate(fn, fn + tp)

    ttd = pd.to_numeric(df.loc[(df["y_true"] == 1) & (df["wazuh_pred"] == 1), "time_to_detection_sec"], errors="coerce")
    ttd = ttd.dropna()
    mean_ttd = float(ttd.mean()) if not ttd.empty else np.nan
    median_ttd = float(ttd.median()) if not ttd.empty else np.nan

    if "alert_count" in df.columns:
        raw_alert_count = int(pd.to_numeric(df["alert_count"], errors="coerce").fillna(0).sum())
    else:
        raw_alert_count = int(tp + fp)

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "alert_count": int(tp + fp),
        "raw_alert_count": raw_alert_count,
        "mean_ttd": mean_ttd,
        "median_ttd": median_ttd,
        "n_samples": int(len(df)),
        "n_attack": int((y_true == 1).sum()),
        "n_benign": int((y_true == 0).sum()),
    }


def save_outputs(input_path: Path, results_dir: Path) -> dict[str, Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Hiányzó korrelált Wazuh predikciós CSV: {input_path}")
    df = pd.read_csv(input_path)
    required = {"event_id", "label", "y_true", "wazuh_pred"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Hiányzó kötelező predikciós oszlopok: {', '.join(missing)}")

    results_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_wazuh_metrics(df)

    metrics_path = results_dir / "metrics_summary.csv"
    predictions_path = results_dir / "predictions.csv"
    confusion_path = results_dir / "confusion_matrix.csv"
    metadata_path = results_dir / "run_metadata.json"

    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    df.to_csv(predictions_path, index=False)
    pd.DataFrame(
        [
            {"actual": "benign", "predicted_benign": metrics["TN"], "predicted_attack": metrics["FP"]},
            {"actual": "attack", "predicted_benign": metrics["FN"], "predicted_attack": metrics["TP"]},
        ]
    ).to_csv(confusion_path, index=False)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(input_path),
        "results_dir": str(results_dir),
        "n_samples": metrics["n_samples"],
        "note": "Natív Wazuh alert export és lab ground truth korrelációján alapuló Wazuh-only baseline.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "confusion_matrix": confusion_path,
        "metadata": metadata_path,
    }


def main() -> None:
    args = parse_args()
    outputs = save_outputs(Path(args.input), Path(args.results_dir))
    for path in outputs.values():
        print(f"[OK] Kimenet: {path}")


if __name__ == "__main__":
    main()
