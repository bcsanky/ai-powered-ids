from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = ["event_id", "label", "y_true", "ae_pred"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--results-dir", default="results/ae_lab")
    return parser.parse_args()


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def compute_binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = safe_rate(tp, tp + fp)
    recall = safe_rate(tp, tp + fn)
    f1 = safe_rate(2 * precision * recall, precision + recall)
    benign_total = tn + fp
    attack_total = tp + fn
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
        "false_positive_rate": safe_rate(fp, benign_total),
        "false_negative_rate": safe_rate(fn, attack_total),
        "true_positive_rate": safe_rate(tp, attack_total),
        "true_negative_rate": safe_rate(tn, benign_total),
        "alert_count": int(tp + fp),
        "n_samples": int(len(y_true)),
        "n_attack": int((y_true == 1).sum()),
        "n_benign": int((y_true == 0).sum()),
    }


def evaluate_ae_lab(input_path: Path, results_dir: Path) -> dict[str, Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Hiányzó AE lab predikciós CSV: {input_path}")
    df = pd.read_csv(input_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Hiányzó kötelező AE lab predikciós oszlopok: {', '.join(missing)}")

    results_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_binary_metrics(df["y_true"], df["ae_pred"])

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
        "note": "AE-Minimal modell offline pontozása címkézett lab eseményeken.",
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
    outputs = evaluate_ae_lab(Path(args.input), Path(args.results_dir))
    for path in outputs.values():
        print(f"[OK] Kimenet: {path}")


if __name__ == "__main__":
    main()
