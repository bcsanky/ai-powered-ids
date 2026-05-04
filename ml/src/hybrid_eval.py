from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve

from ml.src.eval import compute_classification_metrics


REQUIRED_METRIC_COLUMNS = [
    "tn",
    "fp",
    "fn",
    "tp",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "false_negative_rate",
    "true_positive_rate",
    "true_negative_rate",
    "alert_count",
    "roc_auc",
    "n_samples",
    "n_attack",
    "n_benign",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-run-dir", default=None)
    parser.add_argument("--rule-run-dir", default=None)
    parser.add_argument("--ae-root", default="results/final/final-ae-minimal-v1")
    parser.add_argument("--rule-root", default="results/final/final-rule-proxy-v1")
    parser.add_argument("--output-dir", default="results/final/final-hybrid-v1")
    return parser.parse_args()


def find_latest_run(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Hiányzó futtatási gyökérkönyvtár: {root}")

    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "predictions.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"Nem található predictions.csv fájlt tartalmazó futtatási könyvtár itt: {root}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def select_ae_prediction_column(df: pd.DataFrame) -> str:
    for col in ["pred_f1_optimum", "pred_percentile_95"]:
        if col in df.columns:
            return col
    pred_cols = [col for col in df.columns if col.startswith("pred_")]
    if not pred_cols:
        raise ValueError("Az AE predictions.csv nem tartalmaz predikciós oszlopot.")
    return sorted(pred_cols)[0]


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    min_value = float(np.nanmin(values))
    max_value = float(np.nanmax(values))
    if max_value <= min_value:
        return np.zeros_like(values, dtype=float)
    return (values - min_value) / (max_value - min_value)


def validate_inputs(ae_df: pd.DataFrame, rule_df: pd.DataFrame) -> None:
    if len(ae_df) != len(rule_df):
        raise ValueError(
            "Az AE és rule proxy predikciók hossza eltér: "
            f"ae={len(ae_df)}, rule={len(rule_df)}"
        )
    if "y_true" not in ae_df.columns or "y_true" not in rule_df.columns:
        raise ValueError("Mindkét predictions.csv állományban szükséges a y_true oszlop.")
    ae_y = ae_df["y_true"].astype(int).to_numpy()
    rule_y = rule_df["y_true"].astype(int).to_numpy()
    if not np.array_equal(ae_y, rule_y):
        raise ValueError("Az AE és rule proxy y_true értékei nem egyeznek sorindex alapján.")


def build_hybrid_predictions(ae_df: pd.DataFrame, rule_df: pd.DataFrame) -> pd.DataFrame:
    validate_inputs(ae_df, rule_df)
    ae_pred_col = select_ae_prediction_column(ae_df)

    ae_score = pd.to_numeric(ae_df["score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    rule_score = pd.to_numeric(rule_df["score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ae_pred = pd.to_numeric(ae_df[ae_pred_col], errors="coerce").fillna(0).astype(int).to_numpy()
    rule_pred = pd.to_numeric(rule_df["y_pred"], errors="coerce").fillna(0).astype(int).to_numpy()

    ae_score_norm = minmax_normalize(ae_score)
    rule_score_norm = minmax_normalize(rule_score)
    hybrid_score = np.maximum(ae_score_norm, rule_score_norm)
    hybrid_pred = ((ae_pred == 1) | (rule_pred == 1)).astype(int)

    return pd.DataFrame(
        {
            "row_id": np.arange(len(ae_df), dtype=int),
            "y_true": ae_df["y_true"].astype(int).to_numpy(),
            "ae_score": ae_score,
            "ae_pred": ae_pred,
            "rule_score": rule_score,
            "rule_pred": rule_pred,
            "hybrid_score": hybrid_score,
            "hybrid_pred": hybrid_pred,
        }
    )


def save_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Konfúziós mátrix")
    ax.set_xlabel("Prediktált osztály")
    ax.set_ylabel("Valós osztály")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Támadás"])
    ax.set_yticklabels(["Benign", "Támadás"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_score_distribution(df: pd.DataFrame, path: Path) -> None:
    y_true = df["y_true"].to_numpy()
    scores = df["hybrid_score"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores[y_true == 0], bins=40, alpha=0.7, label="Benign")
    ax.hist(scores[y_true == 1], bins=40, alpha=0.7, label="Támadás")
    ax.set_title("Anomáliapontszámok eloszlása")
    ax.set_xlabel("Anomáliapontszám")
    ax.set_ylabel("Darabszám")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_roc(df: pd.DataFrame, path: Path) -> None:
    y_true = df["y_true"].to_numpy(dtype=int)
    if len(np.unique(y_true)) < 2:
        return
    scores = df["hybrid_score"].to_numpy(dtype=float)
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC-görbe")
    ax.set_xlabel("Hamis pozitív arány")
    ax.set_ylabel("Valódi pozitív arány")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: make_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_serializable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_hybrid_eval(ae_run_dir: Path, rule_run_dir: Path, output_root: Path) -> Path:
    ae_predictions_path = ae_run_dir / "predictions.csv"
    rule_predictions_path = rule_run_dir / "predictions.csv"
    if not ae_predictions_path.exists():
        raise FileNotFoundError(f"Hiányzó AE predictions.csv: {ae_predictions_path}")
    if not rule_predictions_path.exists():
        raise FileNotFoundError(f"Hiányzó rule proxy predictions.csv: {rule_predictions_path}")

    ae_df = pd.read_csv(ae_predictions_path)
    rule_df = pd.read_csv(rule_predictions_path)
    hybrid_df = build_hybrid_predictions(ae_df, rule_df)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"hybrid_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    y_true = hybrid_df["y_true"].to_numpy(dtype=int)
    y_pred = hybrid_df["hybrid_pred"].to_numpy(dtype=int)
    scores = hybrid_df["hybrid_score"].to_numpy(dtype=float)

    metrics = compute_classification_metrics(y_true, y_pred, scores)
    metrics["threshold_name"] = "hybrid_union"
    metrics["threshold_value"] = None
    pd.DataFrame([metrics], columns=["threshold_name", "threshold_value", *REQUIRED_METRIC_COLUMNS]).to_csv(
        run_dir / "metrics_summary.csv",
        index=False,
    )

    hybrid_df.to_csv(run_dir / "predictions.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["true_benign", "true_attack"],
        columns=["pred_benign", "pred_attack"],
    ).to_csv(run_dir / "confusion_matrix.csv")
    save_confusion_matrix(cm, run_dir / "confusion_matrix.png")
    save_score_distribution(hybrid_df, run_dir / "score_distribution.png")
    save_roc(hybrid_df, run_dir / "roc_curve.png")

    metadata = {
        "run_dir": str(run_dir),
        "ae_run_dir": str(ae_run_dir),
        "rule_run_dir": str(rule_run_dir),
        "decision": "hybrid_pred = ae_pred OR rule_pred",
        "score": "hybrid_score = max(minmax(ae_score), minmax(rule_score))",
        "limitation": (
            "Az offline hibrid kiértékelés azonos teszthalmaz-sorrendre épül, "
            "nem éles eseménykorreláció."
        ),
    }
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(make_serializable(metadata), f, indent=2, ensure_ascii=False)

    return run_dir


def main() -> None:
    args = parse_args()
    ae_run_dir = Path(args.ae_run_dir) if args.ae_run_dir else find_latest_run(Path(args.ae_root))
    rule_run_dir = Path(args.rule_run_dir) if args.rule_run_dir else find_latest_run(Path(args.rule_root))
    run_dir = run_hybrid_eval(
        ae_run_dir=ae_run_dir,
        rule_run_dir=rule_run_dir,
        output_root=Path(args.output_dir),
    )
    print(f"[OK] Hibrid eredmények mentve ide: {run_dir}")


if __name__ == "__main__":
    main()
