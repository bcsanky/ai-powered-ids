from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

LABEL_CANDIDATES = ["label", "labels", "target", "y", "is_attack", "attack", "class"]
TIMESTAMP_CANDIDATES = ["timestamp", "ts", "event_time", "flow_start", "flow_start_time", "date"]


@dataclass
class EvalArtifacts:
    train_path: Path
    val_path: Path
    test_path: Path
    preprocess_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="stat", choices=["stat", "wazuh"])
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--history-json", default=None)
    return parser.parse_args()


def resolve_artifacts(data_dir: Path) -> EvalArtifacts:
    return EvalArtifacts(
        train_path=data_dir / "train.parquet",
        val_path=data_dir / "val.parquet",
        test_path=data_dir / "test.parquet",
        preprocess_path=data_dir / "preprocess.pkl",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó fájl: {path}")
    return pd.read_parquet(path)


def infer_label(df: pd.DataFrame) -> tuple[str, np.ndarray]:
    for col in LABEL_CANDIDATES:
        if col in df.columns:
            s = df[col]
            if pd.api.types.is_bool_dtype(s):
                y = s.astype(int).to_numpy()
            elif pd.api.types.is_numeric_dtype(s):
                y = s.fillna(0).astype(int).to_numpy()
            else:
                upper = s.astype(str).str.strip().str.upper()
                benign_tokens = {"BENIGN", "NORMAL", "FALSE", "0"}
                y = (~upper.isin(benign_tokens)).astype(int).to_numpy()
            return col, y
    raise ValueError(
        f"Nem található label oszlop. Próbáltak: {LABEL_CANDIDATES}. "
        f"Elérhető oszlopok: {list(df.columns)}"
    )


def infer_timestamp_col(df: pd.DataFrame) -> Optional[str]:
    for col in TIMESTAMP_CANDIDATES:
        if col in df.columns:
            return col
    return None


def to_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


AUXILIARY_NON_FEATURE_COLUMNS = {
    "is_benign",
    "source_file",
    "split",
}

def prepare_feature_matrix(df: pd.DataFrame, preprocess, timestamp_col: Optional[str]):
    raw_feature_cols = list(getattr(preprocess, "feature_names_in_", []))

    # 1) Nyers adat eset: minden, a preprocess által elvárt oszlop jelen van
    if raw_feature_cols and all(col in df.columns for col in raw_feature_cols):
        x_df = df[raw_feature_cols]
        x = to_dense(preprocess.transform(x_df))
        return x, raw_feature_cols, "raw_with_preprocess"

    # 2) Már preprocesselt adat eset: ne transformáljuk újra
    excluded = set(LABEL_CANDIDATES) | AUXILIARY_NON_FEATURE_COLUMNS
    if timestamp_col:
        excluded.add(timestamp_col)

    feature_cols = [c for c in df.columns if c not in excluded]

    if not feature_cols:
        raise ValueError("Nem maradt felhasználható feature oszlop az eval futtatásához.")

    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(
            "Az eval fallback ágában csak numerikus, már preprocesselt feature-ök támogatottak. "
            f"Nem numerikus oszlopok: {non_numeric}"
        )

    x = df[feature_cols].to_numpy(dtype=np.float32)
    return x, feature_cols, "already_preprocessed"


def compute_scores(x: np.ndarray, center: np.ndarray) -> np.ndarray:
    diff = x - center
    return np.sqrt(np.sum(diff * diff, axis=1))


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    metrics = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_samples": int(len(y_true)),
        "n_attack": int(np.sum(y_true == 1)),
        "n_benign": int(np.sum(y_true == 0)),
    }

    if len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def compute_time_metrics(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, timestamp_col: Optional[str]) -> dict:
    if not timestamp_col or timestamp_col not in df.columns:
        return {
            "has_timestamp": False,
            "first_tp_time": None,
            "first_fp_time": None,
            "tp_per_min": None,
            "fp_per_min": None,
        }

    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    tmp = pd.DataFrame(
        {
            "timestamp": ts,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    ).dropna(subset=["timestamp"]).sort_values("timestamp")

    if tmp.empty:
        return {
            "has_timestamp": False,
            "first_tp_time": None,
            "first_fp_time": None,
            "tp_per_min": None,
            "fp_per_min": None,
        }

    tp_mask = (tmp["y_true"] == 1) & (tmp["y_pred"] == 1)
    fp_mask = (tmp["y_true"] == 0) & (tmp["y_pred"] == 1)

    duration_min = max((tmp["timestamp"].max() - tmp["timestamp"].min()).total_seconds() / 60.0, 1e-9)

    first_tp = tmp.loc[tp_mask, "timestamp"].min() if tp_mask.any() else pd.NaT
    first_fp = tmp.loc[fp_mask, "timestamp"].min() if fp_mask.any() else pd.NaT

    return {
        "has_timestamp": True,
        "first_tp_time": None if pd.isna(first_tp) else first_tp.isoformat(),
        "first_fp_time": None if pd.isna(first_fp) else first_fp.isoformat(),
        "tp_per_min": float(tp_mask.sum() / duration_min),
        "fp_per_min": float(fp_mask.sum() / duration_min),
    }


def save_confusion_matrix_png(cm: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Attack"])
    ax.set_yticklabels(["Benign", "Attack"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_roc_png(y_true: np.ndarray, scores: np.ndarray, out_path: Path) -> None:
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_score_distribution_png(scores: np.ndarray, y_true: np.ndarray, threshold: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores[y_true == 0], bins=40, alpha=0.7, label="Benign")
    ax.hist(scores[y_true == 1], bins=40, alpha=0.7, label="Attack")
    ax.axvline(threshold, linestyle="--", label=f"threshold={threshold:.4f}")
    ax.set_title("Score Distribution")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_threshold_curve(scores: np.ndarray, y_true: np.ndarray, out_csv: Path, out_png: Path) -> None:
    thresholds = np.linspace(float(scores.min()), float(scores.max()), 100)
    rows = []

    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        rows.append(
            {
                "threshold": float(thr),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["threshold"], df["precision"], label="precision")
    ax.plot(df["threshold"], df["recall"], label="recall")
    ax.plot(df["threshold"], df["f1"], label="f1")
    ax.set_title("Threshold Sweep")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_loss_curve(history_json: Optional[str], out_path: Path) -> None:
    if not history_json:
        return

    history_path = Path(history_json)
    if not history_path.exists():
        return

    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)

    train_loss = history.get("loss") or history.get("train_loss")
    val_loss = history.get("val_loss")

    if not train_loss:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_loss, label="train_loss")
    if val_loss:
        ax.plot(val_loss, label="val_loss")
    ax.set_title("Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_stat_baseline(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    results_root = Path(args.results_dir)
    artifacts = resolve_artifacts(data_dir)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / f"baseline_{args.baseline}_{run_id}"
    ensure_dir(run_dir)

    train_df = load_df(artifacts.train_path)
    val_df = load_df(artifacts.val_path)
    test_df = load_df(artifacts.test_path)
    preprocess = joblib.load(artifacts.preprocess_path)

    train_label_col, y_train = infer_label(train_df)
    val_label_col, y_val = infer_label(val_df)
    test_label_col, y_test = infer_label(test_df)

    train_ts_col = infer_timestamp_col(train_df)
    val_ts_col = infer_timestamp_col(val_df)
    test_ts_col = infer_timestamp_col(test_df)

    x_train_t, train_feature_cols, feature_mode = prepare_feature_matrix(
        train_df, preprocess, train_ts_col
    )
    x_val_t, val_feature_cols, _ = prepare_feature_matrix(
        val_df, preprocess, val_ts_col
    )
    x_test_t, test_feature_cols, _ = prepare_feature_matrix(
        test_df, preprocess, test_ts_col
    )

    if train_feature_cols != val_feature_cols or train_feature_cols != test_feature_cols:
        raise ValueError(
            "A train/val/test feature oszlopai nem egyeznek. "
            f"train={train_feature_cols}, val={val_feature_cols}, test={test_feature_cols}"
        )

    print(f"[INFO] Feature mode: {feature_mode}")
    print(f"[INFO] X_train shape: {x_train_t.shape}")
    print(f"[INFO] X_val shape: {x_val_t.shape}")
    print(f"[INFO] X_test shape: {x_test_t.shape}")

    center = np.mean(x_train_t, axis=0)
    val_scores = compute_scores(x_val_t, center)
    threshold = float(np.quantile(val_scores, args.threshold_quantile))

    test_scores = compute_scores(x_test_t, center)
    y_pred = (test_scores >= threshold).astype(int)

    metrics = compute_classification_metrics(y_test, y_pred, test_scores)
    metrics["baseline"] = args.baseline
    metrics["threshold"] = threshold
    metrics["threshold_quantile"] = args.threshold_quantile

    time_metrics = compute_time_metrics(test_df, y_test, y_pred, test_ts_col)
    metrics.update(time_metrics)

    pd.DataFrame([metrics]).to_csv(run_dir / "metrics_summary.csv", index=False)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["true_benign", "true_attack"],
        columns=["pred_benign", "pred_attack"],
    ).to_csv(run_dir / "confusion_matrix.csv")
    save_confusion_matrix_png(cm, run_dir / "confusion_matrix.png")

    save_score_distribution_png(test_scores, y_test, threshold, run_dir / "score_distribution.png")
    save_threshold_curve(
        test_scores,
        y_test,
        run_dir / "threshold_curve.csv",
        run_dir / "threshold_curve.png",
    )
    save_roc_png(y_test, test_scores, run_dir / "roc_curve.png")
    save_loss_curve(args.history_json, run_dir / "loss_curve.png")

    pred_df = pd.DataFrame(
        {
            "y_true": y_test,
            "score": test_scores,
            "y_pred": y_pred,
        }
    )
    if test_ts_col and test_ts_col in test_df.columns:
        pred_df.insert(0, "timestamp", test_df[test_ts_col].astype(str).values)

    pred_df.to_csv(run_dir / "predictions.csv", index=False)

    metadata = {
        "baseline": args.baseline,
        "run_dir": str(run_dir),
        "data_dir": str(data_dir),
        "preprocess_path": str(artifacts.preprocess_path),
    }
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Eredmények mentve ide: {run_dir}")


def run_wazuh_baseline(_: argparse.Namespace) -> None:
    raise NotImplementedError(
        "A Wazuh baseline-hez előbb normalizált lab export kell "
        "(pl. alerts + ground truth timestamp mezőkkel). "
        "Ez külön adapterrel köthető be."
    )


def main() -> None:
    args = parse_args()
    if args.baseline == "stat":
        run_stat_baseline(args)
    elif args.baseline == "wazuh":
        run_wazuh_baseline(args)
    else:
        raise ValueError(f"Ismeretlen baseline: {args.baseline}")


if __name__ == "__main__":
    main()
