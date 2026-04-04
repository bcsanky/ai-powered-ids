from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from ml.src.autoencoder import (
    AEConfig,
    build_autoencoder,
    fit_autoencoder,
    reconstruct,
    sample_reconstruction_scores,
    feature_reconstruction_scores,
)
from ml.src.explain import (
    aggregate_feature_errors,
    top_k_group_names,
)
from ml.src.thresholds import (
    best_f1_threshold,
    fixed_threshold,
    percentile_threshold,
    threshold_sweep_dataframe,
)


AUXILIARY_COLUMNS = {
    "label",
    "is_benign",
    "source_file",
    "split",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def read_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó fájl: {path}")
    return pd.read_parquet(path)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in AUXILIARY_COLUMNS]


def to_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return df[feature_cols].to_numpy(dtype=np.float32)


def attack_labels_from_is_benign(df: pd.DataFrame) -> np.ndarray:
    return (1 - df["is_benign"].astype(int)).to_numpy()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
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
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    else:
        metrics["roc_auc"] = None

    return metrics


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_serializable(obj), f, indent=2)


def main() -> None:
    args = parse_args()
    cfg = read_config(args.config)

    output_dir = Path(cfg["dataset"]["output_dir"])
    out_cfg = cfg["output"]
    paths_cfg = cfg.get("paths", {})

    train_df = load_df(output_dir / out_cfg["train_file"])
    val_df = load_df(output_dir / out_cfg["val_file"])
    calib_df = load_df(output_dir / out_cfg["calib_file"])
    test_df = load_df(output_dir / out_cfg["test_file"])

    feature_cols = get_feature_columns(train_df)

    if feature_cols != get_feature_columns(val_df):
        raise ValueError("A train és val feature oszlopai nem egyeznek.")
    if feature_cols != get_feature_columns(calib_df):
        raise ValueError("A train és calib feature oszlopai nem egyeznek.")
    if feature_cols != get_feature_columns(test_df):
        raise ValueError("A train és test feature oszlopai nem egyeznek.")

    x_train = to_matrix(train_df, feature_cols)
    x_val = to_matrix(val_df, feature_cols)
    x_calib = to_matrix(calib_df, feature_cols)
    x_test = to_matrix(test_df, feature_cols)

    y_calib = attack_labels_from_is_benign(calib_df)
    y_test = attack_labels_from_is_benign(test_df)

    model_cfg = cfg["model"]
    effective_batch_size = min(int(model_cfg["batch_size"]), max(1, len(train_df)))
    effective_early_stopping = bool(model_cfg["early_stopping"])
    effective_validation_fraction = float(model_cfg["validation_fraction"])

    if effective_early_stopping and len(train_df) * effective_validation_fraction < 2:
        effective_early_stopping = False
        
    ae_config = AEConfig(
        hidden_layer_sizes=tuple(model_cfg["hidden_layer_sizes"]),
        activation=model_cfg["activation"],
        solver=model_cfg["solver"],
        alpha=float(model_cfg["alpha"]),
        batch_size=effective_batch_size,
        learning_rate_init=float(model_cfg["learning_rate_init"]),
        max_iter=int(model_cfg["max_iter"]),
        tol=float(model_cfg["tol"]),
        n_iter_no_change=int(model_cfg["n_iter_no_change"]),
        early_stopping=effective_early_stopping,
        validation_fraction=effective_validation_fraction,
        random_state=int(cfg["random_seed"]),
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = Path(paths_cfg.get("artifacts_dir", "artifacts/ae")) / f"ae_v1_{run_id}"
    result_dir = Path(paths_cfg.get("results_dir", "results")) / f"ae_v1_{run_id}"

    ensure_dir(artifact_dir)
    ensure_dir(result_dir)

    print("[INFO] Autoencoder építése...")
    model = build_autoencoder(ae_config)

    print("[INFO] Tanítás indul...")
    model = fit_autoencoder(model, x_train)

    print("[INFO] Rekonstrukciók számítása...")
    val_recon = reconstruct(model, x_val)
    calib_recon = reconstruct(model, x_calib)
    test_recon = reconstruct(model, x_test)

    val_scores = sample_reconstruction_scores(x_val, val_recon)
    calib_scores = sample_reconstruction_scores(x_calib, calib_recon)
    test_scores = sample_reconstruction_scores(x_test, test_recon)
    test_feature_errors = feature_reconstruction_scores(x_test, test_recon)
    grouped_test_errors = aggregate_feature_errors(test_feature_errors, feature_cols)
    top_feature_names = top_k_group_names(grouped_test_errors, k=5)

    thr_cfg = cfg["thresholds"]
    thr_fixed = fixed_threshold(float(thr_cfg["fixed"]))
    thr_percentile = percentile_threshold(
        val_scores,
        percentile=float(thr_cfg["percentile"]),
    )
    thr_f1 = best_f1_threshold(
        y_calib,
        calib_scores,
        n_steps=int(thr_cfg.get("search_steps", 200)),
        name="f1_optimum",
    )

    thresholds = {
        "fixed": float(thr_fixed),
        f"percentile_{int(thr_cfg['percentile'])}": float(thr_percentile),
        "f1_optimum": float(thr_f1.threshold),
    }

    rows = []
    prediction_columns = {}

    for name, thr in thresholds.items():
        y_pred = (test_scores >= thr).astype(int)
        prediction_columns[f"pred_{name}"] = y_pred

        metrics = compute_metrics(y_test, y_pred, test_scores)
        metrics["threshold_name"] = name
        metrics["threshold_value"] = float(thr)
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(result_dir / "metrics_summary.csv", index=False)

    predictions_df = pd.DataFrame(
        {
            "score": test_scores,
            "y_true": y_test,
            "source_file": test_df["source_file"].values if "source_file" in test_df.columns else "",
        }
    )

    predictions_df["top1_feature"] = [names[0] if len(names) > 0 else "" for names in top_feature_names]
    predictions_df["top2_feature"] = [names[1] if len(names) > 1 else "" for names in top_feature_names]
    predictions_df["top3_feature"] = [names[2] if len(names) > 2 else "" for names in top_feature_names]
    predictions_df["top4_feature"] = [names[3] if len(names) > 3 else "" for names in top_feature_names]
    predictions_df["top5_feature"] = [names[4] if len(names) > 4 else "" for names in top_feature_names]

    for col_name, values in prediction_columns.items():
        predictions_df[col_name] = values
        predictions_df.to_csv(result_dir / "predictions.csv", index=False)
        top_feature_summary_df = grouped_test_errors.copy()
        top_feature_summary_df["y_true"] = y_test
        top_feature_summary_df["score"] = test_scores
        top_feature_summary_df["top1_feature"] = [names[0] if len(names) > 0 else "" for names in top_feature_names]
        top_feature_summary_df["top2_feature"] = [names[1] if len(names) > 1 else "" for names in top_feature_names]
        top_feature_summary_df["top3_feature"] = [names[2] if len(names) > 2 else "" for names in top_feature_names]
        top_feature_summary_df.to_csv(result_dir / "top_feature_errors.csv", index=False)

    
    sweep_df = threshold_sweep_dataframe(
        y_calib,
        calib_scores,
        n_steps=int(thr_cfg.get("search_steps", 200)),
    )
    sweep_df.to_csv(result_dir / "threshold_curve.csv", index=False)

    best_validation_score = getattr(model, "best_validation_score_", None)
    if best_validation_score is not None:
        best_validation_score = float(best_validation_score)

    history = {
        "loss": [float(x) for x in getattr(model, "loss_curve_", [])],
        "n_iter": int(getattr(model, "n_iter_", 0)),
        "best_validation_score": best_validation_score,
        "out_activation": getattr(model, "out_activation_", None),
    }

    threshold_metadata = {
        "fixed": float(thr_fixed),
        f"percentile_{int(thr_cfg['percentile'])}": float(thr_percentile),
        "f1_optimum": {
            "threshold": float(thr_f1.threshold),
            "precision": float(thr_f1.precision),
            "recall": float(thr_f1.recall),
            "f1": float(thr_f1.f1),
        },
    }

    run_metadata = {
        "run_id": run_id,
        "experiment_id": cfg["experiment_id"],
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "rows_calib": int(len(calib_df)),
        "rows_test": int(len(test_df)),
        "artifact_dir": str(artifact_dir),
        "result_dir": str(result_dir),
    }

    print("[INFO] Model mentése...")
    joblib.dump(model, artifact_dir / "model.joblib")

    save_json(cfg, artifact_dir / "train_config.json")
    save_json(history, artifact_dir / "history.json")
    save_json(threshold_metadata, artifact_dir / "thresholds.json")
    save_json(run_metadata, result_dir / "run_metadata.json")

    print(f"[OK] AE modell mentve ide: {artifact_dir}")
    print(f"[OK] Eredmények mentve ide: {result_dir}")


if __name__ == "__main__":
    main()
