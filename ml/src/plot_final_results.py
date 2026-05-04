from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc


PREFERRED_PREDICTION_COLUMNS = [
    "pred_f1_optimum",
    "pred_percentile_95",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def ensure_predictions(run_dir: Path) -> pd.DataFrame:
    predictions_path = run_dir / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Hiányzó kötelező fájl: {predictions_path}")
    return pd.read_csv(predictions_path)


def select_prediction_column(df: pd.DataFrame) -> str:
    for column in PREFERRED_PREDICTION_COLUMNS:
        if column in df.columns:
            return column

    prediction_columns = sorted(c for c in df.columns if c.startswith("pred_"))
    if prediction_columns:
        return prediction_columns[0]

    raise ValueError(
        "Nem található predikciós oszlop. Elvárt: pred_f1_optimum, "
        "pred_percentile_95 vagy legalább egy pred_* oszlop."
    )


def require_columns(df: pd.DataFrame, columns: list[str], source_name: str) -> bool:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        warn(f"{source_name}: hiányzó oszlopok: {', '.join(missing)}")
        return False
    return True


def save_confusion_matrix(df: pd.DataFrame, pred_col: str, run_dir: Path) -> None:
    if not require_columns(df, ["y_true", pred_col], "predictions.csv"):
        return

    y_true = df["y_true"].astype(int).to_numpy()
    y_pred = df[pred_col].astype(int).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Konfúziós mátrix")
    ax.set_xlabel("Prediktált osztály")
    ax.set_ylabel("Valós osztály")
    ax.set_xticks([0, 1], labels=["Benign", "Támadás"])
    ax.set_yticks([0, 1], labels=["Benign", "Támadás"])

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def save_score_distribution(df: pd.DataFrame, run_dir: Path) -> None:
    if not require_columns(df, ["score", "y_true"], "predictions.csv"):
        return

    benign_scores = df.loc[df["y_true"].astype(int) == 0, "score"].astype(float)
    attack_scores = df.loc[df["y_true"].astype(int) == 1, "score"].astype(float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if not benign_scores.empty:
        ax.hist(benign_scores, bins=80, alpha=0.65, label="Benign", density=True)
    if not attack_scores.empty:
        ax.hist(attack_scores, bins=80, alpha=0.65, label="Támadás", density=True)

    ax.set_title("Anomáliapontszámok eloszlása")
    ax.set_xlabel("Anomáliapontszám")
    ax.set_ylabel("Sűrűség")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "score_distribution.png", dpi=160)
    plt.close(fig)


def save_threshold_curve(run_dir: Path) -> None:
    curve_path = run_dir / "threshold_curve.csv"
    if not curve_path.exists():
        warn(f"Opcionális fájl hiányzik: {curve_path}")
        return

    curve = pd.read_csv(curve_path)
    required = ["threshold", "precision", "recall", "f1"]
    if not require_columns(curve, required, "threshold_curve.csv"):
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(curve["threshold"], curve["precision"], label="Precision")
    ax.plot(curve["threshold"], curve["recall"], label="Recall")
    ax.plot(curve["threshold"], curve["f1"], label="F1")
    ax.set_title("Küszöbérzékenységi görbe")
    ax.set_xlabel("Küszöbérték")
    ax.set_ylabel("Metrika értéke")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "threshold_curve.png", dpi=160)
    plt.close(fig)


def save_roc_curve(df: pd.DataFrame, run_dir: Path) -> None:
    if not require_columns(df, ["score", "y_true"], "predictions.csv"):
        return

    y_true = df["y_true"].astype(int).to_numpy()
    if len(np.unique(y_true)) < 2:
        warn("ROC-görbe nem készül: a y_true nem tartalmaz mindkét osztályt.")
        return

    scores = df["score"].astype(float).to_numpy()
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Véletlen")
    ax.set_title("ROC-görbe")
    ax.set_xlabel("Hamis pozitív arány")
    ax.set_ylabel("Valódi pozitív arány")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(run_dir / "roc_curve.png", dpi=160)
    plt.close(fig)


def load_top_feature_source(predictions: pd.DataFrame, run_dir: Path) -> pd.Series | None:
    if "top1_feature" in predictions.columns:
        return predictions["top1_feature"]

    top_errors_path = run_dir / "top_feature_errors.csv"
    if not top_errors_path.exists():
        warn("Top feature ábra nem készül: nincs top1_feature oszlop vagy top_feature_errors.csv.")
        return None

    top_errors = pd.read_csv(top_errors_path, usecols=lambda c: c == "top1_feature")
    if "top1_feature" not in top_errors.columns:
        warn("top_feature_errors.csv nem tartalmaz top1_feature oszlopot.")
        return None
    return top_errors["top1_feature"]


def save_top_feature_frequency(predictions: pd.DataFrame, run_dir: Path) -> None:
    top_features = load_top_feature_source(predictions, run_dir)
    if top_features is None:
        return

    counts = top_features.dropna().astype(str)
    counts = counts[counts.str.len() > 0].value_counts().head(15)
    if counts.empty:
        warn("Top feature ábra nem készül: nincs ábrázolható top feature érték.")
        return

    fig_height = max(4.2, 0.35 * len(counts) + 1.5)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    y_pos = np.arange(len(counts))
    ax.barh(y_pos, counts.values)
    ax.set_yticks(y_pos, labels=counts.index)
    ax.invert_yaxis()
    ax.set_title("Leggyakoribb magyarázó feature-ök")
    ax.set_xlabel("Darabszám")
    fig.tight_layout()
    fig.savefig(run_dir / "top_feature_frequency.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        raise NotADirectoryError(f"Nem létező run könyvtár: {run_dir}")

    predictions = ensure_predictions(run_dir)
    pred_col = select_prediction_column(predictions)
    print(f"[INFO] Predikciós oszlop: {pred_col}")

    save_confusion_matrix(predictions, pred_col, run_dir)
    save_score_distribution(predictions, run_dir)
    save_threshold_curve(run_dir)
    save_roc_curve(predictions, run_dir)
    save_top_feature_frequency(predictions, run_dir)

    print(f"[OK] Ábrák mentve ide: {run_dir}")


if __name__ == "__main__":
    main()
