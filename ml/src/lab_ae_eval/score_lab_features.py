from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.lab_ae_eval.validate_lab_features import validate_lab_features
from ml.src.scoring_runtime import AEScorer, REQUIRED_FEATURES, result_to_dict
from ml.src.wazuh_baseline.build_ground_truth import validate_ground_truth


OUTPUT_COLUMNS = [
    "event_id",
    "label",
    "y_true",
    "ae_pred",
    "ml_alert",
    "anomaly_score",
    "threshold_name",
    "threshold_value",
    "risk_level",
    "reason",
    "scenario",
    "source_ip",
    "target_ip",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", "--input", dest="features", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-root", default="artifacts/final/final-ae-minimal-v1")
    parser.add_argument("--preprocess", default="data/processed/final/ae_minimal/preprocess.pkl")
    parser.add_argument("--model-version", default="final-ae-minimal-v1")
    return parser.parse_args()


def ensure_same_event_ids(features: pd.DataFrame, ground_truth: pd.DataFrame) -> None:
    feature_ids = set(features["event_id"].astype(str))
    truth_ids = set(ground_truth["event_id"].astype(str))
    missing_truth = sorted(feature_ids - truth_ids)
    missing_features = sorted(truth_ids - feature_ids)
    if missing_truth or missing_features:
        parts = []
        if missing_truth:
            parts.append(f"ground truth nélkül: {missing_truth}")
        if missing_features:
            parts.append(f"feature nélkül: {missing_features}")
        raise ValueError("A lab feature és ground truth event_id készlete eltér: " + "; ".join(parts))


def merge_features_with_ground_truth(features_path: Path, ground_truth_path: Path) -> pd.DataFrame:
    features = validate_lab_features(features_path)
    ground_truth = validate_ground_truth(ground_truth_path)
    ensure_same_event_ids(features, ground_truth)

    truth_cols = ["event_id", "label", "scenario", "source_ip", "target_ip"]
    merged = features.merge(
        ground_truth[truth_cols],
        on="event_id",
        how="inner",
        suffixes=("", "_truth"),
        validate="one_to_one",
    )
    for col in ["scenario", "source_ip", "target_ip"]:
        truth_col = f"{col}_truth"
        if col in features.columns:
            merged[col] = merged[col].fillna("").astype(str)
            merged[col] = merged[col].mask(merged[col].str.strip().eq(""), merged[truth_col])
            merged = merged.drop(columns=[truth_col])
        else:
            merged = merged.rename(columns={truth_col: col})

    merged["label"] = merged["label"].astype(str).str.strip().str.lower()
    merged["y_true"] = (merged["label"] == "attack").astype(int)
    return merged


def row_features(row: pd.Series) -> dict[str, Any]:
    return {feature: row[feature] for feature in REQUIRED_FEATURES}


def score_lab_features(
    *,
    features_path: Path,
    ground_truth_path: Path,
    output_path: Path,
    model_root: Path,
    preprocess_path: Path,
    model_version: str = "final-ae-minimal-v1",
) -> pd.DataFrame:
    merged = merge_features_with_ground_truth(features_path, ground_truth_path)
    scorer = AEScorer(
        model_root=model_root,
        preprocess_path=preprocess_path,
        model_version=model_version,
    )
    scorer.load()

    rows = []
    for _, event in merged.sort_values("event_id", kind="mergesort").iterrows():
        result = scorer.score_event(
            event_id=str(event["event_id"]),
            features=row_features(event),
            rule_flag=False,
            rule_level=0,
        )
        scored = result_to_dict(result)
        rows.append(
            {
                "event_id": event["event_id"],
                "label": event["label"],
                "y_true": int(event["y_true"]),
                "ae_pred": int(scored["ml_alert"]),
                "ml_alert": bool(scored["ml_alert"]),
                "anomaly_score": float(scored["anomaly_score"]),
                "threshold_name": scored["threshold_name"],
                "threshold_value": float(scored["threshold_value"]),
                "risk_level": scored["risk_level"],
                "reason": scored["reason"],
                "scenario": event.get("scenario", ""),
                "source_ip": event.get("source_ip", ""),
                "target_ip": event.get("target_ip", ""),
            }
        )

    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def main() -> None:
    args = parse_args()
    scored = score_lab_features(
        features_path=Path(args.features),
        ground_truth_path=Path(args.ground_truth),
        output_path=Path(args.output),
        model_root=Path(args.model_root),
        preprocess_path=Path(args.preprocess),
        model_version=args.model_version,
    )
    print(f"[OK] AE lab pontozott események: {len(scored)}")
    print(f"[OK] Kimenet: {args.output}")


if __name__ == "__main__":
    main()
