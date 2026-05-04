from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.src.lab_ae_eval.validate_lab_features import REQUIRED_COLUMNS, validate_lab_features
from ml.src.repo_hygiene.common import (
    is_demo_or_fixture_path,
    load_provenance,
    validate_provenance_payload,
)
from ml.src.scoring_runtime import AEScorer
from ml.src.wazuh_baseline.build_ground_truth import validate_ground_truth
from ml.src.wazuh_baseline.parse_wazuh_alerts import load_alert_objects, nested_get, normalize_alert


OUTPUT_COLUMNS = [
    "integration_event_id",
    "timestamp",
    "event_id",
    "match_method",
    "match_status",
    "source_ip",
    "target_ip",
    "scenario",
    "rule_id",
    "rule_level",
    "rule_description",
    "agent_name",
    "anomaly_score",
    "threshold_name",
    "threshold_value",
    "ml_alert",
    "wazuh_pred",
    "hybrid_or_pred",
    "hybrid_priority_level",
    "hybrid_priority_pred",
    "risk_level",
    "reason",
    "top_level_status",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wazuh-alerts", required=True)
    parser.add_argument("--lab-features", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--model-root", default="artifacts/final/final-ae-minimal-v1")
    parser.add_argument("--preprocess", default="data/processed/final/ae_minimal/preprocess.pkl")
    parser.add_argument("--provenance", default="reports/real_measurement/measurement_provenance.json")
    parser.add_argument("--require-provenance", action="store_true")
    parser.add_argument("--allow-time-only-match", action="store_true")
    return parser.parse_args()


def safe_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if pd.isna(value):
            return None
        return float(value)
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: safe_json_value(value) for key, value in record.items()}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "igen"}


def require_real_input_paths(paths: list[Path]) -> None:
    bad_paths = [path.as_posix() for path in paths if is_demo_or_fixture_path(path)]
    if bad_paths:
        raise ValueError(
            "Real integration módban demo/sample/fixture útvonal nem használható: " + ", ".join(bad_paths)
        )


def validate_provenance(provenance_path: Path | None, require_provenance: bool) -> tuple[bool, list[str]]:
    if provenance_path is None:
        if require_provenance:
            raise FileNotFoundError("A --require-provenance kapcsolóhoz provenance fájl szükséges.")
        return False, ["Provenance fájl nincs megadva."]
    provenance = load_provenance(provenance_path)
    valid, errors = validate_provenance_payload(provenance)
    if require_provenance and not valid:
        raise ValueError("Érvénytelen vagy hiányzó provenance: " + "; ".join(errors))
    return valid, errors


def extract_event_id(alert: dict[str, Any]) -> str:
    value = nested_get(
        alert,
        "event_id",
        "data.event_id",
        "fields.event_id",
        "lab.event_id",
        "rule.event_id",
    )
    return str(value or "").strip()


def load_wazuh_alert_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó Wazuh alert input: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Üres Wazuh alert CSV: {path}")
        rows = []
        for _, row in df.fillna("").iterrows():
            timestamp = pd.to_datetime(row.get("timestamp", ""), utc=True, errors="coerce")
            rule_level = pd.to_numeric(row.get("rule_level", 0), errors="coerce")
            rule_level_value = 0 if pd.isna(rule_level) else int(rule_level)
            rows.append(
                {
                    "timestamp": "" if pd.isna(timestamp) else timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "rule_id": str(row.get("rule_id", "")),
                    "rule_level": rule_level_value,
                    "rule_description": str(row.get("rule_description", "")),
                    "agent_name": str(row.get("agent_name", "")),
                    "source_ip": str(row.get("source_ip", "")),
                    "target_ip": str(row.get("target_ip", "")),
                    "full_log": str(row.get("full_log", "")),
                    "event_id": str(row.get("event_id", "")),
                }
            )
        return rows

    alerts = load_alert_objects(path)
    if not alerts:
        raise ValueError(f"A Wazuh alert input nem tartalmaz eseményt: {path}")
    rows = []
    for alert in alerts:
        if not isinstance(alert, dict):
            continue
        normalized = normalize_alert(alert)
        normalized["event_id"] = extract_event_id(alert)
        rows.append(normalized)
    if not rows:
        raise ValueError(f"A Wazuh alert inputból nem volt feldolgozható esemény: {path}")
    return rows


def prepare_ground_truth(path: Path) -> pd.DataFrame:
    df = validate_ground_truth(path)
    df = df.copy()
    df["timestamp_start_dt"] = pd.to_datetime(df["timestamp_start"], utc=True)
    df["timestamp_end_dt"] = pd.to_datetime(df["timestamp_end"], utc=True)
    return df.sort_values(["timestamp_start", "event_id"], kind="mergesort")


def build_feature_map(features: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        str(row["event_id"]): {name: row[name] for name in REQUIRED_COLUMNS if name != "event_id"}
        for _, row in features.iterrows()
    }


def find_matching_event(
    *,
    alert: dict[str, Any],
    ground_truth: pd.DataFrame,
    feature_map: dict[str, dict[str, Any]],
    allow_time_only_match: bool,
) -> tuple[str, str, pd.Series | None]:
    alert_event_id = str(alert.get("event_id") or "").strip()
    if alert_event_id and alert_event_id in feature_map:
        gt = ground_truth[ground_truth["event_id"] == alert_event_id]
        return alert_event_id, "event_id", None if gt.empty else gt.iloc[0]

    alert_time = pd.to_datetime(alert.get("timestamp", ""), utc=True, errors="coerce")
    if pd.isna(alert_time):
        return "", "", None

    in_window = ground_truth[
        (ground_truth["timestamp_start_dt"] <= alert_time) & (alert_time <= ground_truth["timestamp_end_dt"])
    ].copy()
    if in_window.empty:
        return "", "", None

    source_ip = str(alert.get("source_ip") or "").strip()
    target_ip = str(alert.get("target_ip") or "").strip()
    if source_ip and target_ip:
        ip_matches = in_window[
            (in_window["source_ip"].astype(str) == source_ip) & (in_window["target_ip"].astype(str) == target_ip)
        ]
        ip_matches = ip_matches[ip_matches["event_id"].isin(feature_map)]
        if not ip_matches.empty:
            row = ip_matches.sort_values(["timestamp_start", "event_id"], kind="mergesort").iloc[0]
            return str(row["event_id"]), "time_ip", row

    if allow_time_only_match:
        time_matches = in_window[in_window["event_id"].isin(feature_map)]
        if not time_matches.empty:
            row = time_matches.sort_values(["timestamp_start", "event_id"], kind="mergesort").iloc[0]
            return str(row["event_id"]), "time_only", row

    return "", "", None


def hybrid_priority(wazuh_pred: int, ml_alert: bool) -> tuple[str, int, str]:
    if wazuh_pred and ml_alert:
        return "critical", 1, "Wazuh riasztás és ML pozitív döntés egyaránt jelentkezett."
    if wazuh_pred:
        return "high", 1, "Wazuh riasztás érkezett, ML pozitív döntés nélkül."
    if ml_alert:
        return "medium", 1, "ML pozitív döntés jelentkezett Wazuh riasztás nélkül."
    return "normal", 0, "Nem jelentkezett Wazuh vagy ML pozitív döntés."


def result_as_dict(result: Any) -> dict[str, Any]:
    if is_dataclass(result):
        return asdict(result)
    if isinstance(result, dict):
        return result
    return {
        "anomaly_score": getattr(result, "anomaly_score"),
        "threshold_name": getattr(result, "threshold_name"),
        "threshold_value": getattr(result, "threshold_value"),
        "ml_alert": getattr(result, "ml_alert"),
        "reason": getattr(result, "reason", ""),
    }


def score_matched_alert(
    *,
    scorer: Any,
    event_id: str,
    features: dict[str, Any],
    wazuh_pred: int,
    rule_level: int,
) -> dict[str, Any]:
    result = scorer.score_event(
        event_id=event_id,
        features=features,
        rule_flag=bool(wazuh_pred),
        rule_level=rule_level,
    )
    return result_as_dict(result)


def build_output_row(
    *,
    integration_event_id: str,
    alert: dict[str, Any],
    event_id: str = "",
    match_method: str = "",
    match_status: str = "unmatched",
    scenario: str = "",
    score: dict[str, Any] | None = None,
    error_message: str = "",
) -> dict[str, Any]:
    rule_id = str(alert.get("rule_id") or "")
    rule_level = int(alert.get("rule_level") or 0)
    wazuh_pred = int(rule_level > 0 or bool(rule_id.strip()))
    ml_alert = parse_bool(score.get("ml_alert")) if score else False
    priority_level, priority_pred, hybrid_reason = hybrid_priority(wazuh_pred, ml_alert)
    top_level_status = "error" if error_message else ("scored" if score else "unmatched")
    return {
        "integration_event_id": integration_event_id,
        "timestamp": str(alert.get("timestamp") or ""),
        "event_id": event_id,
        "match_method": match_method,
        "match_status": match_status,
        "source_ip": str(alert.get("source_ip") or ""),
        "target_ip": str(alert.get("target_ip") or ""),
        "scenario": scenario,
        "rule_id": rule_id,
        "rule_level": rule_level,
        "rule_description": str(alert.get("rule_description") or ""),
        "agent_name": str(alert.get("agent_name") or ""),
        "anomaly_score": None if not score else score.get("anomaly_score"),
        "threshold_name": "" if not score else str(score.get("threshold_name") or ""),
        "threshold_value": None if not score else score.get("threshold_value"),
        "ml_alert": ml_alert,
        "wazuh_pred": wazuh_pred,
        "hybrid_or_pred": int(bool(wazuh_pred) or ml_alert),
        "hybrid_priority_level": priority_level,
        "hybrid_priority_pred": priority_pred,
        "risk_level": priority_level,
        "reason": error_message or hybrid_reason,
        "top_level_status": top_level_status,
        "error_message": error_message,
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(clean_record(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    summary = {
        "total_alerts": len(df),
        "scored_alerts": int((df["top_level_status"] == "scored").sum()) if not df.empty else 0,
        "unmatched_alerts": int((df["top_level_status"] == "unmatched").sum()) if not df.empty else 0,
        "error_alerts": int((df["top_level_status"] == "error").sum()) if not df.empty else 0,
        "ml_positive_count": int(df["ml_alert"].fillna(False).astype(bool).sum()) if not df.empty else 0,
        "wazuh_positive_count": int(pd.to_numeric(df["wazuh_pred"], errors="coerce").fillna(0).sum()) if not df.empty else 0,
        "hybrid_positive_count": int(pd.to_numeric(df["hybrid_or_pred"], errors="coerce").fillna(0).sum())
        if not df.empty
        else 0,
        "critical_count": int((df["risk_level"] == "critical").sum()) if not df.empty else 0,
        "high_count": int((df["risk_level"] == "high").sum()) if not df.empty else 0,
        "medium_count": int((df["risk_level"] == "medium").sum()) if not df.empty else 0,
        "normal_count": int((df["risk_level"] == "normal").sum()) if not df.empty else 0,
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / "enrichment_summary.csv", index=False)
    return summary_df


def enrich_wazuh_alerts(
    *,
    wazuh_alerts_path: Path,
    lab_features_path: Path,
    ground_truth_path: Path,
    output_jsonl_path: Path,
    output_csv_path: Path,
    model_root: Path,
    preprocess_path: Path,
    provenance_path: Path | None,
    require_provenance: bool,
    allow_time_only_match: bool,
    scorer: Any | None = None,
) -> dict[str, Path]:
    require_real_input_paths([wazuh_alerts_path, lab_features_path, ground_truth_path])
    provenance_valid, provenance_errors = validate_provenance(provenance_path, require_provenance)

    alerts = load_wazuh_alert_rows(wazuh_alerts_path)
    features = validate_lab_features(lab_features_path)
    ground_truth = prepare_ground_truth(ground_truth_path)
    feature_map = build_feature_map(features)

    active_scorer = scorer or AEScorer(model_root=model_root, preprocess_path=preprocess_path)
    rows: list[dict[str, Any]] = []
    for index, alert in enumerate(alerts, start=1):
        integration_event_id = f"live-{index:06d}"
        event_id, match_method, gt_row = find_matching_event(
            alert=alert,
            ground_truth=ground_truth,
            feature_map=feature_map,
            allow_time_only_match=allow_time_only_match,
        )
        if not event_id:
            rows.append(build_output_row(integration_event_id=integration_event_id, alert=alert))
            continue
        scenario = "" if gt_row is None else str(gt_row.get("scenario", "") or "")
        rule_level = int(alert.get("rule_level") or 0)
        rule_id = str(alert.get("rule_id") or "")
        wazuh_pred = int(rule_level > 0 or bool(rule_id.strip()))
        try:
            score = score_matched_alert(
                scorer=active_scorer,
                event_id=event_id,
                features=feature_map[event_id],
                wazuh_pred=wazuh_pred,
                rule_level=rule_level,
            )
            rows.append(
                build_output_row(
                    integration_event_id=integration_event_id,
                    alert=alert,
                    event_id=event_id,
                    match_method=match_method,
                    match_status="matched",
                    scenario=scenario,
                    score=score,
                )
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                build_output_row(
                    integration_event_id=integration_event_id,
                    alert=alert,
                    event_id=event_id,
                    match_method=match_method,
                    match_status="matched",
                    scenario=scenario,
                    error_message=str(exc),
                )
            )

    output_dir = output_csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output_df.to_csv(output_csv_path, index=False)
    write_jsonl(rows, output_jsonl_path)
    unmatched = output_df[output_df["top_level_status"] == "unmatched"].copy()
    unmatched.to_csv(output_dir / "unmatched_alerts.csv", index=False)
    write_summary(rows, output_dir)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "wazuh_alerts": str(wazuh_alerts_path),
        "lab_features": str(lab_features_path),
        "ground_truth": str(ground_truth_path),
        "output_jsonl": str(output_jsonl_path),
        "output_csv": str(output_csv_path),
        "allow_time_only_match": allow_time_only_match,
        "provenance": "" if provenance_path is None else str(provenance_path),
        "provenance_valid": provenance_valid,
        "provenance_errors": provenance_errors,
        "total_alerts": len(rows),
        "scored_alerts": int((output_df["top_level_status"] == "scored").sum()) if not output_df.empty else 0,
        "unmatched_alerts": len(unmatched),
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "jsonl": output_jsonl_path,
        "csv": output_csv_path,
        "unmatched": output_dir / "unmatched_alerts.csv",
        "summary": output_dir / "enrichment_summary.csv",
        "metadata": metadata_path,
    }


def main() -> None:
    args = parse_args()
    outputs = enrich_wazuh_alerts(
        wazuh_alerts_path=Path(args.wazuh_alerts),
        lab_features_path=Path(args.lab_features),
        ground_truth_path=Path(args.ground_truth),
        output_jsonl_path=Path(args.output_jsonl),
        output_csv_path=Path(args.output_csv),
        model_root=Path(args.model_root),
        preprocess_path=Path(args.preprocess),
        provenance_path=Path(args.provenance) if args.provenance else None,
        require_provenance=args.require_provenance,
        allow_time_only_match=args.allow_time_only_match,
    )
    for path in outputs.values():
        print(f"[OK] {path}")


if __name__ == "__main__":
    main()
