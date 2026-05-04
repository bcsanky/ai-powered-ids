from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from ml.src.wazuh_baseline.build_ground_truth import validate_ground_truth
except ModuleNotFoundError:
    from build_ground_truth import validate_ground_truth


OUTPUT_COLUMNS = [
    "event_id",
    "label",
    "y_true",
    "wazuh_pred",
    "first_alert_time",
    "time_to_detection_sec",
    "matched_rule_ids",
    "max_rule_level",
    "alert_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--alerts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-seconds", type=int, default=60)
    return parser.parse_args()


def clean_ip(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def alert_matches_event(event: pd.Series, alert: pd.Series) -> bool:
    checks = []
    event_source = clean_ip(event.get("source_ip"))
    event_target = clean_ip(event.get("target_ip"))
    alert_source = clean_ip(alert.get("source_ip"))
    alert_target = clean_ip(alert.get("target_ip"))

    if alert_source and event_source:
        checks.append(alert_source == event_source)
    if alert_target and event_target:
        checks.append(alert_target == event_target)
    return bool(checks) and all(checks)


def unique_join(values: pd.Series) -> str:
    seen = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.append(text)
    return ";".join(seen)


def correlate_alerts(
    *,
    ground_truth_path: Path,
    alerts_path: Path,
    output_path: Path,
    window_seconds: int = 60,
) -> pd.DataFrame:
    if window_seconds < 0:
        raise ValueError("A window_seconds nem lehet negatív.")
    if not alerts_path.exists():
        raise FileNotFoundError(f"Hiányzó normalizált Wazuh alert CSV: {alerts_path}")

    ground_truth = validate_ground_truth(ground_truth_path)
    alerts = pd.read_csv(alerts_path)
    for col in ["timestamp", "rule_id", "rule_level", "source_ip", "target_ip"]:
        if col not in alerts.columns:
            alerts[col] = ""
    alerts = alerts.copy()
    alerts["timestamp_parsed"] = pd.to_datetime(alerts["timestamp"], utc=True, errors="coerce")
    alerts = alerts.dropna(subset=["timestamp_parsed"]).sort_values("timestamp_parsed", kind="mergesort")

    gt = ground_truth.copy()
    gt["timestamp_start_parsed"] = pd.to_datetime(gt["timestamp_start"], utc=True, errors="coerce")
    gt["timestamp_end_parsed"] = pd.to_datetime(gt["timestamp_end"], utc=True, errors="coerce")

    rows = []
    for _, event in gt.iterrows():
        start = event["timestamp_start_parsed"]
        end = event["timestamp_end_parsed"] + pd.Timedelta(seconds=window_seconds)
        time_candidates = alerts[
            (alerts["timestamp_parsed"] >= start)
            & (alerts["timestamp_parsed"] <= end)
        ]
        matched = time_candidates[time_candidates.apply(lambda row: alert_matches_event(event, row), axis=1)]

        label = str(event["label"]).strip().lower()
        y_true = 1 if label == "attack" else 0
        wazuh_pred = 1 if len(matched) > 0 else 0
        first_alert_time = ""
        time_to_detection = ""
        matched_rule_ids = ""
        max_rule_level = 0

        if wazuh_pred:
            first = matched.iloc[0]
            first_time = first["timestamp_parsed"]
            first_alert_time = first_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            time_to_detection = float((first_time - start).total_seconds())
            matched_rule_ids = unique_join(matched["rule_id"])
            max_rule_level = int(pd.to_numeric(matched["rule_level"], errors="coerce").fillna(0).max())

        rows.append(
            {
                "event_id": event["event_id"],
                "label": label,
                "y_true": y_true,
                "wazuh_pred": wazuh_pred,
                "first_alert_time": first_alert_time,
                "time_to_detection_sec": time_to_detection,
                "matched_rule_ids": matched_rule_ids,
                "max_rule_level": max_rule_level,
                "alert_count": int(len(matched)),
            }
        )

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    args = parse_args()
    df = correlate_alerts(
        ground_truth_path=Path(args.ground_truth),
        alerts_path=Path(args.alerts),
        output_path=Path(args.output),
        window_seconds=args.window_seconds,
    )
    print(f"[OK] Korrelált események: {len(df)}")
    print(f"[OK] Kimenet: {args.output}")


if __name__ == "__main__":
    main()
