from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


LAB_FEATURE_COLUMNS = [
    "event_id",
    "timestamp",
    "destination_port",
    "flow_duration",
    "total_fwd_packets",
    "total_backward_packets",
    "flow_bytes_per_sec",
    "flow_packets_per_sec",
    "protocol",
    "source_ip",
    "target_ip",
    "scenario",
]


def ensure_columns(df: pd.DataFrame, required: list[str], context: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Hiányzó oszlopok ({context}): {', '.join(missing)}")


def parse_utc(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def clean_ip(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def deterministic_numeric_mode(values: pd.Series, default: int = 0) -> int:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return default
    counts = numeric.astype(int).value_counts()
    max_count = counts.max()
    candidates = sorted(int(value) for value in counts[counts == max_count].index.tolist())
    return candidates[0]


def deterministic_text_mode(values: pd.Series, default: str = "unknown") -> str:
    texts = values.dropna().astype(str).str.strip()
    texts = texts[texts != ""]
    if texts.empty:
        return default
    counts = texts.value_counts()
    max_count = counts.max()
    candidates = sorted(counts[counts == max_count].index.tolist())
    return candidates[0]


def standardize_flow_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "timestamp",
        "source_ip",
        "target_ip",
        "destination_port",
        "protocol",
        "duration",
        "total_fwd_packets",
        "total_backward_packets",
        "total_bytes",
    ]
    ensure_columns(df, required, "standardizált flow")
    out = df[required].copy()
    out["timestamp"] = parse_utc(out["timestamp"])
    if out["timestamp"].isna().any():
        raise ValueError("A flow timestamp mező nem parse-olható minden sorban.")
    for col in ["duration", "total_fwd_packets", "total_backward_packets", "total_bytes"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["destination_port"] = pd.to_numeric(out["destination_port"], errors="coerce")
    if out["destination_port"].isna().any():
        raise ValueError("A destination_port mező nem konvertálható számmá minden flow sorban.")
    out["source_ip"] = out["source_ip"].fillna("").astype(str).str.strip()
    out["target_ip"] = out["target_ip"].fillna("").astype(str).str.strip()
    out["protocol"] = out["protocol"].fillna("").astype(str).str.strip()
    return out


def event_flow_subset(flows: pd.DataFrame, event: pd.Series) -> pd.DataFrame:
    start = pd.to_datetime(event["timestamp_start"], utc=True)
    end = pd.to_datetime(event["timestamp_end"], utc=True)
    subset = flows[(flows["timestamp"] >= start) & (flows["timestamp"] <= end)].copy()
    source_ip = clean_ip(event.get("source_ip"))
    target_ip = clean_ip(event.get("target_ip"))
    if source_ip:
        subset = subset[subset["source_ip"] == source_ip]
    if target_ip:
        subset = subset[subset["target_ip"] == target_ip]
    return subset


def aggregate_lab_features(
    *,
    ground_truth: pd.DataFrame,
    flows: pd.DataFrame,
    allow_missing_flow: bool = False,
    epsilon: float = 1e-9,
    source: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    flows = standardize_flow_frame(flows)
    rows = []
    missing_flow_events: list[str] = []
    zero_duration_events: list[str] = []

    for _, event in ground_truth.sort_values("event_id", kind="mergesort").iterrows():
        event_id = str(event["event_id"])
        subset = event_flow_subset(flows, event)
        if subset.empty:
            if not allow_missing_flow:
                raise ValueError(f"Nincs illeszkedő flow ehhez az event_id-hez: {event_id}")
            missing_flow_events.append(event_id)
            rows.append(
                {
                    "event_id": event_id,
                    "timestamp": event["timestamp_start"],
                    "destination_port": 0,
                    "flow_duration": 0.0,
                    "total_fwd_packets": 0.0,
                    "total_backward_packets": 0.0,
                    "flow_bytes_per_sec": 0.0,
                    "flow_packets_per_sec": 0.0,
                    "protocol": "unknown",
                    "source_ip": clean_ip(event.get("source_ip")),
                    "target_ip": clean_ip(event.get("target_ip")),
                    "scenario": str(event.get("scenario", "")),
                }
            )
            continue

        duration_sum = float(np.maximum(subset["duration"].sum(), 0.0))
        duration_for_rate = duration_sum
        if duration_for_rate <= 0:
            zero_duration_events.append(event_id)
            duration_for_rate = epsilon
        total_fwd_packets = float(subset["total_fwd_packets"].sum())
        total_backward_packets = float(subset["total_backward_packets"].sum())
        total_packets = total_fwd_packets + total_backward_packets
        total_bytes = float(subset["total_bytes"].sum())

        rows.append(
            {
                "event_id": event_id,
                "timestamp": event["timestamp_start"],
                "destination_port": deterministic_numeric_mode(subset["destination_port"]),
                "flow_duration": duration_sum,
                "total_fwd_packets": total_fwd_packets,
                "total_backward_packets": total_backward_packets,
                "flow_bytes_per_sec": total_bytes / duration_for_rate,
                "flow_packets_per_sec": total_packets / duration_for_rate,
                "protocol": deterministic_text_mode(subset["protocol"]),
                "source_ip": clean_ip(event.get("source_ip")),
                "target_ip": clean_ip(event.get("target_ip")),
                "scenario": str(event.get("scenario", "")),
            }
        )

    metadata = {
        "source": source,
        "feature_rows": len(rows),
        "allow_missing_flow": allow_missing_flow,
        "missing_flow_events": missing_flow_events,
        "zero_duration_events": zero_duration_events,
        "epsilon": epsilon,
        "aggregation": {
            "flow_duration": "sum_seconds",
            "total_fwd_packets": "sum_orig_or_forward_packets",
            "total_backward_packets": "sum_resp_or_backward_packets",
            "destination_port": "most_frequent_lowest_value_tie_break",
            "protocol": "most_frequent_lexicographic_tie_break",
            "rates": "sum_counts_or_bytes_divided_by_duration_sum",
        },
    }
    return pd.DataFrame(rows, columns=LAB_FEATURE_COLUMNS), metadata
