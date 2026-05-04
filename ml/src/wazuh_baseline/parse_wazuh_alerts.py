from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT_COLUMNS = [
    "timestamp",
    "rule_id",
    "rule_level",
    "rule_description",
    "agent_name",
    "source_ip",
    "target_ip",
    "full_log",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def nested_get(obj: dict[str, Any], *paths: str) -> Any:
    for path in paths:
        current: Any = obj
        ok = True
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                ok = False
                break
        if ok and current is not None:
            return current
    return None


def load_alert_objects(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Hiányzó Wazuh alert export: {input_path}")

    if input_path.suffix.lower() in {".jsonl", ".ndjson"}:
        alerts = []
        with input_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    alerts.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Érvénytelen JSON sor a {line_no}. sorban: {exc}") from exc
        return alerts

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["alerts", "data"]:
            value = data.get(key)
            if isinstance(value, list):
                return value
        hits = nested_get(data, "hits.hits")
        if isinstance(hits, list):
            return [hit.get("_source", hit) if isinstance(hit, dict) else hit for hit in hits]
        return [data]
    raise ValueError("A Wazuh exportnak JSON objektumot, listát vagy JSONL sorokat kell tartalmaznia.")


def normalize_alert(alert: dict[str, Any]) -> dict[str, Any]:
    timestamp = nested_get(alert, "timestamp", "@timestamp")
    parsed_timestamp = pd.to_datetime(timestamp, utc=True, errors="coerce")
    timestamp_value = ""
    if not pd.isna(parsed_timestamp):
        timestamp_value = parsed_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

    rule_level = nested_get(alert, "rule.level", "rule_level")
    try:
        rule_level_value = int(rule_level)
    except (TypeError, ValueError):
        rule_level_value = 0

    full_log = nested_get(alert, "full_log", "fullLog", "message")
    if full_log is None:
        full_log = json.dumps(alert, ensure_ascii=False, sort_keys=True)

    return {
        "timestamp": timestamp_value,
        "rule_id": str(nested_get(alert, "rule.id", "rule_id") or ""),
        "rule_level": rule_level_value,
        "rule_description": str(nested_get(alert, "rule.description", "rule_description") or ""),
        "agent_name": str(nested_get(alert, "agent.name", "agent_name") or ""),
        "source_ip": str(
            nested_get(
                alert,
                "data.srcip",
                "data.src_ip",
                "data.source_ip",
                "srcip",
                "src_ip",
                "source_ip",
                "source.ip",
            )
            or ""
        ),
        "target_ip": str(
            nested_get(
                alert,
                "data.dstip",
                "data.dst_ip",
                "data.target_ip",
                "dstip",
                "dst_ip",
                "target_ip",
                "destination.ip",
            )
            or ""
        ),
        "full_log": str(full_log),
    }


def parse_wazuh_alerts(input_path: Path, output_path: Path) -> pd.DataFrame:
    alerts = load_alert_objects(input_path)
    rows = [normalize_alert(alert) for alert in alerts if isinstance(alert, dict)]
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if not df.empty:
        df = df[df["timestamp"].astype(str).str.len() > 0].copy()
        df = df.sort_values(["timestamp", "rule_id", "source_ip", "target_ip"], kind="mergesort")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    args = parse_args()
    df = parse_wazuh_alerts(Path(args.input), Path(args.output))
    print(f"[OK] Normalizált Wazuh alert sorok: {len(df)}")
    print(f"[OK] Kimenet: {args.output}")


if __name__ == "__main__":
    main()
