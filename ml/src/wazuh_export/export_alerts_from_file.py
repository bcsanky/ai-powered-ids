from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.wazuh_baseline.parse_wazuh_alerts import load_alert_objects, nested_get


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--time-start", required=True)
    parser.add_argument("--time-end", required=True)
    parser.add_argument("--metadata-output")
    return parser.parse_args()


def parse_time(value: str, name: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"A(z) {name} nem parse-olható időbélyeg: {value}")
    return parsed


def alert_timestamp(alert: dict[str, Any]) -> pd.Timestamp:
    value = nested_get(alert, "@timestamp", "timestamp")
    return pd.to_datetime(value, utc=True, errors="coerce")


def filter_alerts_by_time(
    alerts: list[dict[str, Any]],
    *,
    time_start: str,
    time_end: str,
) -> list[dict[str, Any]]:
    start = parse_time(time_start, "time_start")
    end = parse_time(time_end, "time_end")
    if end < start:
        raise ValueError("A time_end nem lehet korábbi, mint a time_start.")

    filtered = []
    for alert in alerts:
        timestamp = alert_timestamp(alert)
        if pd.isna(timestamp):
            continue
        if start <= timestamp <= end:
            filtered.append(alert)
    return filtered


def write_jsonl(path: Path, alerts: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for alert in alerts:
            f.write(json.dumps(alert, ensure_ascii=False, sort_keys=True) + "\n")


def export_alerts_from_file(
    *,
    input_path: Path,
    output_path: Path,
    time_start: str,
    time_end: str,
    metadata_output_path: Path | None = None,
) -> dict[str, Any]:
    alerts = load_alert_objects(input_path)
    filtered = filter_alerts_by_time(alerts, time_start=time_start, time_end=time_end)
    write_jsonl(output_path, filtered)

    metadata_path = metadata_output_path or (output_path.parent / "alerts_file_export_metadata.json")
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(input_path),
        "time_start": time_start,
        "time_end": time_end,
        "output": str(output_path),
        "event_count": len(filtered),
        "warning": "0 találat volt az időablakban." if not filtered else "",
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    metadata = export_alerts_from_file(
        input_path=Path(args.input),
        output_path=Path(args.output),
        time_start=args.time_start,
        time_end=args.time_end,
        metadata_output_path=Path(args.metadata_output) if args.metadata_output else None,
    )
    if metadata["event_count"] == 0:
        print("[WARN] A megadott időablakban nem volt exportálható alert.")
    print(f"[OK] Exportált Wazuh alert sorok: {metadata['event_count']}")
    print(f"[OK] Kimenet: {metadata['output']}")


if __name__ == "__main__":
    main()
