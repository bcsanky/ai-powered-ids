from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.wazuh_baseline.build_ground_truth import REQUIRED_COLUMNS, VALID_LABELS, validate_ground_truth


DEFAULT_STATE_FILE = Path("data/lab/session_events.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_state(state_file: Path) -> list[dict[str, Any]]:
    if not state_file.exists():
        return []
    with state_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"A state fájlnak eseménylistát kell tartalmaznia: {state_file}")
    return data


def write_state(state_file: Path, events: list[dict[str, Any]]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with state_file.open("w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)


def normalize_label(label: str) -> str:
    normalized = str(label).strip().lower()
    if normalized not in VALID_LABELS:
        raise ValueError("A label értéke csak benign vagy attack lehet.")
    return normalized


def find_event(events: list[dict[str, Any]], event_id: str) -> dict[str, Any] | None:
    for event in events:
        if str(event.get("event_id", "")) == event_id:
            return event
    return None


def start_event(
    *,
    state_file: Path,
    event_id: str,
    scenario: str,
    label: str,
    attack_type: str,
    source_ip: str,
    target_ip: str,
    timestamp: str | None = None,
) -> dict[str, Any]:
    event_id = str(event_id).strip()
    if not event_id:
        raise ValueError("Az event_id nem lehet üres.")
    events = read_state(state_file)
    if find_event(events, event_id) is not None:
        raise ValueError(f"Már létezik ilyen event_id a state fájlban: {event_id}")

    event = {
        "event_id": event_id,
        "timestamp_start": timestamp or utc_now(),
        "timestamp_end": "",
        "scenario": str(scenario).strip(),
        "label": normalize_label(label),
        "attack_type": str(attack_type).strip(),
        "source_ip": str(source_ip).strip(),
        "target_ip": str(target_ip).strip(),
    }
    events.append(event)
    write_state(state_file, events)
    return event


def end_event(*, state_file: Path, event_id: str, timestamp: str | None = None) -> dict[str, Any]:
    events = read_state(state_file)
    event = find_event(events, str(event_id).strip())
    if event is None:
        raise ValueError(f"Nem található event_id a state fájlban: {event_id}")
    if str(event.get("timestamp_end", "")).strip():
        raise ValueError(f"Az esemény már le van zárva: {event_id}")
    event["timestamp_end"] = timestamp or utc_now()
    write_state(state_file, events)
    return event


def export_events(*, state_file: Path, output_path: Path) -> pd.DataFrame:
    events = read_state(state_file)
    if not events:
        raise ValueError("Nincs exportálható esemény a state fájlban.")
    open_events = [event["event_id"] for event in events if not str(event.get("timestamp_end", "")).strip()]
    if open_events:
        raise ValueError(f"Lezáratlan események nem exportálhatók: {', '.join(open_events)}")

    df = pd.DataFrame(events)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Hiányzó state mezők: {', '.join(missing)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[REQUIRED_COLUMNS].to_csv(output_path, index=False)
    return validate_ground_truth(output_path, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE))
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start")
    start.add_argument("--event-id", required=True)
    start.add_argument("--scenario", required=True)
    start.add_argument("--label", required=True)
    start.add_argument("--attack-type", default="")
    start.add_argument("--source-ip", required=True)
    start.add_argument("--target-ip", required=True)
    start.add_argument("--timestamp")

    end = subparsers.add_parser("end")
    end.add_argument("--event-id", required=True)
    end.add_argument("--timestamp")

    subparsers.add_parser("list")

    export = subparsers.add_parser("export")
    export.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    state_file = Path(args.state_file)
    if args.command == "start":
        event = start_event(
            state_file=state_file,
            event_id=args.event_id,
            scenario=args.scenario,
            label=args.label,
            attack_type=args.attack_type,
            source_ip=args.source_ip,
            target_ip=args.target_ip,
            timestamp=args.timestamp,
        )
        print(f"[OK] Esemény indítva: {event['event_id']} {event['timestamp_start']}")
    elif args.command == "end":
        event = end_event(state_file=state_file, event_id=args.event_id, timestamp=args.timestamp)
        print(f"[OK] Esemény lezárva: {event['event_id']} {event['timestamp_end']}")
    elif args.command == "list":
        print(json.dumps(read_state(state_file), indent=2, ensure_ascii=False))
    elif args.command == "export":
        df = export_events(state_file=state_file, output_path=Path(args.output))
        print(f"[OK] Exportált ground truth események: {len(df)}")
        print(f"[OK] Kimenet: {args.output}")


if __name__ == "__main__":
    main()
