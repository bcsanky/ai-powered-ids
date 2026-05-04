from __future__ import annotations

import argparse
from pathlib import Path

from ml.src.lab_capture.event_marker import DEFAULT_STATE_FILE, end_event, export_events, read_state, start_event


SCENARIO_MAPPING = {
    "benign_ssh_login": {"label": "benign", "attack_type": "none"},
    "benign_package_update": {"label": "benign", "attack_type": "none"},
    "port_scan": {"label": "attack", "attack_type": "port_scan"},
    "ssh_failed_logins": {"label": "attack", "attack_type": "failed_login"},
    "ssh_bruteforce": {"label": "attack", "attack_type": "brute_force"},
    "file_integrity_change": {"label": "attack", "attack_type": "file_integrity_change"},
    "privilege_change": {"label": "attack", "attack_type": "privilege_change"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE))
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start")
    start.add_argument("--scenario", required=True, choices=sorted(SCENARIO_MAPPING))
    start.add_argument("--event-id", required=True)
    start.add_argument("--source-ip", required=True)
    start.add_argument("--target-ip", required=True)
    start.add_argument("--timestamp")

    end = subparsers.add_parser("end")
    end.add_argument("--event-id", required=True)
    end.add_argument("--timestamp")

    subparsers.add_parser("list")

    export = subparsers.add_parser("export")
    export.add_argument("--output", required=True)
    return parser.parse_args()


def scenario_defaults(scenario: str) -> dict[str, str]:
    if scenario not in SCENARIO_MAPPING:
        raise ValueError(f"Ismeretlen scenario: {scenario}")
    return SCENARIO_MAPPING[scenario]


def start_scenario(
    *,
    state_file: Path,
    scenario: str,
    event_id: str,
    source_ip: str,
    target_ip: str,
    timestamp: str | None = None,
) -> dict[str, str]:
    defaults = scenario_defaults(scenario)
    return start_event(
        state_file=state_file,
        event_id=event_id,
        scenario=scenario,
        label=defaults["label"],
        attack_type=defaults["attack_type"],
        source_ip=source_ip,
        target_ip=target_ip,
        timestamp=timestamp,
    )


def main() -> None:
    args = parse_args()
    state_file = Path(args.state_file)
    if args.command == "start":
        event = start_scenario(
            state_file=state_file,
            scenario=args.scenario,
            event_id=args.event_id,
            source_ip=args.source_ip,
            target_ip=args.target_ip,
            timestamp=args.timestamp,
        )
        print(f"[OK] Scenario indítva: {event['event_id']} {event['scenario']} {event['timestamp_start']}")
    elif args.command == "end":
        event = end_event(state_file=state_file, event_id=args.event_id, timestamp=args.timestamp)
        print(f"[OK] Scenario lezárva: {event['event_id']} {event['timestamp_end']}")
    elif args.command == "list":
        for event in read_state(state_file):
            print(event)
    elif args.command == "export":
        df = export_events(state_file=state_file, output_path=Path(args.output))
        print(f"[OK] Exportált ground truth események: {len(df)}")
        print(f"[OK] Kimenet: {args.output}")


if __name__ == "__main__":
    main()

