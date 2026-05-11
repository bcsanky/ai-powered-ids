#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import Counter
from pathlib import Path


REQUIRED_COLUMNS = [
    "event_id",
    "timestamp_start",
    "timestamp_end",
    "scenario",
    "label",
    "attack_type",
    "source_ip",
    "target_ip",
]

EXPECTED_LABEL_COUNTS = {"benign": 60, "attack": 40}
EXPECTED_SCENARIO_COUNTS = {
    "benign_ssh_login": 30,
    "benign_package_update": 30,
    "port_scan": 8,
    "ssh_failed_logins": 8,
    "ssh_bruteforce": 8,
    "file_integrity_change": 8,
    "privilege_change": 8,
}
EXPECTED_RERUN_EVENT_IDS = {
    "ABRUTE-006",
    "APORT-007",
    "APORT-008",
    "APRIV-006",
    "APRIV-008",
    "BPKG-014",
    "BPKG-016",
    "BPKG-021",
    "BPKG-025",
    "BPKG-026",
    "BPKG-027",
    "BPKG-028",
    "BPKG-029",
    "BPKG-030",
    "BSSH-015",
    "BSSH-024",
    "BSSH-025",
    "BSSH-026",
    "BSSH-027",
    "BSSH-028",
    "BSSH-029",
    "BSSH-030",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace base ground-truth rows with real rerun rows for missing Zeek-flow events."
    )
    parser.add_argument("--base", default="data/lab/lab_ground_truth.csv")
    parser.add_argument("--rerun", default="data/lab/rerun_missing_flow/lab_ground_truth.csv")
    parser.add_argument("--output", default="data/lab/lab_ground_truth.csv")
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Üres vagy fejléc nélküli CSV: {path}")
        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"Hiányzó oszlopok {path}: {', '.join(missing)}")
        rows = [{col: (row.get(col) or "").strip() for col in reader.fieldnames} for row in reader]
    return list(reader.fieldnames), rows


def require_unique_event_ids(rows: list[dict[str, str]], context: str) -> None:
    event_ids = [row["event_id"] for row in rows]
    if any(not event_id for event_id in event_ids):
        raise ValueError(f"{context}: üres event_id található.")
    duplicates = sorted(event_id for event_id, count in Counter(event_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"{context}: duplikált event_id: {', '.join(duplicates)}")


def validate_rerun_rows(rerun_rows: list[dict[str, str]], base_ids: set[str]) -> None:
    rerun_ids = {row["event_id"] for row in rerun_rows}
    if rerun_ids != EXPECTED_RERUN_EVENT_IDS:
        missing = sorted(EXPECTED_RERUN_EVENT_IDS - rerun_ids)
        extra = sorted(rerun_ids - EXPECTED_RERUN_EVENT_IDS)
        details = []
        if missing:
            details.append("hiányzó: " + ", ".join(missing))
        if extra:
            details.append("váratlan: " + ", ".join(extra))
        raise ValueError("A rerun CSV-nek pontosan a 22 missing-flow event ID-t kell tartalmaznia; " + "; ".join(details))
    unknown = sorted(row["event_id"] for row in rerun_rows if row["event_id"] not in base_ids)
    if unknown:
        raise ValueError(f"A rerun ismeretlen event_id-t tartalmaz: {', '.join(unknown)}")
    missing_timestamps = [
        row["event_id"]
        for row in rerun_rows
        if not row.get("timestamp_start", "").strip() or not row.get("timestamp_end", "").strip()
    ]
    if missing_timestamps:
        raise ValueError(
            "A rerun soroknak valós marker timestampet kell tartalmazniuk; hiányos event_id-k: "
            + ", ".join(missing_timestamps)
        )


def validate_output(rows: list[dict[str, str]]) -> None:
    if len(rows) != 100:
        raise ValueError(f"A kimenetnek 100 sort kell tartalmaznia, aktuális sorok: {len(rows)}")
    require_unique_event_ids(rows, "output")
    label_counts = Counter(row["label"].strip().lower() for row in rows)
    if dict(label_counts) != EXPECTED_LABEL_COUNTS:
        raise ValueError(f"Hibás label eloszlás: {dict(label_counts)} != {EXPECTED_LABEL_COUNTS}")
    scenario_counts = Counter(row["scenario"].strip() for row in rows)
    if dict(scenario_counts) != EXPECTED_SCENARIO_COUNTS:
        raise ValueError(f"Hibás scenario eloszlás: {dict(scenario_counts)} != {EXPECTED_SCENARIO_COUNTS}")


def write_rows_atomic(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({col: row.get(col, "") for col in fieldnames})
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    args = parse_args()
    base_path = Path(args.base)
    rerun_path = Path(args.rerun)
    output_path = Path(args.output)

    fieldnames, base_rows = read_rows(base_path)
    _, rerun_rows = read_rows(rerun_path)
    require_unique_event_ids(base_rows, "base")
    require_unique_event_ids(rerun_rows, "rerun")

    base_ids = {row["event_id"] for row in base_rows}
    validate_rerun_rows(rerun_rows, base_ids)

    rerun_by_id = {row["event_id"]: row for row in rerun_rows}
    merged_rows = [rerun_by_id.get(row["event_id"], row) for row in base_rows]
    validate_output(merged_rows)

    write_rows_atomic(output_path, fieldnames, merged_rows)
    print(f"[OK] Base sorok: {len(base_rows)}")
    print(f"[OK] Cserélt rerun sorok: {len(rerun_rows)}")
    print(f"[OK] Kimenet: {output_path}")


if __name__ == "__main__":
    main()
