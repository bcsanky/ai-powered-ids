from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "event_id",
    "destination_port",
    "flow_duration",
    "total_fwd_packets",
    "total_backward_packets",
    "flow_bytes_per_sec",
    "flow_packets_per_sec",
    "protocol",
]
NUMERIC_COLUMNS = [
    "destination_port",
    "flow_duration",
    "total_fwd_packets",
    "total_backward_packets",
    "flow_bytes_per_sec",
    "flow_packets_per_sec",
]
OPTIONAL_COLUMNS = ["timestamp", "source_ip", "target_ip", "scenario"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def validate_lab_features(input_path: Path, output_path: Path | None = None) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Hiányzó lab feature CSV: {input_path}")

    df = pd.read_csv(input_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Hiányzó kötelező lab feature oszlopok: {', '.join(missing)}")

    kept_columns = REQUIRED_COLUMNS + [col for col in OPTIONAL_COLUMNS if col in df.columns]
    df = df[kept_columns].copy()

    if df["event_id"].isna().any():
        raise ValueError("Az event_id oszlop nem tartalmazhat üres értéket.")
    df["event_id"] = df["event_id"].astype(str).str.strip()
    if df["event_id"].eq("").any():
        raise ValueError("Az event_id oszlop nem tartalmazhat üres értéket.")
    if df["event_id"].duplicated().any():
        duplicates = ", ".join(sorted(df.loc[df["event_id"].duplicated(), "event_id"].unique()))
        raise ValueError(f"Duplikált event_id értékek: {duplicates}")

    for col in NUMERIC_COLUMNS:
        converted = pd.to_numeric(df[col], errors="coerce")
        invalid = converted.isna()
        if invalid.any():
            raise ValueError(f"A(z) {col} oszlop minden értékének számmá konvertálhatónak kell lennie.")
        df[col] = converted

    if df["protocol"].isna().any():
        raise ValueError("A protocol oszlop nem lehet üres.")
    df["protocol"] = df["protocol"].astype(str).str.strip()
    if df["protocol"].eq("").any():
        raise ValueError("A protocol oszlop nem lehet üres.")

    if "timestamp" in df.columns:
        timestamp = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if timestamp.isna().any():
            raise ValueError("A timestamp oszlopnak ISO-szerűen parse-olható értékeket kell tartalmaznia.")
        df["timestamp"] = timestamp.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    for col in ["source_ip", "target_ip", "scenario"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


def main() -> None:
    args = parse_args()
    validated = validate_lab_features(Path(args.input), Path(args.output))
    print(f"[OK] Validált lab feature események: {len(validated)}")
    print(f"[OK] Kimenet: {args.output}")


if __name__ == "__main__":
    main()
