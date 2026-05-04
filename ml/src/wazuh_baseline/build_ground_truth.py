from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


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
VALID_LABELS = {"benign", "attack"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def validate_ground_truth(input_path: Path, output_path: Path | None = None) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Hiányzó ground truth CSV: {input_path}")

    df = pd.read_csv(input_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Hiányzó kötelező oszlopok: {', '.join(missing)}")

    df = df[REQUIRED_COLUMNS].copy()
    if df["event_id"].isna().any():
        raise ValueError("Az event_id oszlop nem tartalmazhat üres értéket.")
    df["event_id"] = df["event_id"].astype(str).str.strip()
    if df["event_id"].eq("").any():
        raise ValueError("Az event_id oszlop nem tartalmazhat üres értéket.")
    if df["event_id"].duplicated().any():
        duplicates = ", ".join(sorted(df.loc[df["event_id"].duplicated(), "event_id"].unique()))
        raise ValueError(f"Duplikált event_id értékek: {duplicates}")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    invalid_labels = sorted(set(df["label"]) - VALID_LABELS)
    if invalid_labels:
        raise ValueError(f"Érvénytelen label értékek: {', '.join(invalid_labels)}")

    start = pd.to_datetime(df["timestamp_start"], utc=True, errors="coerce")
    end = pd.to_datetime(df["timestamp_end"], utc=True, errors="coerce")
    if start.isna().any() or end.isna().any():
        raise ValueError("A timestamp_start és timestamp_end oszlopoknak érvényes időbélyegeket kell tartalmazniuk.")
    if (end < start).any():
        raise ValueError("A timestamp_end nem lehet korábbi, mint a timestamp_start.")

    df["timestamp_start"] = start.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df["timestamp_end"] = end.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    for col in ["scenario", "attack_type", "source_ip", "target_ip"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


def main() -> None:
    args = parse_args()
    output_path = Path(args.output) if args.output else None
    df = validate_ground_truth(Path(args.input), output_path)
    print(f"[OK] Validált ground truth események: {len(df)}")
    if output_path is not None:
        print(f"[OK] Kimenet: {output_path}")


if __name__ == "__main__":
    main()
