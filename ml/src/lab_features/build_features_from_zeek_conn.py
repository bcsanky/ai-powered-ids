from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.lab_features.common import aggregate_lab_features
from ml.src.wazuh_baseline.build_ground_truth import validate_ground_truth


DEFAULT_METADATA_PATH = Path("data/processed/lab_features/feature_build_metadata.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conn-log", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--allow-missing-flow", action="store_true")
    parser.add_argument("--epsilon", type=float, default=1e-9)
    return parser.parse_args()


def decode_zeek_separator(value: str) -> str:
    text = value.strip()
    if text == r"\x09":
        return "\t"
    if text == r"\x20":
        return " "
    return text.encode("utf-8").decode("unicode_escape")


def read_zeek_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó Zeek conn log: {path}")
    separator = "\t"
    fields: list[str] | None = None
    data_rows: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n")
            if stripped.startswith("#separator"):
                parts = stripped.split(maxsplit=1)
                if len(parts) == 2:
                    separator = decode_zeek_separator(parts[1])
            elif stripped.startswith("#fields"):
                fields = stripped.split(separator)[1:]
            elif stripped.startswith("#"):
                continue
            elif stripped.strip():
                data_rows.append(stripped)
    if fields is None:
        raise ValueError("A Zeek conn.log nem tartalmaz #fields sort.")
    if not data_rows:
        return pd.DataFrame(columns=fields)
    parsed = [row.split(separator) for row in data_rows]
    return pd.DataFrame(parsed, columns=fields)


def read_conn_log(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        if not path.exists():
            raise FileNotFoundError(f"Hiányzó Zeek conn CSV: {path}")
        return pd.read_csv(path)
    return read_zeek_tsv(path)


def first_existing(df: pd.DataFrame, candidates: list[str], context: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Nem található {context} oszlop. Próbált oszlopok: {', '.join(candidates)}")


def numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def standardize_zeek_conn(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
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
        )
    ts_col = first_existing(df, ["ts", "timestamp"], "timestamp")
    src_col = first_existing(df, ["id.orig_h", "orig_h", "source_ip", "src_ip"], "source_ip")
    dst_col = first_existing(df, ["id.resp_h", "resp_h", "target_ip", "dst_ip"], "target_ip")
    port_col = first_existing(df, ["id.resp_p", "resp_p", "destination_port", "dst_port"], "destination_port")
    proto_col = first_existing(df, ["proto", "protocol"], "protocol")
    duration_col = first_existing(df, ["duration", "flow_duration"], "duration")
    orig_pkts_col = first_existing(df, ["orig_pkts", "total_fwd_packets"], "orig_pkts")
    resp_pkts_col = first_existing(df, ["resp_pkts", "total_backward_packets"], "resp_pkts")
    orig_bytes_col = first_existing(df, ["orig_bytes", "orig_ip_bytes"], "orig_bytes")
    resp_bytes_col = first_existing(df, ["resp_bytes", "resp_ip_bytes"], "resp_bytes")

    return pd.DataFrame(
        {
            "timestamp": df[ts_col],
            "source_ip": df[src_col],
            "target_ip": df[dst_col],
            "destination_port": df[port_col],
            "protocol": df[proto_col],
            "duration": numeric_series(df, duration_col),
            "total_fwd_packets": numeric_series(df, orig_pkts_col),
            "total_backward_packets": numeric_series(df, resp_pkts_col),
            "total_bytes": numeric_series(df, orig_bytes_col) + numeric_series(df, resp_bytes_col),
        }
    )


def build_features_from_zeek(
    *,
    conn_log_path: Path,
    ground_truth_path: Path,
    output_path: Path,
    metadata_output_path: Path = DEFAULT_METADATA_PATH,
    allow_missing_flow: bool = False,
    epsilon: float = 1e-9,
) -> pd.DataFrame:
    ground_truth = validate_ground_truth(ground_truth_path)
    raw_conn = read_conn_log(conn_log_path)
    standardized = standardize_zeek_conn(raw_conn)
    features, metadata = aggregate_lab_features(
        ground_truth=ground_truth,
        flows=standardized,
        allow_missing_flow=allow_missing_flow,
        epsilon=epsilon,
        source="zeek_conn",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)

    metadata_payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "conn_log": str(conn_log_path),
        "ground_truth": str(ground_truth_path),
        "output": str(output_path),
        **metadata,
    }
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.write_text(json.dumps(metadata_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return features


def main() -> None:
    args = parse_args()
    features = build_features_from_zeek(
        conn_log_path=Path(args.conn_log),
        ground_truth_path=Path(args.ground_truth),
        output_path=Path(args.output),
        metadata_output_path=Path(args.metadata_output),
        allow_missing_flow=args.allow_missing_flow,
        epsilon=args.epsilon,
    )
    print(f"[OK] Lab feature sorok: {len(features)}")
    print(f"[OK] Kimenet: {args.output}")


if __name__ == "__main__":
    main()
