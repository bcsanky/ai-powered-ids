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
    parser.add_argument("--flow-csv", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--src-ip-col", default="source_ip")
    parser.add_argument("--dst-ip-col", default="target_ip")
    parser.add_argument("--dst-port-col", default="destination_port")
    parser.add_argument("--protocol-col", default="protocol")
    parser.add_argument("--duration-col", default="flow_duration")
    parser.add_argument("--fwd-packets-col", default="total_fwd_packets")
    parser.add_argument("--bwd-packets-col", default="total_backward_packets")
    parser.add_argument("--bytes-per-sec-col", default="flow_bytes_per_sec")
    parser.add_argument("--packets-per-sec-col", default="flow_packets_per_sec")
    parser.add_argument("--metadata-output", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--allow-missing-flow", action="store_true")
    parser.add_argument("--epsilon", type=float, default=1e-9)
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Hiányzó flow CSV oszlopok: {', '.join(missing)}")


def standardize_flow_csv(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    required = [
        args.timestamp_col,
        args.src_ip_col,
        args.dst_ip_col,
        args.dst_port_col,
        args.protocol_col,
        args.duration_col,
        args.fwd_packets_col,
        args.bwd_packets_col,
        args.bytes_per_sec_col,
        args.packets_per_sec_col,
    ]
    require_columns(df, required)
    duration = pd.to_numeric(df[args.duration_col], errors="coerce").fillna(0.0)
    bytes_per_sec = pd.to_numeric(df[args.bytes_per_sec_col], errors="coerce").fillna(0.0)
    return pd.DataFrame(
        {
            "timestamp": df[args.timestamp_col],
            "source_ip": df[args.src_ip_col],
            "target_ip": df[args.dst_ip_col],
            "destination_port": df[args.dst_port_col],
            "protocol": df[args.protocol_col],
            "duration": duration,
            "total_fwd_packets": pd.to_numeric(df[args.fwd_packets_col], errors="coerce").fillna(0.0),
            "total_backward_packets": pd.to_numeric(df[args.bwd_packets_col], errors="coerce").fillna(0.0),
            "total_bytes": bytes_per_sec * duration,
        }
    )


def build_features_from_flow_csv(
    *,
    flow_csv_path: Path,
    ground_truth_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    metadata_output_path: Path = DEFAULT_METADATA_PATH,
    allow_missing_flow: bool = False,
    epsilon: float = 1e-9,
) -> pd.DataFrame:
    if not flow_csv_path.exists():
        raise FileNotFoundError(f"Hiányzó flow CSV: {flow_csv_path}")
    ground_truth = validate_ground_truth(ground_truth_path)
    raw = pd.read_csv(flow_csv_path)
    standardized = standardize_flow_csv(raw, args)
    features, metadata = aggregate_lab_features(
        ground_truth=ground_truth,
        flows=standardized,
        allow_missing_flow=allow_missing_flow,
        epsilon=epsilon,
        source="flow_csv",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    metadata_payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "flow_csv": str(flow_csv_path),
        "ground_truth": str(ground_truth_path),
        "output": str(output_path),
        "column_mapping": {
            "timestamp": args.timestamp_col,
            "source_ip": args.src_ip_col,
            "target_ip": args.dst_ip_col,
            "destination_port": args.dst_port_col,
            "protocol": args.protocol_col,
            "duration": args.duration_col,
            "total_fwd_packets": args.fwd_packets_col,
            "total_backward_packets": args.bwd_packets_col,
            "flow_bytes_per_sec": args.bytes_per_sec_col,
            "flow_packets_per_sec": args.packets_per_sec_col,
        },
        **metadata,
    }
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.write_text(json.dumps(metadata_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return features


def main() -> None:
    args = parse_args()
    features = build_features_from_flow_csv(
        flow_csv_path=Path(args.flow_csv),
        ground_truth_path=Path(args.ground_truth),
        output_path=Path(args.output),
        args=args,
        metadata_output_path=Path(args.metadata_output),
        allow_missing_flow=args.allow_missing_flow,
        epsilon=args.epsilon,
    )
    print(f"[OK] Lab feature sorok: {len(features)}")
    print(f"[OK] Kimenet: {args.output}")


if __name__ == "__main__":
    main()
