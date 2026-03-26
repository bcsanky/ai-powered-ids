from __future__ import annotations

COLUMN_ALIASES = {
    "destination_port": [
        "destination_port",
        "destination port",
        "dst_port",
        "dst port",
    ],
    "flow_duration": [
        "flow_duration",
        "flow duration",
    ],
    "total_fwd_packets": [
        "total_fwd_packets",
        "total fwd packets",
        "total_forward_packets",
    ],
    "total_backward_packets": [
        "total_backward_packets",
        "total backward packets",
        "total_bwd_packets",
    ],
    "flow_bytes_per_sec": [
        "flow_bytes_per_sec",
        "flow bytes/s",
        "flow bytes per sec",
    ],
    "flow_packets_per_sec": [
        "flow_packets_per_sec",
        "flow packets/s",
        "flow packets per sec",
    ],
    "protocol": [
        "protocol",
    ],
    "label": [
        "label",
    ],
}


def normalize_colname(col: str) -> str:
    col = col.strip().lower()
    col = col.replace("/", "_per_")
    col = col.replace(" ", "_")
    col = col.replace("-", "_")
    return col


def canonicalize_columns(columns: list[str]) -> dict[str, str]:
    normalized = {normalize_colname(c): c for c in columns}
    result: dict[str, str] = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_norm = normalize_colname(alias)
            if alias_norm in normalized:
                result[canonical] = normalized[alias_norm]
                break

    return result
