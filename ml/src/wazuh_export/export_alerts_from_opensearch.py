from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from ml.src.wazuh_export.export_alerts_from_file import parse_time, write_jsonl


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "igen"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opensearch-url", required=True)
    parser.add_argument("--index-pattern", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--time-start", required=True)
    parser.add_argument("--time-end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-tls", default="true")
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--metadata-output")
    return parser.parse_args()


def build_query(
    *,
    time_start: str,
    time_end: str,
    page_size: int,
    search_after: list[Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "size": page_size,
        "query": {
            "range": {
                "@timestamp": {
                    "gte": time_start,
                    "lte": time_end,
                    "format": "strict_date_optional_time",
                }
            }
        },
        "sort": [
            {"@timestamp": {"order": "asc"}},
            {"_id": {"order": "asc"}},
        ],
    }
    if search_after is not None:
        body["search_after"] = search_after
    return body


def fetch_alerts(
    *,
    opensearch_url: str,
    index_pattern: str,
    username: str,
    password: str,
    time_start: str,
    time_end: str,
    verify_tls: bool,
    page_size: int,
    max_events: int | None = None,
) -> list[dict[str, Any]]:
    parse_time(time_start, "time_start")
    parse_time(time_end, "time_end")
    if page_size <= 0:
        raise ValueError("A page_size pozitív egész kell legyen.")

    alerts: list[dict[str, Any]] = []
    search_after = None
    url = f"{opensearch_url.rstrip('/')}/{index_pattern}/_search"
    while True:
        body = build_query(
            time_start=time_start,
            time_end=time_end,
            page_size=page_size,
            search_after=search_after,
        )
        response = requests.post(
            url,
            auth=(username, password),
            json=body,
            verify=verify_tls,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        hits = payload.get("hits", {}).get("hits", [])
        if not hits:
            break
        for hit in hits:
            source = hit.get("_source", hit)
            if isinstance(source, dict):
                alerts.append(source)
            if max_events is not None and len(alerts) >= max_events:
                return alerts[:max_events]
        search_after = hits[-1].get("sort")
        if search_after is None or len(hits) < page_size:
            break
    return alerts


def export_alerts_from_opensearch(
    *,
    opensearch_url: str,
    index_pattern: str,
    username: str,
    password: str,
    time_start: str,
    time_end: str,
    output_path: Path,
    verify_tls: bool,
    page_size: int = 1000,
    max_events: int | None = None,
    metadata_output_path: Path | None = None,
) -> dict[str, Any]:
    alerts = fetch_alerts(
        opensearch_url=opensearch_url,
        index_pattern=index_pattern,
        username=username,
        password=password,
        time_start=time_start,
        time_end=time_end,
        verify_tls=verify_tls,
        page_size=page_size,
        max_events=max_events,
    )
    write_jsonl(output_path, alerts)
    metadata_path = metadata_output_path or (output_path.parent / "alerts_export_metadata.json")
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "opensearch_url": opensearch_url,
        "index_pattern": index_pattern,
        "time_start": time_start,
        "time_end": time_end,
        "output": str(output_path),
        "event_count": len(alerts),
        "page_size": page_size,
        "max_events": max_events,
        "verify_tls": verify_tls,
        "warning": "0 találat volt az időablakban." if not alerts else "",
        "tls_note": "TLS ellenőrzés kikapcsolva labor környezetben." if not verify_tls else "",
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    metadata = export_alerts_from_opensearch(
        opensearch_url=args.opensearch_url,
        index_pattern=args.index_pattern,
        username=args.username,
        password=args.password,
        time_start=args.time_start,
        time_end=args.time_end,
        output_path=Path(args.output),
        verify_tls=parse_bool(args.verify_tls),
        page_size=args.page_size,
        max_events=args.max_events,
        metadata_output_path=Path(args.metadata_output) if args.metadata_output else None,
    )
    if metadata["event_count"] == 0:
        print("[WARN] A megadott időablakban nem volt exportálható alert.")
    print(f"[OK] Exportált Wazuh alert sorok: {metadata['event_count']}")
    print(f"[OK] Kimenet: {metadata['output']}")


if __name__ == "__main__":
    main()
