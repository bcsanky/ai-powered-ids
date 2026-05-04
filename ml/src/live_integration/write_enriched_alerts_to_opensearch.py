from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--opensearch-url", required=True)
    parser.add_argument("--index-name", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--verify-tls", choices=["true", "false"], default="true")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--metadata-output", default="reports/live_integration/opensearch_write_metadata.json")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó enriched alert JSONL: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Érvénytelen JSONL sor: {line_no}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    if not rows:
        raise ValueError(f"Az enriched alert JSONL üres: {path}")
    return rows


def chunks(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("A batch-size pozitív egész kell legyen.")
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def write_batch(
    *,
    rows: list[dict[str, Any]],
    opensearch_url: str,
    index_name: str,
    username: str,
    password: str,
    verify_tls: bool,
) -> dict[str, Any]:
    body_lines = []
    for row in rows:
        document_id = row.get("integration_event_id")
        action: dict[str, Any] = {"index": {"_index": index_name}}
        if document_id:
            action["index"]["_id"] = str(document_id)
        body_lines.append(json.dumps(action, ensure_ascii=False))
        body_lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    body = "\n".join(body_lines) + "\n"
    response = requests.post(
        opensearch_url.rstrip("/") + "/_bulk",
        auth=(username, password),
        data=body.encode("utf-8"),
        headers={"Content-Type": "application/x-ndjson"},
        verify=verify_tls,
        timeout=30,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"OpenSearch bulk API hiba: HTTP {response.status_code} - {response.text[:500]}")
    return response.json()


def write_enriched_alerts(
    *,
    input_path: Path,
    opensearch_url: str,
    index_name: str,
    username: str,
    password: str,
    verify_tls: bool,
    batch_size: int,
    metadata_output: Path,
) -> dict[str, Any]:
    rows = read_jsonl(input_path)
    total_errors = 0
    batches = chunks(rows, batch_size)
    for batch in batches:
        result = write_batch(
            rows=batch,
            opensearch_url=opensearch_url,
            index_name=index_name,
            username=username,
            password=password,
            verify_tls=verify_tls,
        )
        if result.get("errors"):
            items = result.get("items", [])
            total_errors += sum(1 for item in items if item.get("index", {}).get("error"))

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(input_path),
        "opensearch_url": opensearch_url,
        "index_name": index_name,
        "document_count": len(rows),
        "batch_size": batch_size,
        "batch_count": len(batches),
        "verify_tls": verify_tls,
        "bulk_error_count": total_errors,
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    if not args.password:
        raise ValueError("OpenSearch jelszó megadása szükséges.")
    metadata = write_enriched_alerts(
        input_path=Path(args.input),
        opensearch_url=args.opensearch_url,
        index_name=args.index_name,
        username=args.username,
        password=args.password,
        verify_tls=args.verify_tls == "true",
        batch_size=args.batch_size,
        metadata_output=Path(args.metadata_output),
    )
    print(f"[OK] OpenSearch dokumentumok: {metadata['document_count']}")
    print(f"[OK] Metadata: {args.metadata_output}")


if __name__ == "__main__":
    main()

