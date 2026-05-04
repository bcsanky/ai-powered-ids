from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any


IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
HOST_COLUMNS = {"agent_name", "host", "hostname", "agent", "agent.name"}
TEXT_SUFFIXES = {".md", ".html", ".json", ".jsonl", ".csv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mapping-output", required=True)
    return parser.parse_args()


class Redactor:
    def __init__(self) -> None:
        self.ip_map: dict[str, str] = {}
        self.host_map: dict[str, str] = {}

    def token_for_ip(self, value: str) -> str:
        if value not in self.ip_map:
            self.ip_map[value] = f"IP_{len(self.ip_map) + 1:03d}"
        return self.ip_map[value]

    def token_for_host(self, value: str) -> str:
        if value not in self.host_map:
            self.host_map[value] = f"HOST_{len(self.host_map) + 1:03d}"
        return self.host_map[value]

    def redact_text(self, text: str) -> str:
        return IP_RE.sub(lambda match: self.token_for_ip(match.group(0)), text)

    def redact_json_value(self, value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {k: self.redact_json_value(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [self.redact_json_value(item, key) for item in value]
        if isinstance(value, str):
            redacted = self.redact_text(value)
            if key.lower() in HOST_COLUMNS and redacted:
                return self.token_for_host(redacted)
            return redacted
        return value

    def mapping(self) -> dict[str, dict[str, str]]:
        return {"ip": self.ip_map, "host": self.host_map}


def redact_csv(input_path: Path, output_path: Path, redactor: Redactor) -> None:
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        fieldnames = reader.fieldnames or []
        for row in reader:
            new_row = {}
            for key, value in row.items():
                text = redactor.redact_text(value or "")
                if key.lower() in HOST_COLUMNS and text:
                    text = redactor.token_for_host(text)
                new_row[key] = text
            rows.append(new_row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def redact_json_file(input_path: Path, output_path: Path, redactor: Redactor) -> None:
    if input_path.suffix.lower() == ".jsonl":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
            for line in src:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                dst.write(json.dumps(redactor.redact_json_value(payload), ensure_ascii=False, sort_keys=True) + "\n")
        return
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(redactor.redact_json_value(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def redact_outputs(input_dir: Path, output_dir: Path, mapping_output: Path) -> dict[str, Any]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Hiányzó input könyvtár: {input_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    redactor = Redactor()
    copied = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(input_dir)
        output_path = output_dir / rel
        suffix = path.suffix.lower()
        if suffix not in TEXT_SUFFIXES:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output_path)
        elif suffix == ".csv":
            redact_csv(path, output_path, redactor)
        elif suffix in {".json", ".jsonl"}:
            redact_json_file(path, output_path, redactor)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(redactor.redact_text(path.read_text(encoding="utf-8")), encoding="utf-8")
        copied.append(output_path)

    mapping_output.parent.mkdir(parents=True, exist_ok=True)
    mapping_output.write_text(json.dumps(redactor.mapping(), indent=2, ensure_ascii=False), encoding="utf-8")
    return {"files": copied, "mapping": mapping_output, "mapping_data": redactor.mapping()}


def main() -> None:
    args = parse_args()
    result = redact_outputs(Path(args.input_dir), Path(args.output_dir), Path(args.mapping_output))
    print(f"[OK] Anonimizált fájlok: {len(result['files'])}")
    print(f"[OK] Mapping: {result['mapping']}")


if __name__ == "__main__":
    main()
