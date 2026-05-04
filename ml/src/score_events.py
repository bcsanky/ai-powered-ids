from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.scoring_runtime import AEScorer, REQUIRED_FEATURES, result_to_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-root", default="artifacts/final/final-ae-minimal-v1")
    parser.add_argument("--preprocess", default="data/processed/final/ae_minimal/preprocess.pkl")
    parser.add_argument("--thresholds-auto", action="store_true")
    parser.add_argument("--model-version", default="final-ae-minimal-v1")
    return parser.parse_args()


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó input fájl: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        events = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events
    if suffix == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    raise ValueError("Támogatott input formátumok: .jsonl, .csv")


def split_event(record: dict[str, Any]) -> tuple[str, dict[str, Any], bool, int]:
    event_id = str(record.get("event_id", ""))
    if not event_id:
        raise ValueError("Minden eseményhez szükséges event_id.")

    if isinstance(record.get("features"), dict):
        features = dict(record["features"])
    else:
        features = {name: record.get(name) for name in REQUIRED_FEATURES}

    missing = [name for name in REQUIRED_FEATURES if features.get(name) is None]
    if missing:
        raise ValueError(f"Hiányzó bemeneti mezők az {event_id} eseményben: {missing}")

    rule_flag = bool(record.get("rule_flag", False))
    rule_level = int(record.get("rule_level", 0) or 0)
    return event_id, features, rule_flag, rule_level


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    if suffix == ".csv":
        pd.DataFrame(rows).to_csv(path, index=False)
        return
    raise ValueError("Támogatott output formátumok: .jsonl, .csv")


def score_events(
    *,
    input_path: Path,
    output_path: Path,
    model_root: Path,
    preprocess_path: Path,
    model_version: str,
) -> list[dict[str, Any]]:
    scorer = AEScorer(
        model_root=model_root,
        preprocess_path=preprocess_path,
        model_version=model_version,
    )
    scorer.load()

    rows = []
    for record in read_events(input_path):
        event_id, features, rule_flag, rule_level = split_event(record)
        result = scorer.score_event(
            event_id=event_id,
            features=features,
            rule_flag=rule_flag,
            rule_level=rule_level,
        )
        rows.append(result_to_dict(result))

    write_results(output_path, rows)
    return rows


def main() -> None:
    args = parse_args()
    rows = score_events(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model_root=Path(args.model_root),
        preprocess_path=Path(args.preprocess),
        model_version=args.model_version,
    )
    print(f"[OK] Pontozott események: {len(rows)}")
    print(f"[OK] Kimeneti állomány: {args.output}")


if __name__ == "__main__":
    main()
