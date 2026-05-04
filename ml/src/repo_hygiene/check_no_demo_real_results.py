from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ml.src.repo_hygiene.common import PROVENANCE_PATH, load_provenance, validate_provenance_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="reports/repo_hygiene")
    return parser.parse_args()


def write_outputs(status: str, messages: list[str], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "no_demo_real_results_check.md"
    json_path = output_dir / "no_demo_real_results_check.json"
    lines = [
        "# No-demo real-lab guard",
        "",
        f"Státusz: **{status}**",
        "",
        *[f"- {message}" for message in messages],
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {"created_at": datetime.now().isoformat(timespec="seconds"), "status": status, "messages": messages}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"markdown": md_path, "json": json_path}


def check_no_demo_real_results(root: Path, output_dir: Path) -> dict[str, Any]:
    comparison_path = root / "results/real_comparison/metrics_comparison.csv"
    if not comparison_path.exists():
        outputs = write_outputs("PASS", ["Nincs ellenőrizhető real-lab eredmény."], output_dir)
        return {"status": "PASS", "outputs": outputs, "messages": ["Nincs ellenőrizhető real-lab eredmény."]}

    provenance_path = root / PROVENANCE_PATH
    provenance = load_provenance(provenance_path)
    valid, errors = validate_provenance_payload(provenance)
    if not valid:
        messages = ["Real-lab comparison létezik, de a provenance nem érvényes.", *errors]
        outputs = write_outputs("FAIL", messages, output_dir)
        return {"status": "FAIL", "outputs": outputs, "messages": messages}
    outputs = write_outputs("PASS", ["A real-lab eredmény provenance alapján nem demo/sample/fixture inputból származik."], output_dir)
    return {"status": "PASS", "outputs": outputs, "messages": ["Provenance rendben."]}


def main() -> None:
    args = parse_args()
    result = check_no_demo_real_results(Path(args.root), Path(args.output_dir))
    for path in result["outputs"].values():
        print(f"[OK] Kimenet: {path}")
    if result["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
