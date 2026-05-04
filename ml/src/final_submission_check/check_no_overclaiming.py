from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ml.src.final_submission_check.common import check_row, has_fail, write_outputs


SCAN_PATTERNS = [
    "docs/*.md",
    "reports/thesis_integration/*.md",
    "reports/real_measurement/*.md",
    "reports/live_integration/*.md",
]
FAIL_TOKENS = [
    "production-ready",
    "bizonyítottan javítja",
    "garantált javulás",
    "teljes körű IDS",
    "valós idejű éles rendszer",
    "generated measurement",
    "synthetic real-lab result",
]
CONTEXT_TOKENS = ["laboratóriumi prototípus", "integrációs demonstráció", "a vizsgált lab mérésben", "verified provenance esetén"]
SUSPICIOUS_TOKENS = ["éles SOC rendszer", "fake", "dummy"]
SAFE_CONTEXT = ["nem ", "ne ", "tilt", "elutasít", "no-demo", "no_fake", "nem használ", "nem helyettesít"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/final_submission_check")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def safe_context(line: str) -> bool:
    lower = line.lower()
    return any(token in lower for token in SAFE_CONTEXT) or any(token.lower() in lower for token in CONTEXT_TOKENS)


def scan_file(path: Path) -> list[dict[str, str]]:
    rows = []
    text = path.read_text(encoding="utf-8")
    for line_no, line in enumerate(text.splitlines(), start=1):
        lower = line.lower()
        for token in FAIL_TOKENS:
            if token.lower() in lower:
                rows.append(
                    check_row(
                        f"overclaim_{path.name}_{line_no}",
                        "Túlzó állítás",
                        "FAIL",
                        f"{token}: {line.strip()}",
                        "Pontosítsd óvatos, méréshez kötött megfogalmazásra.",
                        path.as_posix(),
                    )
                )
        for token in SUSPICIOUS_TOKENS:
            if token.lower() in lower:
                rows.append(
                    check_row(
                        f"suspicious_{path.name}_{line_no}",
                        "Túlzó állítás",
                        "WARN" if safe_context(line) else "FAIL",
                        f"{token}: {line.strip()}",
                        "Ellenőrizd, hogy a megfogalmazás nem állít-e éles vagy nem igazolt eredményt.",
                        path.as_posix(),
                    )
                )
    return rows


def run_check(output_dir: Path, root: Path = Path(".")) -> dict:
    rows = []
    for pattern in SCAN_PATTERNS:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                rows.extend(scan_file(path))
    if not rows:
        rows.append(check_row("no_overclaiming", "Túlzó állítás", "PASS", "nem találtam túlzó állítást"))
    return write_outputs(
        output_dir=output_dir,
        basename="no_overclaiming_check",
        title="No-overclaiming ellenőrzés",
        rows=rows,
        intro="Ez az ellenőrzés a túlzó vagy nem igazolt dolgozati állításokat keresi.",
    )


def main() -> None:
    args = parse_args()
    result = run_check(Path(args.output_dir), Path(args.root))
    print(f"[OK] Kimenet: {result['markdown']}")
    if has_fail(result["rows"]):
        sys.exit(1)


if __name__ == "__main__":
    main()

