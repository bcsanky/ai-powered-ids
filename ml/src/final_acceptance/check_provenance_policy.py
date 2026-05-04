from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ml.src.final_acceptance.common import check_row, file_exists_row, has_fail, read_text_optional, write_check_outputs


REQUIRED_FILES = [
    "docs/data_provenance_and_no_fake_measurements.md",
    "docs/repository_clean_state.md",
    "ml/src/repo_hygiene/common.py",
    "ml/src/repo_hygiene/check_no_demo_real_results.py",
    "ml/src/repo_hygiene/create_measurement_provenance.py",
    "ml/src/repo_hygiene/check_tracked_generated_outputs.py",
]
GITIGNORE_PATTERNS = [
    "reports/real_measurement/",
    "reports/live_integration/",
    "reports/thesis_integration/",
    "reports/lab_session/",
    "results/real_comparison/",
    "results/wazuh_real/",
    "results/ae_lab/",
    "results/hybrid_real/",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/final_acceptance")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def contains_any(text: str | None, tokens: list[str]) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(token.lower() in lower for token in tokens)


def run_check(root: Path, output_dir: Path) -> dict:
    rows = [file_exists_row(f"file_{Path(rel).stem}", root / rel, "Policy fájl") for rel in REQUIRED_FILES]
    gitignore = read_text_optional(root / ".gitignore") or ""
    for pattern in GITIGNORE_PATTERNS:
        rows.append(
            check_row(
                f"gitignore_{pattern.strip('/').replace('/', '_')}",
                ".gitignore",
                "PASS" if pattern in gitignore else "FAIL",
                f"ignore szabály megvan: {pattern}" if pattern in gitignore else f"hiányzó ignore szabály: {pattern}",
                "" if pattern in gitignore else "Add hozzá a futási output könyvtárat a .gitignore fájlhoz.",
                ".gitignore",
            )
        )
    for rel_path, tokens in [
        ("examples/lab/README.md", ["demonstrációs", "demo"]),
        ("examples/scoring/README.md", ["demonstrációs", "demo"]),
        ("reports/README.md", ["futási kimenet", "output"]),
        ("results/README.md", ["futási kimenet", "output"]),
    ]:
        text = read_text_optional(root / rel_path)
        rows.append(
            check_row(
                f"doc_{Path(rel_path).stem}_{Path(rel_path).parent.name}",
                "Dokumentáció",
                "PASS" if contains_any(text, tokens) else "FAIL",
                f"{rel_path} jelöli a szerepét" if contains_any(text, tokens) else f"{rel_path} nem tartalmaz elég egyértelmű jelölést",
                "Pontosítsd a dokumentációt demo vagy futási kimenet megjelöléssel.",
                rel_path,
            )
        )
    return write_check_outputs(
        output_dir=output_dir,
        basename="provenance_policy_check",
        title="Provenance és no-demo policy ellenőrzés",
        rows=rows,
    )


def main() -> None:
    args = parse_args()
    result = run_check(Path(args.root), Path(args.output_dir))
    print(f"[OK] Kimenet: {result['markdown']}")
    if has_fail(result["rows"]):
        sys.exit(1)


if __name__ == "__main__":
    main()

