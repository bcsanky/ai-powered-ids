from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ml.src.final_acceptance.common import check_row, has_fail, write_check_outputs


REQUIRED_DOCS = [
    "docs/real_lab_execution_day_runbook.md",
    "docs/real_lab_scenario_script.md",
    "docs/real_lab_operator_command_sheet.md",
    "docs/real_lab_minimum_acceptance_criteria.md",
    "docs/real_lab_after_action_review_template.md",
    "docs/real_lab_troubleshooting_guide.md",
    "docs/real_lab_one_day_execution_plan.md",
]
REQUIRED_KEYWORDS = [
    "provenance",
    "Wazuh",
    "lab_ground_truth.csv",
    "lab_features.csv",
    "alerts.jsonl",
    "final-real-measurement-package-with-provenance",
    "final-measurement-quality",
    "tilos idegen IP",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/final_acceptance")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def check_docs(
    root: Path,
    output_dir: Path,
    docs: list[str] | None = None,
    keywords: list[str] | None = None,
) -> dict:
    docs = docs or REQUIRED_DOCS
    keywords = keywords or REQUIRED_KEYWORDS
    rows = []
    for doc in docs:
        path = root / doc
        if not path.exists():
            rows.append(
                check_row(
                    f"exists_{Path(doc).stem}",
                    "real-lab runbook docs",
                    "FAIL",
                    f"Hiányzó mérésnapi dokumentum: {doc}",
                    "Hozd létre a dokumentumot a real-lab execution pack részeként.",
                    doc,
                )
            )
            continue
        rows.append(
            check_row(
                f"exists_{Path(doc).stem}",
                "real-lab runbook docs",
                "PASS",
                f"Dokumentum létezik: {doc}",
                "",
                doc,
            )
        )
        text = path.read_text(encoding="utf-8")
        text_lower = text.lower()
        for keyword in keywords:
            ok = keyword.lower() in text_lower
            rows.append(
                check_row(
                    f"keyword_{Path(doc).stem}_{keyword.replace('/', '_').replace(' ', '_')}",
                    "kötelező kulcsszó",
                    "PASS" if ok else "FAIL",
                    f"'{keyword}' {'szerepel' if ok else 'hiányzik'}: {doc}",
                    "" if ok else "Pótold a kötelező biztonsági vagy pipeline hivatkozást.",
                    doc,
                )
            )
    return write_check_outputs(
        output_dir=output_dir,
        basename="real_lab_runbook_docs_check",
        title="Real-lab mérésnapi dokumentáció ellenőrzése",
        rows=rows,
        intro=(
            "Ez az ellenőrzés azt vizsgálja, hogy a real-lab execution pack dokumentumai "
            "léteznek-e, és tartalmazzák-e a provenance, Wazuh, inputfájl, pipeline és safety kulcspontokat."
        ),
    )


def main() -> None:
    args = parse_args()
    result = check_docs(Path(args.root), Path(args.output_dir))
    print(f"[OK] Kimenet: {result['markdown']}")
    if has_fail(result["rows"]):
        sys.exit(1)


if __name__ == "__main__":
    main()

