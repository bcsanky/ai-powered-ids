from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ml.src.final_submission_check.common import (
    check_row,
    evidence_kind,
    has_fail,
    read_yaml,
    resolve_evidence,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", default="docs/final_submission_requirements.yaml")
    parser.add_argument("--output-dir", default="reports/final_submission_check")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def requirement_row(root: Path, requirement: dict[str, Any]) -> dict[str, str]:
    req_id = str(requirement.get("id", "unknown"))
    title = str(requirement.get("title", req_id))
    evidence_items = [str(item) for item in requirement.get("evidence", [])]
    existing: list[str] = []
    runtime_missing: list[str] = []
    repository_missing: list[str] = []
    manual_items: list[str] = []
    for evidence in evidence_items:
        kind = evidence_kind(evidence)
        matches = resolve_evidence(root, evidence)
        if kind == "manual":
            manual_items.append(evidence)
        elif matches:
            existing.extend(path.as_posix() for path in matches)
        elif kind == "runtime":
            runtime_missing.append(evidence)
        else:
            repository_missing.append(evidence)

    if existing:
        status = "PASS" if not repository_missing else "WARN"
        message = f"{title}: {len(existing)} bizonyíték elérhető"
        recommendation = "" if not repository_missing else "A hiányzó repository evidence elemeket kézzel ellenőrizd."
    elif runtime_missing and not repository_missing:
        status = "WARN"
        message = f"{title}: csak valós mérés után előálló runtime evidence hiányzik"
        recommendation = "READY_AFTER_REAL_MEASUREMENT: futtasd a mérési és thesis integration lépéseket."
    elif manual_items and not repository_missing:
        status = "WARN"
        message = f"{title}: kézi Word dokumentum ellenőrzést igényel"
        recommendation = "A feltöltött dolgozatban kézzel ellenőrizd a fejezetet."
    else:
        status = "FAIL"
        message = f"{title}: nincs elérhető bizonyíték"
        recommendation = "Pótold a hiányzó forráskódot vagy dokumentációt."
    path_summary = "; ".join(existing or runtime_missing or repository_missing or manual_items)
    return check_row(req_id, "Feladatlap követelmény", status, message, recommendation, path_summary)


def run_check(requirements_path: Path, output_dir: Path, root: Path = Path(".")) -> dict:
    if not requirements_path.exists():
        rows = [
            check_row(
                "requirements_file",
                "Konfiguráció",
                "FAIL",
                f"hiányzó követelményfájl: {requirements_path}",
                "Hozd létre a final_submission_requirements.yaml fájlt.",
                requirements_path.as_posix(),
            )
        ]
    else:
        payload = read_yaml(requirements_path)
        requirements = payload.get("requirements", [])
        rows = [requirement_row(root, requirement) for requirement in requirements]
        if not requirements:
            rows.append(check_row("requirements_nonempty", "Konfiguráció", "FAIL", "nincs requirement lista"))
    return write_outputs(
        output_dir=output_dir,
        basename="requirement_coverage",
        title="Feladatlap-követelmények lefedettsége",
        rows=rows,
        intro="Ez az ellenőrzés azt vizsgálja, hogy a repository és a futási kimeneti terv lefedi-e a fő dolgozati elvárásokat.",
    )


def main() -> None:
    args = parse_args()
    result = run_check(Path(args.requirements), Path(args.output_dir), Path(args.root))
    print(f"[OK] Kimenet: {result['markdown']}")
    if has_fail(result["rows"]):
        sys.exit(1)


if __name__ == "__main__":
    main()

