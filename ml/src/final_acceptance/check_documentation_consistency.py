from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ml.src.final_acceptance.common import check_row, has_fail, read_text_optional, write_check_outputs
from ml.src.repo_hygiene.common import load_provenance, validate_provenance_payload


DOCS = [
    "docs/final_real_measurement_checklist.md",
    "docs/data_provenance_and_no_fake_measurements.md",
    "docs/live_integration_runbook.md",
    "docs/lab_session_orchestration_runbook.md",
    "docs/thesis_integration_runbook.md",
    "docs/thesis_result_artifacts.md",
]
FORBIDDEN_ALWAYS = [
    "bizonyított javulás",
    "éles SOC rendszer",
    "production-ready",
    "Wazuh teljesítménye bizonyítottan javult",
    "real-lab benchmark completed",
]
FORBIDDEN_WITHOUT_PROVENANCE = [
    "valódi lab eredmény elkészült",
    "verified real-lab result",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/final_acceptance")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def line_matches(text: str, token: str) -> list[tuple[int, str]]:
    matches = []
    token_lower = token.lower()
    for index, line in enumerate(text.splitlines(), start=1):
        lower = line.lower()
        if token_lower in lower and "nem " + token_lower not in lower:
            matches.append((index, line.strip()))
    return matches


def run_check(root: Path, output_dir: Path) -> dict:
    provenance = load_provenance(root / "reports/real_measurement/measurement_provenance.json")
    verified, _ = validate_provenance_payload(provenance)
    rows = []
    for rel_path in DOCS:
        path = root / rel_path
        text = read_text_optional(path)
        if text is None:
            rows.append(check_row(f"doc_{Path(rel_path).stem}", "Dokumentáció", "WARN", f"nem található: {rel_path}", path=rel_path))
            continue
        found = []
        for token in FORBIDDEN_ALWAYS:
            found.extend((token, line_no, line) for line_no, line in line_matches(text, token))
        if not verified:
            for token in FORBIDDEN_WITHOUT_PROVENANCE:
                found.extend((token, line_no, line) for line_no, line in line_matches(text, token))
        if found:
            detail = "; ".join(f"{token} @ {line_no}: {line}" for token, line_no, line in found[:5])
            rows.append(
                check_row(
                    f"doc_{Path(rel_path).stem}",
                    "Dokumentáció",
                    "FAIL",
                    detail,
                    "Pontosítsd a megfogalmazást feltételes, provenance-hez kötött állításra.",
                    rel_path,
                )
            )
        else:
            rows.append(check_row(f"doc_{Path(rel_path).stem}", "Dokumentáció", "PASS", "nincs tiltott állítás", path=rel_path))
    return write_check_outputs(
        output_dir=output_dir,
        basename="documentation_consistency_check",
        title="Dokumentációs konzisztencia ellenőrzés",
        rows=rows,
        intro="Az ellenőrzés azt vizsgálja, hogy a dokumentáció ne sugalljon nem létező vagy provenance nélküli real-lab eredményt.",
    )


def main() -> None:
    args = parse_args()
    result = run_check(Path(args.root), Path(args.output_dir))
    print(f"[OK] Kimenet: {result['markdown']}")
    if has_fail(result["rows"]):
        sys.exit(1)


if __name__ == "__main__":
    main()

