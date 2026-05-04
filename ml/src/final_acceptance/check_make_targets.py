from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ml.src.final_acceptance.common import (
    check_row,
    command_sequence_contains,
    has_fail,
    makefile_target_body,
    makefile_targets,
    write_check_outputs,
)


REQUIRED_TARGETS = [
    "final-validate",
    "repo-hygiene-check",
    "check-tracked-generated-outputs",
    "check-no-demo-real-results",
    "lab-session-prep",
    "lab-session-after-capture",
    "lab-session-after-results",
    "final-real-measurement-package-with-provenance",
    "final-live-integration",
    "final-thesis-integration",
    "real-measurement-provenance",
    "real-measurement-postrun-qa",
    "thesis-check-inputs",
    "thesis-generate-chapter5",
    "thesis-generate-chapter6",
    "final-acceptance-runbook-docs",
]
EXPECTED_FINAL_ACCEPTANCE_SEQUENCE = [
    "final-acceptance-make-targets",
    "final-acceptance-failure-modes",
    "final-acceptance-provenance-policy",
    "final-acceptance-docs",
    "final-acceptance-runbook-docs",
    "repo-hygiene-check",
    "final-acceptance-readiness",
    "final-acceptance-brief",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/final_acceptance")
    parser.add_argument("--makefile", default="Makefile")
    return parser.parse_args()


def run_check(makefile_path: Path, output_dir: Path) -> dict:
    targets = makefile_targets(makefile_path)
    rows = []
    for target in REQUIRED_TARGETS:
        rows.append(
            check_row(
                f"target_{target}",
                "Makefile target",
                "PASS" if target in targets else "FAIL",
                f"target elérhető: {target}" if target in targets else f"hiányzó target: {target}",
                "" if target in targets else "Pótold a Makefile cél definícióját.",
                makefile_path.as_posix(),
            )
        )
    if "final-acceptance" in targets:
        body = makefile_target_body(makefile_path, "final-acceptance")
        ok = command_sequence_contains(body, EXPECTED_FINAL_ACCEPTANCE_SEQUENCE)
        rows.append(
            check_row(
                "target_final_acceptance_order",
                "Makefile sorrend",
                "PASS" if ok else "FAIL",
                "final-acceptance sorrend rendben" if ok else "final-acceptance sorrend hiányos vagy eltér",
                "A release-candidate cél futtassa az összes acceptance lépést a dokumentált sorrendben.",
                makefile_path.as_posix(),
            )
        )
    else:
        rows.append(
            check_row(
                "target_final_acceptance",
                "Makefile target",
                "FAIL",
                "hiányzó target: final-acceptance",
                "Add hozzá a végső acceptance célpontot.",
                makefile_path.as_posix(),
            )
        )
    return write_check_outputs(
        output_dir=output_dir,
        basename="make_targets_check",
        title="Makefile target ellenőrzés",
        rows=rows,
        intro="Ez az ellenőrzés a beadás előtti mérnöki workflow célpontjainak meglétét vizsgálja.",
    )


def main() -> None:
    args = parse_args()
    result = run_check(Path(args.makefile), Path(args.output_dir))
    print(f"[OK] Kimenet: {result['markdown']}")
    if has_fail(result["rows"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
