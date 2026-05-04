from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ml.src.live_smoke.common import check_row, has_fail, run_command, summarize_command_result, write_check_outputs


DRY_RUN_TARGETS = [
    "lab-session-prep",
    "lab-session-after-capture",
    "final-real-measurement-package-with-provenance",
    "final-live-integration",
    "final-thesis-integration",
    "final-submission-check",
    "final-acceptance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/live_smoke")
    parser.add_argument("--makefile", default="Makefile")
    return parser.parse_args()


def run_check(output_dir: Path, makefile: Path = Path("Makefile")) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    if not makefile.exists():
        rows.append(check_row("makefile", "Make dry-run", "FAIL", "Makefile hiányzik", "A workflow ellenőrzéshez szükséges.", makefile.as_posix()))
        return write_check_outputs(
            output_dir=output_dir,
            basename="make_workflow_dry_run",
            title="Makefile workflow dry-run ellenőrzés",
            rows=rows,
            intro="A dry-run ellenőrzés make -n módban fut, ezért mérési parancsokat nem hajt végre.",
        )

    for target in DRY_RUN_TARGETS:
        result = run_command(["make", "-n", target], timeout_seconds=30)
        ok = result.returncode == 0
        rows.append(
            check_row(
                f"make_n_{target}",
                "Make dry-run",
                "PASS" if ok else "FAIL",
                f"make -n {target} sikeres" if ok else f"make -n {target} hibás: {summarize_command_result(result)}",
                "" if ok else "Ellenőrizd a Makefile targetet és változóit.",
                target,
            )
        )

    return write_check_outputs(
        output_dir=output_dir,
        basename="make_workflow_dry_run",
        title="Makefile workflow dry-run ellenőrzés",
        rows=rows,
        intro="A dry-run ellenőrzés make -n módban fut, ezért mérési parancsokat nem hajt végre.",
    )


def main() -> None:
    args = parse_args()
    result = run_check(Path(args.output_dir), Path(args.makefile))
    print(f"[OK] Kimenet: {result['markdown']}")
    if has_fail(result["rows"]):
        sys.exit(1)


if __name__ == "__main__":
    main()

