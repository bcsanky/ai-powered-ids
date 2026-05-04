from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ml.src.final_submission_check.common import check_row, read_csv_optional, read_json_optional, write_outputs
from ml.src.repo_hygiene.common import load_provenance, validate_provenance_payload


INPUTS = [
    ("requirement_coverage", "requirement_coverage.csv"),
    ("thesis_structure", "thesis_structure_check.csv"),
    ("no_overclaiming", "no_overclaiming_check.csv"),
    ("submission_artifact_plan", "submission_artifact_plan.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/final_submission_check")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def input_status(path: Path) -> str:
    df = read_csv_optional(path)
    if df is None or df.empty or "status" not in df.columns:
        return "FAIL"
    statuses = set(df["status"].astype(str))
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def derive_readiness(root: Path, output_dir: Path) -> tuple[list[dict[str, str]], str]:
    rows = []
    statuses: dict[str, str] = {}
    for check_id, file_name in INPUTS:
        path = output_dir / file_name
        status = input_status(path)
        statuses[check_id] = status
        rows.append(check_row(f"input_{check_id}", "Readiness input", status, f"{file_name}: {status}", path=path.as_posix()))
    provenance_valid, _ = validate_provenance_payload(load_provenance(root / "reports/real_measurement/measurement_provenance.json"))
    thesis_ready = (root / "reports/thesis_integration/chapter6_results_generated.md").exists()
    final_acceptance = read_json_optional(root / "reports/final_acceptance/release_candidate_readiness.json")
    acceptance_ready = bool(final_acceptance and final_acceptance.get("readiness") in {"READY_FOR_REAL_LAB_RUN", "READY_WITH_WARNINGS"})
    if statuses.get("no_overclaiming") == "FAIL" or statuses.get("requirement_coverage") == "FAIL":
        readiness = "NOT_READY"
    elif not provenance_valid and acceptance_ready:
        readiness = "READY_FOR_REAL_MEASUREMENT"
    elif provenance_valid and thesis_ready and all(status != "FAIL" for status in statuses.values()):
        readiness = "READY_FOR_SUBMISSION_REVIEW" if all(status == "PASS" for status in statuses.values()) else "READY_FOR_THESIS_INTEGRATION"
    elif provenance_valid:
        readiness = "READY_FOR_THESIS_INTEGRATION"
    else:
        readiness = "READY_FOR_REAL_MEASUREMENT" if all(status != "FAIL" for status in statuses.values()) else "NOT_READY"
    rows.append(
        check_row(
            "final_submission_readiness",
            "Összesítés",
            "PASS" if readiness != "NOT_READY" else "FAIL",
            readiness,
            "Ez beadási QA státusz, nem mérési eredmény.",
        )
    )
    return rows, readiness


def run_check(output_dir: Path, root: Path = Path(".")) -> dict[str, Any]:
    rows, readiness = derive_readiness(root, output_dir)
    result = write_outputs(
        output_dir=output_dir,
        basename="final_submission_readiness",
        title="Final submission readiness",
        rows=rows,
        extra_payload={"readiness": readiness},
    )
    md = result["markdown"]
    text = md.read_text(encoding="utf-8")
    text += f"\nVégső beadási QA státusz: **{readiness}**\n"
    if readiness == "READY_FOR_REAL_MEASUREMENT":
        text += "\nA mérnöki pipeline előkészített, de végleges beadási review-hoz verified real-lab provenance szükséges.\n"
    md.write_text(text, encoding="utf-8")
    result["readiness"] = readiness
    return result


def main() -> None:
    args = parse_args()
    result = run_check(Path(args.output_dir), Path(args.root))
    print(f"[OK] Kimenet: {result['markdown']}")
    if result["readiness"] == "NOT_READY":
        sys.exit(1)


if __name__ == "__main__":
    main()

