from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ml.src.final_acceptance.common import check_row, read_csv_optional, write_check_outputs


INPUTS = [
    ("make_targets", "make_targets_check.csv"),
    ("failure_modes", "failure_modes_check.csv"),
    ("provenance_policy", "provenance_policy_check.csv"),
    ("documentation", "documentation_consistency_check.csv"),
    ("real_lab_runbook_docs", "real_lab_runbook_docs_check.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/final_acceptance")
    return parser.parse_args()


def summarize_checks(output_dir: Path) -> tuple[list[dict[str, str]], str]:
    rows = []
    any_warn = False
    any_fail = False
    for check_id, file_name in INPUTS:
        path = output_dir / file_name
        df = read_csv_optional(path)
        if df is None or df.empty:
            rows.append(
                check_row(
                    f"input_{check_id}",
                    "Readiness input",
                    "FAIL",
                    f"hiányzó vagy üres ellenőrzési kimenet: {file_name}",
                    "Futtasd a kapcsolódó final-acceptance célpontot.",
                    path.as_posix(),
                )
            )
            any_fail = True
            continue
        statuses = set(df["status"].astype(str)) if "status" in df.columns else {"FAIL"}
        if "FAIL" in statuses:
            status = "FAIL"
            any_fail = True
        elif "WARN" in statuses:
            status = "WARN"
            any_warn = True
        else:
            status = "PASS"
        rows.append(check_row(f"input_{check_id}", "Readiness input", status, f"{file_name}: {status}", path=path.as_posix()))
    if any_fail:
        readiness = "NOT_READY"
    elif any_warn:
        readiness = "READY_WITH_WARNINGS"
    else:
        readiness = "READY_FOR_REAL_LAB_RUN"
    rows.append(
        check_row(
            "release_candidate_status",
            "Összesítés",
            "PASS" if readiness == "READY_FOR_REAL_LAB_RUN" else "WARN" if readiness == "READY_WITH_WARNINGS" else "FAIL",
            readiness,
            "Ez a státusz a valós lab mérés előtti készültséget jelzi, nem dolgozati kész állapotot.",
        )
    )
    return rows, readiness


def run_check(output_dir: Path) -> dict[str, Any]:
    rows, readiness = summarize_checks(output_dir)
    result = write_check_outputs(
        output_dir=output_dir,
        basename="release_candidate_readiness",
        title="Release-candidate readiness",
        rows=rows,
        intro="Ez az összegzés azt jelzi, hogy a rendszer készen áll-e a tényleges real-lab mérés futtatására.",
        extra_payload={"readiness": readiness},
    )
    md_path = result["markdown"]
    text = md_path.read_text(encoding="utf-8")
    text += f"\nVégső release-candidate státusz: **{readiness}**\n"
    text += "\nEz nem azt jelenti, hogy a dolgozat kész; csak a mérési workflow előkészítettségét minősíti.\n"
    md_path.write_text(text, encoding="utf-8")
    result["readiness"] = readiness
    return result


def main() -> None:
    args = parse_args()
    result = run_check(Path(args.output_dir))
    print(f"[OK] Kimenet: {result['markdown']}")
    if result["readiness"] == "NOT_READY":
        sys.exit(1)


if __name__ == "__main__":
    main()
