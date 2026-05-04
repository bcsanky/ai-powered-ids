from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ml.src.submission_bundle.common import (
    bool_from_csv,
    ensure_dir,
    is_runtime_output_path,
    is_safe_relative_path,
    is_sensitive_path,
    load_verified_provenance,
    read_csv_rows,
    read_yaml,
    runtime_candidate_allowed,
    status_from_rows,
    write_csv,
    write_json,
    write_markdown_report,
)


FIELDS = ["check_id", "relative_path", "category", "status", "message", "recommendation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="docs/submission_bundle_policy.yaml")
    parser.add_argument("--output-dir", default="reports/submission_bundle")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def row(check_id: str, relative_path: str, category: str, status: str, message: str, recommendation: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "relative_path": relative_path,
        "category": category,
        "status": status,
        "message": message,
        "recommendation": recommendation,
    }


def validate_candidates(root: Path, policy_path: Path, candidates_path: Path) -> tuple[list[dict[str, Any]], str]:
    policy = read_yaml(policy_path)
    provenance_status, _payload, provenance_errors = load_verified_provenance(root)
    runtime_allowed = runtime_candidate_allowed(provenance_status, policy)
    candidates = read_csv_rows(candidates_path)
    rows: list[dict[str, Any]] = []
    if not candidates:
        rows.append(row("candidate_list", "", "input", "FAIL", "A candidate lista hiányzik vagy üres.", "Futtasd: make submission-candidates"))
        return rows, provenance_status

    for index, candidate in enumerate(candidates, start=1):
        rel = candidate.get("relative_path", "")
        group = candidate.get("group", "")
        include = bool_from_csv(candidate.get("include_candidate", "false"))
        if not include:
            rows.append(row(f"candidate_{index}", rel, group, "SKIP", "Nem include_candidate, ezért nem csomagolási jelölt.", "Nincs teendő."))
            continue
        path = root / rel
        if not is_safe_relative_path(rel):
            rows.append(row(f"candidate_{index}", rel, group, "FAIL", "Abszolút vagy szülő könyvtárra mutató path.", "Csak repo relatív útvonal engedélyezett."))
            continue
        if not path.exists():
            rows.append(row(f"candidate_{index}", rel, group, "FAIL", "A jelölt fájl nem létezik.", "Frissítsd a candidate listát."))
            continue
        if is_sensitive_path(rel, policy):
            rows.append(row(f"candidate_{index}", rel, group, "FAIL", "Tiltott vagy érzékeny fájl lenne a csomagban.", "Vedd ki a policyből vagy hagyd futási mellékletként."))
            continue
        if rel.startswith("examples/") and group != "demo_example":
            rows.append(
                row(
                    f"candidate_{index}",
                    rel,
                    group,
                    "FAIL",
                    "Examples alatti fájl nem demo_example csoportban szerepel.",
                    "Az examples fájlok csak demonstrációs példaként csomagolhatók.",
                )
            )
            continue
        if is_runtime_output_path(rel, policy) and not runtime_allowed:
            rows.append(
                row(
                    f"candidate_{index}",
                    rel,
                    group,
                    "FAIL",
                    "Runtime output verified real_lab provenance nélkül.",
                    "; ".join(provenance_errors) or "Készíts valid provenance fájlt valódi mérés után.",
                )
            )
            continue
        rows.append(row(f"candidate_{index}", rel, group, "PASS", "A candidate fájl csomagolható.", ""))
    return rows, provenance_status


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, provenance_status: str) -> None:
    ensure_dir(output_dir)
    status = status_from_rows([r for r in rows if r["status"] != "SKIP"])
    write_csv(output_dir / "submission_candidate_validation.csv", rows, FIELDS)
    write_markdown_report(
        output_dir / "submission_candidate_validation.md",
        "Submission candidate validáció",
        rows,
        FIELDS,
        "A validáció ellenőrzi, hogy a jelölt lista nem tartalmaz raw, érzékeny vagy provenance nélküli runtime eredményt.",
        status,
    )
    write_json(
        output_dir / "submission_candidate_validation.json",
        {"status": status, "provenance_status": provenance_status, "checks": rows},
    )


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    rows, provenance_status = validate_candidates(
        root,
        root / args.policy,
        output_dir / "submission_candidates.csv",
    )
    write_outputs(rows, output_dir, provenance_status)
    if status_from_rows([r for r in rows if r["status"] != "SKIP"]) == "FAIL":
        raise SystemExit(1)
    print(f"[OK] Candidate validáció: {output_dir / 'submission_candidate_validation.md'}")


if __name__ == "__main__":
    main()
