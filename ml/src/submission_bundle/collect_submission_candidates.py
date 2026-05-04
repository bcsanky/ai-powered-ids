from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ml.src.submission_bundle.common import (
    ensure_dir,
    is_runtime_output_path,
    is_sensitive_path,
    iter_existing_files,
    load_verified_provenance,
    markdown_table,
    normalize_relative,
    read_yaml,
    runtime_candidate_allowed,
    write_csv,
    write_json,
    write_markdown_report,
)


FIELDS = [
    "relative_path",
    "group",
    "exists",
    "include_candidate",
    "requires_provenance",
    "reason",
    "warning",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="docs/submission_bundle_policy.yaml")
    parser.add_argument("--output-dir", default="reports/submission_bundle")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def example_readme_warning(root: Path, rel_path: str) -> str:
    if rel_path.startswith("examples/lab/"):
        readme = root / "examples/lab/README.md"
    elif rel_path.startswith("examples/scoring/"):
        readme = root / "examples/scoring/README.md"
    else:
        return ""
    if not readme.exists():
        return "Demo README hiányzik."
    text = readme.read_text(encoding="utf-8").lower()
    if "demo" not in text and "demonstráció" not in text:
        return "A README nem jelöli egyértelműen demo inputként."
    return ""


def candidate_row(
    *,
    relative_path: str,
    group: str,
    exists: bool,
    include_candidate: bool,
    requires_provenance: bool,
    reason: str,
    warning: str = "",
) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "group": group,
        "exists": exists,
        "include_candidate": include_candidate,
        "requires_provenance": requires_provenance,
        "reason": reason,
        "warning": warning,
    }


def collect_candidates(root: Path, policy_path: Path) -> tuple[list[dict[str, Any]], str, list[str]]:
    policy = read_yaml(policy_path)
    provenance_status, _payload, provenance_errors = load_verified_provenance(root)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for group, patterns in policy.get("include_groups", {}).items():
        output_group = "demo_example" if group == "examples_demo" else group
        for pattern in patterns:
            for path in iter_existing_files(root, pattern):
                rel = normalize_relative(path, root)
                if rel in seen:
                    continue
                seen.add(rel)
                sensitive = is_sensitive_path(rel, policy)
                include = not sensitive
                reason = "Engedélyezett forrás/config/dokumentáció/sablon fájl."
                warning = ""
                if output_group == "demo_example":
                    reason = "Verziózott demonstrációs példa; nem mérési eredmény."
                    warning = example_readme_warning(root, rel)
                if sensitive:
                    reason = "Policy alapján tiltott vagy érzékeny fájl."
                    warning = "Nem kerülhet beadási csomagba."
                rows.append(
                    candidate_row(
                        relative_path=rel,
                        group=output_group,
                        exists=path.exists(),
                        include_candidate=include,
                        requires_provenance=False,
                        reason=reason,
                        warning=warning,
                    )
                )

    runtime_allowed = runtime_candidate_allowed(provenance_status, policy)
    for pattern in policy.get("runtime_outputs_allowed_if_verified", []):
        for path in iter_existing_files(root, pattern):
            rel = normalize_relative(path, root)
            if rel in seen:
                continue
            seen.add(rel)
            sensitive = is_sensitive_path(rel, policy)
            include = runtime_allowed and not sensitive and is_runtime_output_path(rel, policy)
            if include:
                reason = "Verified real_lab provenance mellett engedélyezett runtime riport vagy eredmény."
                warning = ""
            elif sensitive:
                reason = "Runtime fájl, de érzékeny vagy tiltott mintára illeszkedik."
                warning = "Nem kerülhet beadási csomagba."
            else:
                reason = "Runtime output csak verified real_lab provenance mellett csomagolható."
                warning = "; ".join(provenance_errors) or "Provenance hiányzik vagy nem valid."
            rows.append(
                candidate_row(
                    relative_path=rel,
                    group="runtime_output",
                    exists=path.exists(),
                    include_candidate=include,
                    requires_provenance=True,
                    reason=reason,
                    warning=warning,
                )
            )

    return sorted(rows, key=lambda row: str(row["relative_path"])), provenance_status, provenance_errors


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, provenance_status: str, provenance_errors: list[str]) -> None:
    ensure_dir(output_dir)
    csv_path = output_dir / "submission_candidates.csv"
    md_path = output_dir / "submission_candidates.md"
    json_path = output_dir / "submission_candidates.json"
    write_csv(csv_path, rows, FIELDS)
    intro = (
        "A lista policy alapján készült. A `demo_example` csoport kizárólag demonstrációs bemenet, "
        "nem mérési eredmény. Runtime eredmény csak verified real_lab provenance mellett lehet jelölt."
    )
    write_markdown_report(md_path, "Submission bundle candidate fájlok", rows, FIELDS, intro)
    write_json(
        json_path,
        {
            "provenance_status": provenance_status,
            "provenance_errors": provenance_errors,
            "candidate_count": len(rows),
            "include_candidate_count": sum(1 for row in rows if row["include_candidate"]),
            "rows": rows,
        },
    )


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows, provenance_status, provenance_errors = collect_candidates(root, root / args.policy)
    write_outputs(rows, Path(args.output_dir), provenance_status, provenance_errors)
    print(markdown_table(rows[:5], FIELDS))
    print(f"[OK] Candidate lista: {args.output_dir}/submission_candidates.csv")


if __name__ == "__main__":
    main()

