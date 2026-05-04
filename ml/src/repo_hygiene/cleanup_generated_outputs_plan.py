from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ml.src.repo_hygiene.audit_generated_artifacts import audit


ACTION_BY_POLICY = {
    "keep_tracked": "keep",
    "should_ignore": "ignore_future",
    "should_remove_from_git": "remove_from_git",
    "real_lab_only": "keep",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/repo_hygiene")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def suggested_action(row: pd.Series) -> str:
    path = str(row["relative_path"])
    category = str(row["category"])
    if path.startswith("examples/lab/") or path.startswith("examples/scoring/"):
        return "keep_but_mark_demo"
    if path.startswith("templates/lab/"):
        return "keep"
    if path.startswith("reports/lab/"):
        return "keep_but_mark_demo"
    if category in {"generated_offline_result", "unknown_generated"}:
        return "remove_from_git"
    return ACTION_BY_POLICY.get(str(row["tracked_policy"]), "keep")


def create_cleanup_plan(root: Path, output_dir: Path) -> dict[str, Path]:
    audit_outputs = audit(root, output_dir)
    audit_df = pd.read_csv(audit_outputs["csv"])
    if audit_df.empty:
        plan = pd.DataFrame(columns=["relative_path", "suggested_action", "reason"])
    else:
        plan = audit_df[["relative_path", "category", "tracked_policy", "reason"]].copy()
        plan["suggested_action"] = plan.apply(suggested_action, axis=1)
        plan = plan[["relative_path", "suggested_action", "reason", "category", "tracked_policy"]]
    csv_path = output_dir / "cleanup_plan.csv"
    md_path = output_dir / "cleanup_plan.md"
    plan.to_csv(csv_path, index=False)
    write_markdown(plan, md_path)
    return {"csv": csv_path, "markdown": md_path}


def write_markdown(df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Generált és demo kimenetek tisztítási terve",
        "",
        "A terv nem töröl fájlokat, csak javaslatot ad a verziókezelési kezelésre.",
        "",
        "| fájl | javasolt művelet | indoklás | kategória | policy |",
        "|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['relative_path']} | {row['suggested_action']} | {row['reason']} | "
            f"{row['category']} | {row['tracked_policy']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    outputs = create_cleanup_plan(Path(args.root), Path(args.output_dir))
    for path in outputs.values():
        print(f"[OK] Kimenet: {path}")


if __name__ == "__main__":
    main()
