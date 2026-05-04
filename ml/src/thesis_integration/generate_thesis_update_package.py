from __future__ import annotations

import argparse
from pathlib import Path

from ml.src.thesis_integration.common import ensure_output_dir, write_rows_csv


GENERATED_FILES = [
    ("chapter5_implementation_generated.md", "5. fejezet", "Implementációs szöveg"),
    ("chapter6_results_generated.md", "6. fejezet", "Eredmények és értékelés"),
    ("chapter6_tables.md", "6. fejezet", "Metrikatáblák"),
    ("chapter6_limitations.md", "6. fejezet", "Korlátok"),
    ("chapter6_research_question_answer.md", "6. fejezet", "Kutatási kérdésre adott válasz"),
    ("chapter7_osszegzes_generated.md", "7. fejezet", "Magyar összegzés"),
    ("chapter8_summary_generated.md", "8. fejezet", "Angol Summary"),
    ("abstract_hu_generated.md", "Absztrakt", "Magyar absztrakt"),
    ("abstract_en_generated.md", "Abstract", "Angol abstract"),
    ("figures_plan.md", "Ábrajegyzék", "Ábraterv"),
    ("tables_plan.md", "Táblázatjegyzék", "Táblázatterv"),
    ("appendix_plan.md", "Melléklet", "Mellékletterv"),
    ("defense_questions_generated.md", "Védés", "Kérdés-válasz segédlet"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/thesis_integration")
    return parser.parse_args()


def generate_package(output_dir: Path) -> dict[str, Path]:
    output_dir = ensure_output_dir(output_dir)
    rows = []
    for file_name, section, purpose in GENERATED_FILES:
        path = output_dir / file_name
        rows.append(
            {
                "generated_file": path.as_posix(),
                "target_section": section,
                "purpose": purpose,
                "status": "available" if path.exists() else "missing",
                "manual_check": "szükséges",
            }
        )
    lines = [
        "# Dolgozati frissítési csomag",
        "",
        "A lista azt mutatja, hogy a futási kimenetek melyik Word fejezethez használhatók. A beillesztés előtt minden metrikát, hivatkozást és ábraszámot kézzel ellenőrizni kell.",
        "",
        "| Fájl | Célfejezet | Szerep | Státusz | Kézi ellenőrzés |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['generated_file']} | {row['target_section']} | {row['purpose']} | {row['status']} | {row['manual_check']} |"
        )
    checklist = """# Dolgozati frissítési ellenőrzőlista

- Hivatkozások ellenőrzése.
- Ábraszámok frissítése.
- Táblázatszámok frissítése.
- Word tartalomjegyzék frissítése.
- Ábrajegyzék és táblázatjegyzék frissítése.
- IEEE hivatkozási sorrend ellenőrzése.
- Rövidítések jegyzékének frissítése.
- Absztrakt és Abstract terjedelmének ellenőrzése.
- A provenance státusz ellenőrzése a végleges PDF előtt.
- Nincs placeholder a végleges dokumentumban.
"""
    outputs = {
        "package": output_dir / "thesis_update_package.md",
        "manifest": output_dir / "thesis_update_manifest.csv",
        "checklist": output_dir / "thesis_update_checklist.md",
    }
    outputs["package"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_rows_csv(outputs["manifest"], rows, ["generated_file", "target_section", "purpose", "status", "manual_check"])
    outputs["checklist"].write_text(checklist, encoding="utf-8")
    return outputs


def main() -> None:
    args = parse_args()
    outputs = generate_package(Path(args.output_dir))
    for path in outputs.values():
        print(f"[OK] Kimenet: {path}")


if __name__ == "__main__":
    main()

