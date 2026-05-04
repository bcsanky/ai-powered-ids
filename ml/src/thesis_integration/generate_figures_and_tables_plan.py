from __future__ import annotations

import argparse
from pathlib import Path

from ml.src.thesis_integration.common import ensure_output_dir, provenance_status, write_rows_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/thesis_integration")
    parser.add_argument("--provenance", default="reports/real_measurement/measurement_provenance.json")
    return parser.parse_args()


def availability(path: Path, verified: bool) -> tuple[str, str]:
    if not path.exists():
        return "no", "hiányzó forrásfájl"
    return ("yes", "verified real-lab provenance") if verified else ("with_limitations", "provenance hiányzik vagy nem verified")


def plan_rows(provenance_path: Path) -> list[dict[str, str]]:
    verified = provenance_status(provenance_path) == "verified_real_lab"
    items = [
        ("table", "Real-lab konfigurációk összehasonlító metrikái", "6. fejezet", "results/real_comparison/metrics_comparison.csv"),
        ("table", "Wazuh-only baseline metrikái", "6. fejezet", "results/wazuh_real/metrics_summary.csv"),
        ("table", "AE-Minimal lab metrikái", "6. fejezet", "results/ae_lab/metrics_summary.csv"),
        ("table", "Hibrid stratégiák metrikái", "6. fejezet", "results/hybrid_real/metrics_summary.csv"),
        ("figure", "Precision, recall és F1 összehasonlítás", "6. fejezet", "results/real_comparison/fig_precision_recall_f1.png"),
        ("figure", "Hamis pozitív arány összehasonlítás", "6. fejezet", "results/real_comparison/fig_false_positive_rate.png"),
        ("figure", "Riasztásszám összehasonlítás", "6. fejezet", "results/real_comparison/fig_alert_count.png"),
        ("figure", "Átlagos TTD összehasonlítás", "6. fejezet", "results/real_comparison/fig_mean_ttd.png"),
        ("table", "Live integration dashboard összefoglaló", "5. fejezet", "reports/live_integration/dashboard_cards.csv"),
        ("table", "Mérési manifest kivonata", "Melléklet", "reports/real_measurement/measurement_manifest.csv"),
    ]
    rows = []
    for kind, name, section, source in items:
        include, note = availability(Path(source), verified)
        rows.append(
            {
                "type": kind,
                "name": name,
                "thesis_section": section,
                "source_file": source,
                "provenance_status": provenance_status(provenance_path),
                "can_include": include,
                "note": note,
            }
        )
    return rows


def generate_plan(output_dir: Path, provenance_path: Path) -> dict[str, Path]:
    output_dir = ensure_output_dir(output_dir)
    rows = plan_rows(provenance_path)
    figure_rows = [row for row in rows if row["type"] == "figure"]
    table_rows = [row for row in rows if row["type"] == "table"]
    outputs = {
        "figures": output_dir / "figures_plan.md",
        "tables": output_dir / "tables_plan.md",
        "manifest": output_dir / "figures_and_tables_manifest.csv",
    }
    for path, title, selected in [
        (outputs["figures"], "# Ábraterv", figure_rows),
        (outputs["tables"], "# Táblázatterv", table_rows),
    ]:
        lines = [
            title,
            "",
            "| Név | Fejezet | Forrásfájl | Provenance státusz | Beemelhető | Megjegyzés |",
            "|---|---|---|---|---|---|",
        ]
        for row in selected:
            lines.append(
                f"| {row['name']} | {row['thesis_section']} | {row['source_file']} | {row['provenance_status']} | {row['can_include']} | {row['note']} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_rows_csv(outputs["manifest"], rows, ["type", "name", "thesis_section", "source_file", "provenance_status", "can_include", "note"])
    return outputs


def main() -> None:
    args = parse_args()
    outputs = generate_plan(Path(args.output_dir), Path(args.provenance))
    for path in outputs.values():
        print(f"[OK] Kimenet: {path}")


if __name__ == "__main__":
    main()

