from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.repo_hygiene.common import load_provenance, validate_provenance_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/live_integration")
    parser.add_argument("--provenance", default="reports/real_measurement/measurement_provenance.json")
    parser.add_argument("--dashboard-summary", default="")
    return parser.parse_args()


def read_one_row_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Üres CSV: {path}")
    return df.iloc[0].to_dict()


def read_readiness(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"readiness": "NOT_READY", "summary": {}, "checks": []}
    return json.loads(path.read_text(encoding="utf-8"))


def provenance_section(provenance_path: Path) -> list[str]:
    provenance = load_provenance(provenance_path)
    valid, errors = validate_provenance_payload(provenance)
    lines = ["## Adateredet és provenance", ""]
    if provenance is None:
        lines.extend(
            [
                "A live integration kimenethez nem áll rendelkezésre provenance fájl. Emiatt az eredmény nem használható végleges real-lab bizonyítékként, csak a feldolgozási lánc működésének korlátozott ellenőrzéseként.",
                "",
            ]
        )
        return lines
    lines.extend(
        [
            f"- measurement_source: `{provenance.get('measurement_source', 'nincs adat')}`",
            f"- ground_truth_path: `{provenance.get('ground_truth_path', 'nincs adat')}`",
            f"- ground_truth_sha256: `{provenance.get('ground_truth_sha256', 'nincs adat')}`",
            f"- lab_features_path: `{provenance.get('lab_features_path', 'nincs adat')}`",
            f"- lab_features_sha256: `{provenance.get('lab_features_sha256', 'nincs adat')}`",
            f"- wazuh_alerts_path: `{provenance.get('wazuh_alerts_path', 'nincs adat')}`",
            f"- wazuh_alerts_sha256: `{provenance.get('wazuh_alerts_sha256', 'nincs adat')}`",
            "",
        ]
    )
    if valid:
        lines.append("A provenance alapján az inputok verified real-lab mérési lánchoz kapcsolhatók.")
    else:
        lines.append(
            "A provenance nem teljes vagy hibás, ezért a kimenet nem használható végleges real-lab bizonyítékként. Hibák: "
            + "; ".join(errors)
        )
    lines.append("")
    return lines


def metric_line(summary: dict[str, Any], key: str, label: str) -> str:
    return f"- {label}: {summary.get(key, 'nincs adat')}"


def build_thesis_section(
    summary: dict[str, Any],
    readiness: dict[str, Any],
    provenance_path: Path,
    dashboard_summary_path: Path | None = None,
) -> str:
    status = readiness.get("readiness", "NOT_READY")
    lines = [
        "# End-to-end Wazuh+AE integrációs demonstráció",
        "",
        "## Cél",
        "",
        "Az integrációs réteg célja annak bemutatása, hogy a Wazuh alert exportból származó események összekapcsolhatók a lab feature adatokkal, majd az AE-Minimal modell pontozásával és hibrid prioritási logikával dashboard-ready kimenetté alakíthatók.",
        "",
        *provenance_section(provenance_path),
        "## Feldolgozási lépések",
        "",
        "1. A Wazuh alert JSONL, JSON vagy normalizált CSV bemenet beolvasása.",
        "2. Event_id alapú, majd időablak és IP-cím alapú illesztés a lab ground truth és feature állományokhoz.",
        "3. AE-Minimal scoring futtatása a megtalált feature sorokon.",
        "4. Hibrid kockázati döntés képzése Wazuh és ML jelzés alapján.",
        "5. Enriched alert CSV/JSONL és dashboard payload előállítása.",
        "",
        "## Fő integrációs mutatók",
        "",
        metric_line(summary, "total_alerts", "Összes alert"),
        metric_line(summary, "scored_alerts", "ML pontszámmal ellátott alert"),
        metric_line(summary, "unmatched_alerts", "Feature mapping nélkül maradt alert"),
        metric_line(summary, "ml_positive_count", "ML pozitív döntés"),
        metric_line(summary, "wazuh_positive_count", "Wazuh pozitív jelzés"),
        metric_line(summary, "hybrid_positive_count", "Hibrid pozitív döntés"),
        metric_line(summary, "critical_count", "Kritikus kockázati szint"),
        metric_line(summary, "high_count", "Magas kockázati szint"),
        "",
        "## Értelmezés",
        "",
        f"A validációs státusz: `{status}`. A kimenet integrációs bizonyíték, vagyis azt igazolja, hogy a Wazuh alert feldolgozás, az ML scoring és a hibrid prioritás technikailag összekapcsolható. Ez nem önálló benchmark, és nem helyettesíti a Wazuh-only, AE-only és hibrid konfigurációk metrikai összehasonlítását.",
        "",
        "## Korlátok",
        "",
        "- Az integrációs kimenet nem éles üzemi SOC-rendszer teljesítményét méri.",
        "- Az ML scoring csak azoknál az alert eseményeknél értelmezhető, amelyekhez megbízható feature mapping található.",
        "- Az unmatched arányt a dolgozatban közölni kell, mert közvetlenül befolyásolja a dashboard értelmezhetőségét.",
        "- A hibrid prioritás mérnöki döntési szabály, nem önmagában vett detektálási benchmark.",
    ]
    if dashboard_summary_path and dashboard_summary_path.exists():
        lines.extend(
            [
                "",
                "## Dashboard összefoglaló forrása",
                "",
                f"A dashboard-jellegű Markdown összefoglaló forrása: `{dashboard_summary_path}`.",
            ]
        )
    return "\n".join(lines) + "\n"


def build_defense_notes(summary: dict[str, Any], readiness: dict[str, Any]) -> str:
    lines = [
        "# Védési jegyzetek az end-to-end integrációhoz",
        "",
        "## Mit mutat be ez a rész?",
        "A Wazuh alertből induló, ML scoringgal gazdagított, hibrid kockázati döntést adó feldolgozási láncot.",
        "",
        "## Ez benchmark?",
        "Nem. Ez integrációs demonstráció; a metrikai összehasonlítás külön real-lab mérési táblában szerepel.",
        "",
        "## Miért fontos az unmatched arány?",
        "Mert csak a feature mappinggel rendelkező alert események kaphatnak AE pontszámot.",
        "",
        "## Milyen státuszt adott a validáció?",
        f"`{readiness.get('readiness', 'NOT_READY')}`",
        "",
        "## Fő számok",
        "",
        metric_line(summary, "total_alerts", "Összes alert"),
        metric_line(summary, "scored_alerts", "Pontozott alert"),
        metric_line(summary, "unmatched_alerts", "Unmatched alert"),
        metric_line(summary, "hybrid_positive_count", "Hibrid pozitív döntés"),
        "",
        "## Legfontosabb korlát",
        "A dashboard-ready kimenet csak verified real-lab provenance mellett használható dolgozati integrációs bizonyítékként.",
    ]
    return "\n".join(lines) + "\n"


def generate_thesis_section(
    output_dir: Path,
    provenance_path: Path,
    dashboard_summary_path: Path | None = None,
) -> dict[str, Path]:
    summary = read_one_row_csv(output_dir / "enrichment_summary.csv")
    readiness = read_readiness(output_dir / "live_integration_readiness.json")
    thesis_text = build_thesis_section(summary, readiness, provenance_path, dashboard_summary_path)
    notes_text = build_defense_notes(summary, readiness)
    output_dir.mkdir(parents=True, exist_ok=True)
    thesis_path = output_dir / "thesis_live_integration_section.md"
    notes_path = output_dir / "live_integration_defense_notes.md"
    metadata_path = output_dir / "thesis_section_metadata.json"
    thesis_path.write_text(thesis_text, encoding="utf-8")
    notes_path.write_text(notes_text, encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "summary": str(output_dir / "enrichment_summary.csv"),
                "dashboard_summary": "" if dashboard_summary_path is None else str(dashboard_summary_path),
                "readiness": str(output_dir / "live_integration_readiness.json"),
                "provenance": str(provenance_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {"thesis": thesis_path, "defense_notes": notes_path, "metadata": metadata_path}


def main() -> None:
    args = parse_args()
    dashboard_summary_path = Path(args.dashboard_summary) if args.dashboard_summary else Path(args.output_dir) / "dashboard_summary.md"
    outputs = generate_thesis_section(Path(args.output_dir), Path(args.provenance), dashboard_summary_path)
    for path in outputs.values():
        print(f"[OK] {path}")


if __name__ == "__main__":
    main()
