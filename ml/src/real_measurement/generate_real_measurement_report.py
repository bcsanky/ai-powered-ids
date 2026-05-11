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
    parser.add_argument("--comparison", default="results/real_comparison/metrics_comparison.csv")
    parser.add_argument("--wazuh-metrics", default="results/wazuh_real/metrics_summary.csv")
    parser.add_argument("--ae-metrics", default="results/ae_lab/metrics_summary.csv")
    parser.add_argument("--hybrid-metrics", default="results/hybrid_real/metrics_summary.csv")
    parser.add_argument("--input-validation", default="reports/lab_input_validation/input_validation_report.md")
    parser.add_argument("--wazuh-summary", default="reports/wazuh_export/wazuh_export_summary.md")
    parser.add_argument("--provenance", default="reports/real_measurement/measurement_provenance.json")
    parser.add_argument("--output-dir", default="reports/real_measurement")
    return parser.parse_args()


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Üres CSV: {path}")
    return df


def markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def get_f1(df: pd.DataFrame, configuration: str) -> float | None:
    row = df[df["configuration"] == configuration]
    if row.empty or "f1" not in row.columns:
        return None
    value = pd.to_numeric(row.iloc[0]["f1"], errors="coerce")
    return None if pd.isna(value) else float(value)


def get_metric(df: pd.DataFrame, configuration: str, metric: str) -> float | None:
    row = df[df["configuration"] == configuration]
    if row.empty or metric not in row.columns:
        return None
    value = pd.to_numeric(row.iloc[0][metric], errors="coerce")
    return None if pd.isna(value) else float(value)


def best_hybrid_f1(df: pd.DataFrame) -> tuple[str, float] | None:
    hybrids = df[df["configuration"].astype(str).str.startswith("Hybrid")].copy()
    if hybrids.empty or "f1" not in hybrids.columns:
        return None
    hybrids["f1_numeric"] = pd.to_numeric(hybrids["f1"], errors="coerce")
    hybrids = hybrids.dropna(subset=["f1_numeric"])
    if hybrids.empty:
        return None
    row = hybrids.sort_values("f1_numeric", ascending=False, kind="mergesort").iloc[0]
    return str(row["configuration"]), float(row["f1_numeric"])


def improvement_text(comparison: pd.DataFrame) -> str:
    wazuh_f1 = get_f1(comparison, "Wazuh-only")
    best_hybrid = best_hybrid_f1(comparison)
    if wazuh_f1 is None or best_hybrid is None:
        return "A javulás mértéke a rendelkezésre álló metrikák alapján nem ítélhető meg."
    hybrid_name, hybrid_f1 = best_hybrid
    if hybrid_f1 > wazuh_f1:
        return (
            f"Az összefoglaló értékelés szerint a vizsgált lab mérésben javulás figyelhető meg: "
            f"a legjobb hibrid konfiguráció ({hybrid_name}) F1 értéke magasabb, mint a Wazuh-only baseline F1 értéke."
        )
    return "A vizsgált lab mérésben nem igazolható egyértelmű javulás a Wazuh-only baseline-hoz képest."


def metric_change_text(comparison: pd.DataFrame, metric: str, label: str, lower_is_better: bool = False) -> str:
    wazuh = get_metric(comparison, "Wazuh-only", metric)
    best_hybrid = best_hybrid_f1(comparison)
    if wazuh is None or best_hybrid is None:
        return f"- {label}: nem értelmezhető a rendelkezésre álló adatokból."
    hybrid_name, _ = best_hybrid
    hybrid_value = get_metric(comparison, hybrid_name, metric)
    if hybrid_value is None:
        return f"- {label}: nem értelmezhető a kiválasztott hibrid konfigurációra."
    if lower_is_better:
        relation = "alacsonyabb" if hybrid_value < wazuh else "nem alacsonyabb"
    else:
        relation = "magasabb" if hybrid_value > wazuh else "nem magasabb"
    return f"- {label}: a {hybrid_name} értéke {relation}, mint a Wazuh-only érték."


def html_from_markdown(markdown: str) -> str:
    escaped = markdown.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"<html><body><pre>{escaped}</pre></body></html>\n"


def optional_section(path: Path, title: str) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [f"## {title}", "", text.strip(), ""]


def provenance_lines(path: Path) -> tuple[list[str], bool]:
    provenance = load_provenance(path)
    valid, errors = validate_provenance_payload(provenance)
    if provenance is None:
        return (
            [
                "## Adateredet és provenance",
                "",
                "Figyelmeztetés: provenance fájl hiányzik, ezért az eredmények nem használhatók végleges real-lab bizonyítékként.",
                "",
            ],
            False,
        )
    lines = [
        "## Adateredet és provenance",
        "",
        f"- measurement_source: `{provenance.get('measurement_source', 'nincs adat')}`",
        f"- ground_truth_path: `{provenance.get('ground_truth_path', 'nincs adat')}`",
        f"- ground_truth_sha256: `{provenance.get('ground_truth_sha256', 'nincs adat')}`",
        f"- lab_features_path: `{provenance.get('lab_features_path', 'nincs adat')}`",
        f"- lab_features_sha256: `{provenance.get('lab_features_sha256', 'nincs adat')}`",
        f"- wazuh_alerts_path: `{provenance.get('wazuh_alerts_path', 'nincs adat')}`",
        f"- wazuh_alerts_sha256: `{provenance.get('wazuh_alerts_sha256', 'nincs adat')}`",
        "",
    ]
    if not valid:
        lines.extend(
            [
                "Figyelmeztetés: a provenance fájl nem érvényes, ezért az eredmények nem használhatók végleges real-lab bizonyítékként.",
                "Hibák: " + "; ".join(errors),
                "",
            ]
        )
    return lines, valid


def generate_report(
    *,
    comparison_path: Path,
    wazuh_metrics_path: Path,
    ae_metrics_path: Path,
    hybrid_metrics_path: Path,
    input_validation_path: Path,
    wazuh_summary_path: Path,
    provenance_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    comparison = read_required_csv(comparison_path)
    read_required_csv(wazuh_metrics_path)
    read_required_csv(ae_metrics_path)
    read_required_csv(hybrid_metrics_path)

    table = markdown_table(comparison)
    conclusion = improvement_text(comparison)
    metric_notes = [
        metric_change_text(comparison, "recall", "Recall"),
        metric_change_text(comparison, "false_positive_rate", "FPR", lower_is_better=True),
        metric_change_text(comparison, "alert_count", "Riasztásszám", lower_is_better=True),
    ]
    provenance_section, has_valid_provenance = provenance_lines(provenance_path)
    title = "# Valós lab-alapú Wazuh+AE eredmények" if has_valid_provenance else "# Lab-alapú Wazuh+AE eredmények provenance nélkül"
    thesis_lines = [
        title,
        "",
        *provenance_section,
        "## Mérési cél",
        "A mérés célja annak ellenőrzése, hogy ugyanazon címkézett lab eseményeken hogyan viszonyul egymáshoz a Wazuh-only szabályalapú baseline, az AE-Minimal offline lab pontozás és a hibrid Wazuh+AE döntés.",
        "",
        "## Inputok",
        "A mérés a `lab_ground_truth.csv`, a `lab_features.csv` és a Wazuh alert export alapján készült. A metrikák kizárólag ezekből és a feldolgozási lánc kimeneteiből származnak.",
        "",
        "## Konfigurációk értelmezése",
        "- Wazuh-only: natív Wazuh alert exportból korrelált szabályalapú jelzés.",
        "- AE-Minimal lab: a végleges AE-Minimal modell offline pontozása a lab feature-ökön.",
        "- All-positive baseline: naiv kontrollsor, amely minden eseményt pozitívnak jelöl; nem valós detektor és nem Wazuh-, Zeek-, AE- vagy hibrid predikció.",
        "- Hybrid OR, weighted és priority: a Wazuh és AE jelzések kontrollált kombinációi.",
        "",
        "## Fő eredménytábla",
        "",
        table,
        "",
        "## Mérnöki értékelés",
        conclusion,
        "",
        *metric_notes,
        "",
        "## Korlátok",
        "- A mérés lab környezetben készült, nem hosszú idejű éles SOC-validáció.",
        "- Az AE-only ág offline scoring, ezért natív detektálási idő csak a Wazuh-alapú riasztásoknál értelmezhető.",
        "- A hibrid döntés minősége az event_id alapú illesztés és az input feature mapping pontosságától függ.",
        "- A Wazuh export teljessége és az időszinkron közvetlenül befolyásolja az eredményt.",
    ]
    report_lines = [
        "# Real-lab mérési riport",
        "",
        *thesis_lines[2:],
        "",
        *optional_section(input_validation_path, "Input validáció"),
        *optional_section(wazuh_summary_path, "Wazuh export összefoglaló"),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "real_lab_results_report.md"
    html_path = output_dir / "real_lab_results_report.html"
    thesis_path = output_dir / "thesis_real_lab_section.md"
    metadata_path = output_dir / "run_metadata.json"

    report_text = "\n".join(report_lines) + "\n"
    thesis_text = "\n".join(thesis_lines) + "\n"
    report_path.write_text(report_text, encoding="utf-8")
    html_path.write_text(html_from_markdown(report_text), encoding="utf-8")
    thesis_path.write_text(thesis_text, encoding="utf-8")
    metadata: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "comparison": str(comparison_path),
        "wazuh_metrics": str(wazuh_metrics_path),
        "ae_metrics": str(ae_metrics_path),
        "hybrid_metrics": str(hybrid_metrics_path),
        "output_dir": str(output_dir),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"report": report_path, "html": html_path, "thesis": thesis_path, "metadata": metadata_path}


def main() -> None:
    args = parse_args()
    outputs = generate_report(
        comparison_path=Path(args.comparison),
        wazuh_metrics_path=Path(args.wazuh_metrics),
        ae_metrics_path=Path(args.ae_metrics),
        hybrid_metrics_path=Path(args.hybrid_metrics),
        input_validation_path=Path(args.input_validation),
        wazuh_summary_path=Path(args.wazuh_summary),
        provenance_path=Path(args.provenance),
        output_dir=Path(args.output_dir),
    )
    for path in outputs.values():
        print(f"[OK] Kimenet: {path}")


if __name__ == "__main__":
    main()
