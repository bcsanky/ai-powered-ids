from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.real_measurement_qa.postrun_quality_gate import compute_research_answer


METRIC_LABELS = {
    "configuration": "Konfiguráció",
    "strategy": "Stratégia",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "false_positive_rate": "Hamis pozitív arány",
    "false_negative_rate": "Hamis negatív arány",
    "alert_count": "Riasztásszám",
    "mean_ttd": "Átlagos TTD (s)",
    "median_ttd": "Medián TTD (s)",
    "n_samples": "Mintaszám",
    "n_attack": "Támadó esemény",
    "n_benign": "Benign esemény",
}
REAL_COMPARISON_COLUMNS = [
    "configuration",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "false_negative_rate",
    "alert_count",
    "mean_ttd",
    "n_samples",
]
SINGLE_METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "false_negative_rate",
    "alert_count",
    "mean_ttd",
    "median_ttd",
    "n_samples",
    "n_attack",
    "n_benign",
]
HYBRID_COLUMNS = ["strategy", *SINGLE_METRIC_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", default="results/real_comparison/metrics_comparison.csv")
    parser.add_argument("--wazuh-metrics", default="results/wazuh_real/metrics_summary.csv")
    parser.add_argument("--ae-metrics", default="results/ae_lab/metrics_summary.csv")
    parser.add_argument("--hybrid-metrics", default="results/hybrid_real/metrics_summary.csv")
    parser.add_argument("--research-answer", default="reports/real_measurement_qa/research_question_answer.json")
    parser.add_argument("--output-dir", default="reports/real_measurement_qa")
    return parser.parse_args()


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Üres CSV: {path}")
    return df


def format_cell(value: Any, column: str) -> str:
    if value is None or pd.isna(value):
        return "nincs adat"
    if column == "alert_count" or column.startswith("n_"):
        converted = pd.to_numeric(value, errors="coerce")
        return "nincs adat" if pd.isna(converted) else str(int(converted))
    if column in {"precision", "recall", "f1", "false_positive_rate", "false_negative_rate", "mean_ttd", "median_ttd"}:
        converted = pd.to_numeric(value, errors="coerce")
        return "nincs adat" if pd.isna(converted) else f"{float(converted):.4f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    present_columns = [col for col in columns if col in df.columns]
    header = [METRIC_LABELS.get(col, col) for col in present_columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(format_cell(row[col], col) for col in present_columns) + " |")
    return "\n".join(lines) + "\n"


def load_research_answer(path: Path, comparison: pd.DataFrame) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return compute_research_answer(comparison)


def bullet_lines(comparison: pd.DataFrame, answer: dict[str, Any]) -> list[str]:
    best = answer.get("best_hybrid_by_f1") or "nincs adat"
    f1_delta = answer.get("f1_delta_vs_wazuh")
    fpr_delta = answer.get("fpr_delta_vs_wazuh")
    alert_delta = answer.get("alert_count_delta_vs_wazuh")
    rows = len(comparison)
    lines = [
        f"- Az összehasonlító táblázat {rows} konfiguráció metrikáit tartalmazza ugyanazon real-lab mérési készleten.",
        f"- A legjobb F1 érték szerinti hibrid konfiguráció: `{best}`.",
    ]
    if f1_delta is None:
        lines.append("- A Wazuh-only baseline-hoz viszonyított F1 változás a rendelkezésre álló adatokból nem dönthető el.")
    elif f1_delta > 0:
        lines.append("- A vizsgált lab mérés alapján a legjobb hibrid konfiguráció F1 értéke magasabb, mint a Wazuh-only baseline értéke.")
    else:
        lines.append("- A vizsgált lab mérés alapján nem igazolható egyértelmű F1 javulás a Wazuh-only baseline-hoz képest.")
    if fpr_delta is None:
        lines.append("- A hamis pozitív arány változása nem értelmezhető teljesen a rendelkezésre álló adatokból.")
    elif fpr_delta > 0:
        lines.append("- A legjobb F1 szerinti hibrid konfigurációnál a hamis pozitív arány magasabb, ezért ezt a korlátok között jelezni kell.")
    else:
        lines.append("- A legjobb F1 szerinti hibrid konfigurációnál a hamis pozitív arány nem magasabb, mint a Wazuh-only baseline értéke.")
    if alert_delta is None:
        lines.append("- A riasztásszám változása nem dönthető el.")
    elif alert_delta > 0:
        lines.append("- A hibrid döntés riasztásszám-növekedést okozhat, ami üzemeltetési szempontból külön értelmezendő.")
    else:
        lines.append("- A legjobb F1 szerinti hibrid konfiguráció nem növelte a riasztásszámot a Wazuh-only baseline-hoz képest.")
    lines.extend(
        [
            "- Az eredmények csak az adott lab eseménykészletre vonatkoznak.",
            "- A real-lab mérés nem hosszú idejű éles SOC-validáció.",
        ]
    )
    return lines


def generate_tables(
    *,
    comparison_path: Path,
    wazuh_metrics_path: Path,
    ae_metrics_path: Path,
    hybrid_metrics_path: Path,
    research_answer_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    comparison = read_required_csv(comparison_path)
    wazuh = read_required_csv(wazuh_metrics_path)
    ae = read_required_csv(ae_metrics_path)
    hybrid = read_required_csv(hybrid_metrics_path)
    answer = load_research_answer(research_answer_path, comparison)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "real_comparison": output_dir / "thesis_table_real_comparison.md",
        "wazuh_only": output_dir / "thesis_table_wazuh_only.md",
        "ae_lab": output_dir / "thesis_table_ae_lab.md",
        "hybrid": output_dir / "thesis_table_hybrid_strategies.md",
        "bullets": output_dir / "thesis_interpretation_bullets.md",
    }
    outputs["real_comparison"].write_text(markdown_table(comparison, REAL_COMPARISON_COLUMNS), encoding="utf-8")
    outputs["wazuh_only"].write_text(markdown_table(wazuh.head(1), SINGLE_METRIC_COLUMNS), encoding="utf-8")
    outputs["ae_lab"].write_text(markdown_table(ae.head(1), SINGLE_METRIC_COLUMNS), encoding="utf-8")
    outputs["hybrid"].write_text(markdown_table(hybrid, HYBRID_COLUMNS), encoding="utf-8")
    outputs["bullets"].write_text("\n".join(bullet_lines(comparison, answer)) + "\n", encoding="utf-8")
    return outputs


def main() -> None:
    args = parse_args()
    outputs = generate_tables(
        comparison_path=Path(args.comparison),
        wazuh_metrics_path=Path(args.wazuh_metrics),
        ae_metrics_path=Path(args.ae_metrics),
        hybrid_metrics_path=Path(args.hybrid_metrics),
        research_answer_path=Path(args.research_answer),
        output_dir=Path(args.output_dir),
    )
    for path in outputs.values():
        print(f"[OK] Kimenet: {path}")


if __name__ == "__main__":
    main()
