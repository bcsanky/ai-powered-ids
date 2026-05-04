from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.real_measurement_qa.postrun_quality_gate import compute_research_answer


QUESTIONS = [
    "Mit mértünk?",
    "Miért kellett Wazuh-only baseline?",
    "Mit jelent az AE-only eredmény?",
    "Mit jelent a hibrid stratégia?",
    "Javult-e a Wazuh eredmény?",
    "Miért lehet magas a false positive rate?",
    "Miért nem éles SOC bizonyítás?",
    "Mi a legfontosabb mérnöki eredmény?",
    "Milyen továbbfejlesztés lenne indokolt?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", default="results/real_comparison/metrics_comparison.csv")
    parser.add_argument("--research-answer", default="reports/real_measurement_qa/research_question_answer.json")
    parser.add_argument("--readiness", default="reports/real_measurement_qa/thesis_readiness.md")
    parser.add_argument("--output-dir", default="reports/real_measurement_qa")
    return parser.parse_args()


def read_comparison(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó comparison CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Üres comparison CSV: {path}")
    return df


def load_answer(path: Path, comparison: pd.DataFrame) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return compute_research_answer(comparison)


def fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "nincs adat"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def improvement_sentence(answer: dict[str, Any]) -> str:
    delta = answer.get("f1_delta_vs_wazuh")
    best = answer.get("best_hybrid_by_f1") or "nincs adat"
    if delta is None:
        return "A javulás nem dönthető el a rendelkezésre álló metrikákból."
    if delta > 0:
        return f"A vizsgált lab mérés alapján a `{best}` F1 értéke magasabb volt a Wazuh-only eredménynél."
    return "A vizsgált lab mérés alapján nem igazolható egyértelmű F1 javulás a Wazuh-only eredményhez képest."


def generate_notes(
    *,
    comparison_path: Path,
    research_answer_path: Path,
    readiness_path: Path,
    output_dir: Path,
) -> Path:
    comparison = read_comparison(comparison_path)
    answer = load_answer(research_answer_path, comparison)
    readiness_text = readiness_path.read_text(encoding="utf-8") if readiness_path.exists() else ""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "defense_notes_real_measurement.md"
    lines = [
        "# Real-lab mérés védési jegyzet",
        "",
        "## Mit mértünk?",
        "Ugyanazon címkézett lab eseményeken hasonlítottuk össze a Wazuh-only szabályalapú baseline-t, az AE-Minimal offline lab pontozást és a hibrid Wazuh+AE stratégiákat.",
        "",
        "## Miért kellett Wazuh-only baseline?",
        "A Wazuh-only eredmény adja azt a szabályalapú viszonyítási pontot, amelyhez az AE-only és a hibrid döntések mérnöki szempontból hasonlíthatók.",
        "",
        "## Mit jelent az AE-only eredmény?",
        "Az AE-only ág azt mutatja meg, hogy a végleges AE-Minimal modell a lab feature-ök alapján, Wazuh riasztási információ nélkül milyen besorolást ad.",
        "",
        "## Mit jelent a hibrid stratégia?",
        "A hibrid stratégiák a Wazuh riasztást és az AE pontozást kombinálják. Az OR, weighted és priority változat eltérő döntési szabályt képvisel.",
        "",
        "## Javult-e a Wazuh eredmény?",
        improvement_sentence(answer),
        f"Wazuh F1: {fmt(answer.get('wazuh_f1'))}; legjobb hibrid F1: {fmt(answer.get('best_hybrid_f1'))}.",
        "",
        "## Miért lehet magas a false positive rate?",
        "A szabályalapú és hibrid döntések érzékenyek lehetnek a lab eseményablakok időzítésére, a feature mapping pontosságára és a Wazuh rule-ok konfigurációjára.",
        "",
        "## Miért nem éles SOC bizonyítás?",
        "A mérés kontrollált lab környezetben készült, rövid mérési ablakokkal. Nem hosszú idejű, változó terhelésű éles SOC-validáció.",
        "",
        "## Mi a legfontosabb mérnöki eredmény?",
        "A prototípus ugyanazon event_id készleten képes Wazuh-only, AE-only és hibrid eredményeket előállítani, majd ellenőrzött metrikákkal összehasonlítani.",
        "",
        "## Milyen továbbfejlesztés lenne indokolt?",
        "Nagyobb lab eseménykészlet, hosszabb mérési időablak, stabilabb eseménykorrelációs kulcsok és élesebb Wazuh exportfolyamat növelné a mérés külső érvényességét.",
        "",
        "## Beemelhetőségi kivonat",
        readiness_text.strip() if readiness_text else "A thesis_readiness.md még nem áll rendelkezésre.",
        "",
        "## Ellenőrző kérdések",
        *[f"- {question}" for question in QUESTIONS],
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    output = generate_notes(
        comparison_path=Path(args.comparison),
        research_answer_path=Path(args.research_answer),
        readiness_path=Path(args.readiness),
        output_dir=Path(args.output_dir),
    )
    print(f"[OK] Kimenet: {output}")


if __name__ == "__main__":
    main()
