from __future__ import annotations

import argparse
from pathlib import Path

from ml.src.thesis_integration.common import (
    DEFAULT_PROVENANCE,
    cautious_improvement_sentence,
    compute_research_answer,
    ensure_output_dir,
    provenance_warning_text,
    read_csv_optional,
    read_text_optional,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", default="results/real_comparison/metrics_comparison.csv")
    parser.add_argument("--research-answer-text", default="reports/thesis_integration/chapter6_research_question_answer.md")
    parser.add_argument("--provenance", default=str(DEFAULT_PROVENANCE))
    parser.add_argument("--output-dir", default="reports/thesis_integration")
    return parser.parse_args()


def generate_summary_hu(comparison_path: Path, research_text_path: Path, provenance_path: Path, output_dir: Path) -> Path:
    output_dir = ensure_output_dir(output_dir)
    comparison = read_csv_optional(comparison_path)
    answer = compute_research_answer(comparison)
    research_text = read_text_optional(research_text_path)
    improvement = cautious_improvement_sentence(answer)
    warning = provenance_warning_text(provenance_path)
    body = f"""# 7. Összegzés

A dolgozat egy AI-alapú kiberfenyegetés-felderítő és -elemző laboratóriumi prototípus megvalósítását mutatja be. A kiinduló probléma az volt, hogy a szabályalapú SIEM/IDS riasztások és a gépi tanulásra épülő anomáliadetektálás eltérő erősségekkel rendelkeznek, ezért mérnöki szempontból indokolt volt egy összehasonlítható, hibrid feldolgozási lánc kialakítása.

A megvalósított rendszer Wazuh-only baseline feldolgozást, AE-Minimal autoencoder pontozást és több hibrid döntési stratégiát tartalmaz. A mérési lánc külön kezeli az offline CIC-IDS2017 kísérleteket és a provenance-szel igazolt real-lab eredményeket, így a dolgozati következtetések forrása ellenőrizhető marad.

{warning} A mérési módszer központi eleme az azonos ground truth eseményeken végzett összehasonlítás. Ez lehetővé teszi, hogy a Wazuh-only, AE-only és hibrid stratégiák precision, recall, F1, hamis pozitív arány és riasztásszám alapján kerüljenek értékelésre.

{improvement} Ez a következtetés kizárólag a vizsgált lab mérésre vonatkozik, és nem jelent általános production teljesítménygaranciát.

A munka legfontosabb mérnöki eredménye a reprodukálható pipeline: lab input rögzítés, Wazuh alert export, AE scoring, hibrid döntés, provenance guard, QA és dolgozati riportkészítés. A továbbfejlesztés iránya a nagyobb lab eseménykészlet, stabilabb feature mapping, hosszabb időtávú mérés és éles környezethez közelebb álló validáció lehet.
"""
    if research_text:
        body += "\n## Kapcsolódó kutatási válasz\n\nA részletes kutatási válasz a 6. fejezethez készített külön szakaszban található.\n"
    output = output_dir / "chapter7_osszegzes_generated.md"
    output.write_text(body, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    output = generate_summary_hu(Path(args.comparison), Path(args.research_answer_text), Path(args.provenance), Path(args.output_dir))
    print(f"[OK] Kimenet: {output}")


if __name__ == "__main__":
    main()

