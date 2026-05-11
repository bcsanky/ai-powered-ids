from __future__ import annotations

import argparse
from pathlib import Path

from ml.src.thesis_integration.common import (
    DEFAULT_PROVENANCE,
    METRIC_COLUMNS,
    cautious_improvement_sentence,
    compute_research_answer,
    dataframe_markdown_table,
    ensure_output_dir,
    format_metric,
    provenance_warning_text,
    read_csv_optional,
    read_json_optional,
    tradeoff_sentences,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", default="results/real_comparison/metrics_comparison.csv")
    parser.add_argument("--wazuh-metrics", default="results/wazuh_real/metrics_summary.csv")
    parser.add_argument("--ae-metrics", default="results/ae_lab/metrics_summary.csv")
    parser.add_argument("--hybrid-metrics", default="results/hybrid_real/metrics_summary.csv")
    parser.add_argument("--research-answer", default="reports/real_measurement_qa/research_question_answer.json")
    parser.add_argument("--provenance", default=str(DEFAULT_PROVENANCE))
    parser.add_argument("--output-dir", default="reports/thesis_integration")
    return parser.parse_args()


def metric_sentence(comparison, configuration: str) -> str:
    if comparison is None or comparison.empty or "configuration" not in comparison.columns:
        return f"A {configuration} metrikái nem állnak rendelkezésre."
    rows = comparison[comparison["configuration"].astype(str) == configuration]
    if rows.empty:
        return f"A {configuration} sor nem található az összehasonlító táblában."
    row = rows.iloc[0]
    return (
        f"A {configuration} sor fő értékei: precision={format_metric(row.get('precision'), 'precision')}, "
        f"recall={format_metric(row.get('recall'), 'recall')}, F1={format_metric(row.get('f1'), 'f1')}, "
        f"hamis pozitív arány={format_metric(row.get('false_positive_rate'), 'false_positive_rate')}, "
        f"riasztásszám={format_metric(row.get('alert_count'), 'alert_count')}."
    )


def generate_chapter6(
    *,
    comparison_path: Path,
    wazuh_metrics_path: Path,
    ae_metrics_path: Path,
    hybrid_metrics_path: Path,
    research_answer_path: Path,
    provenance_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir = ensure_output_dir(output_dir)
    comparison = read_csv_optional(comparison_path)
    hybrid = read_csv_optional(hybrid_metrics_path)
    answer = read_json_optional(research_answer_path) or compute_research_answer(comparison)
    provenance_note = provenance_warning_text(provenance_path)
    metric_table = dataframe_markdown_table(comparison, METRIC_COLUMNS) if comparison is not None and not comparison.empty else "Nincs elérhető összehasonlító tábla.\n"
    hybrid_table = dataframe_markdown_table(hybrid, ["strategy", *METRIC_COLUMNS[1:]]) if hybrid is not None and not hybrid.empty else "Nincs elérhető hibrid metrikatábla.\n"
    tradeoffs = "\n".join(f"- {sentence}" for sentence in tradeoff_sentences(answer))
    improvement = cautious_improvement_sentence(answer)
    best_name = answer.get("best_hybrid_by_f1") or "nincs adat"

    body = f"""# 6. Eredmények és értékelés

## 6.1 Értékelési cél és módszertan

Az értékelés célja annak vizsgálata, hogy ugyanazon címkézett lab eseményeken hogyan viselkedik a Wazuh-only baseline, az AE-Minimal lab pontozás és a három hibrid döntési stratégia. A fő metrikák a precision, recall, F1, hamis pozitív arány, hamis negatív arány, riasztásszám és ahol értelmezhető, a time-to-detection.

Az All-positive baseline külön naiv kontrollsor: minden eseményt pozitívnak jelöl, ezért nem Wazuh-, Zeek-, AE- vagy hibrid detektor, hanem triviális viszonyítási alap.

## 6.2 Mérési környezet és inputok

{provenance_note} A mérés bemeneteit a ground truth eseményablak, a lab feature tábla és a Wazuh alert export adja. A mérési eredmények csak a vizsgált lab mérésben értelmezhetők.

## 6.3 Wazuh-only baseline eredményei

{metric_sentence(comparison, "Wazuh-only")}

## 6.4 AE-Minimal lab eredményei

{metric_sentence(comparison, "AE-Minimal lab")} Az AE-only eredmény az offline scoring döntését mutatja, ezért a time-to-detection nem ugyanúgy értelmezhető, mint Wazuh alert esetén.

## 6.5 Hibrid stratégiák eredményei

{metric_sentence(comparison, "Hybrid OR")}
{metric_sentence(comparison, "Hybrid weighted")}
{metric_sentence(comparison, "Hybrid priority")}

## 6.6 Összehasonlító értékelés

{improvement} A legjobb F1 szerinti hibrid konfiguráció: `{best_name}`.

{tradeoffs}

## 6.7 Téves riasztások és riasztási terhelés

A hamis pozitív arány és a riasztásszám együtt értelmezendő. Ha a hibrid stratégia több eseményre riaszt, az jobb lefedettséget adhat, de egyúttal nagyobb elemzői terheléssel járhat. A vizsgált lab mérésben ezért nem elegendő kizárólag az F1 értéket figyelni.

## 6.8 Time-to-detection értelmezése

A Wazuh alertből származó time-to-detection az első illeszkedő alert és a ground truth kezdete közötti időt jelenti. AE-only esetben a pontozás offline batch scoring folyamatban történik, ezért az AE eredmény nem natív eseményidőben mért detekciós késleltetés.

## 6.9 Hibaanalízis

A false positive események oka lehet túl érzékeny küszöb, nem megfelelő feature mapping vagy a lab forgalom sajátos eloszlása. False negative esetben előfordulhat, hogy a Wazuh szabály nem riasztott, vagy az AE rekonstrukciós hiba nem lépte át a küszöböt. Ha az AE-only sor mindenre vagy aránytalanul sok eseményre riaszt, az threshold- vagy feature mapping problémaként kezelendő.

## 6.10 Korlátok

A mérés kontrollált lab környezetre vonatkozik. Nem hosszú idejű éles SOC-validáció, nem teljes SIEM end-to-end teljesítményteszt, és nem általánosítható változtatás nélkül más hálózatokra. A Wazuh logadat és a flow jellegű feature tábla adatmodellje eltér, ezért a korreláció pontossága külön korlát.

## 6.11 Válasz a kutatási kérdésre

{improvement} Mérnöki szempontból a prototípus bizonyítja, hogy a Wazuh alert feldolgozás, az AE-Minimal scoring és a hibrid döntés egy reprodukálható pipeline-ba rendezhető. A mérés nem bizonyít általános ipari érvényességet, de a vizsgált lab mérésben összehasonlíthatóvá teszi a Wazuh-only, AE-only és hibrid stratégiákat.
"""
    tables = f"""# 6. fejezet táblázatai

## Összehasonlító metrikák

{metric_table}
## Hibrid stratégiák

{hybrid_table}
"""
    limitations = """# 6. fejezet korlátai

- A mérés kontrollált lab eseménykészletre vonatkozik.
- A Wazuh alert és a flow feature adatmodell eltérő reprezentáció.
- Az AE-only pontozás offline scoring, ezért TTD szempontból nem azonos a Wazuh riasztási idővel.
- A hibrid döntés üzemeltetési hatását a riasztásszám és a hamis pozitív arány alapján kell értelmezni.
- A mérés nem hosszú idejű éles SOC-validáció.
"""
    research = f"""# Kutatási kérdésre adott válasz

{improvement}

Legjobb F1 szerinti hibrid konfiguráció: `{best_name}`.

## Mérnöki értelmezés

A prototípus a vizsgált lab mérésben lehetővé teszi a Wazuh-only baseline, az AE-Minimal lab pontozás és a hibrid stratégiák azonos ground truth eseményeken történő összehasonlítását. A végső következtetést az F1, recall, hamis pozitív arány és riasztásszám együttes figyelembevételével kell megfogalmazni.
"""
    outputs = {
        "chapter6": output_dir / "chapter6_results_generated.md",
        "tables": output_dir / "chapter6_tables.md",
        "limitations": output_dir / "chapter6_limitations.md",
        "research": output_dir / "chapter6_research_question_answer.md",
    }
    outputs["chapter6"].write_text(body, encoding="utf-8")
    outputs["tables"].write_text(tables, encoding="utf-8")
    outputs["limitations"].write_text(limitations, encoding="utf-8")
    outputs["research"].write_text(research, encoding="utf-8")
    write_json(output_dir / "chapter6_research_question_answer.json", answer)
    return outputs


def main() -> None:
    args = parse_args()
    outputs = generate_chapter6(
        comparison_path=Path(args.comparison),
        wazuh_metrics_path=Path(args.wazuh_metrics),
        ae_metrics_path=Path(args.ae_metrics),
        hybrid_metrics_path=Path(args.hybrid_metrics),
        research_answer_path=Path(args.research_answer),
        provenance_path=Path(args.provenance),
        output_dir=Path(args.output_dir),
    )
    for path in outputs.values():
        print(f"[OK] Kimenet: {path}")


if __name__ == "__main__":
    main()
