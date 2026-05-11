from __future__ import annotations

import argparse
from pathlib import Path

from ml.src.thesis_integration.common import (
    DEFAULT_PROVENANCE,
    compute_research_answer,
    ensure_output_dir,
    provenance_warning_text,
    read_csv_optional,
    read_json_optional,
    write_json,
)


COMPARISON_COLUMNS = [
    "configuration",
    "TP",
    "FP",
    "TN",
    "FN",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "false_negative_rate",
    "alert_count",
    "mean_ttd",
]
COMPARISON_LABELS = {
    "configuration": "Konfiguráció",
    "TP": "TP",
    "FP": "FP",
    "TN": "TN",
    "FN": "FN",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "false_positive_rate": "Hamis pozitív arány",
    "false_negative_rate": "Hamis negatív arány",
    "alert_count": "Riasztásszám",
    "mean_ttd": "Átlagos TTD (s)",
}


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


def row_for(comparison, configuration: str):
    if comparison is None or comparison.empty or "configuration" not in comparison.columns:
        return None
    rows = comparison[comparison["configuration"].astype(str) == configuration]
    if rows.empty:
        return None
    return rows.iloc[0]


def numeric(row, column: str) -> float | None:
    if row is None or column not in row.index:
        return None
    value = row.get(column)
    try:
        import pandas as pd

        converted = pd.to_numeric(value, errors="coerce")
        if pd.isna(converted):
            return None
        return float(converted)
    except Exception:
        return None


def fmt_count(row, column: str) -> str:
    value = numeric(row, column)
    return "nincs adat" if value is None else str(int(value))


def fmt_metric(row, column: str, *, zero_text: str = "0", one_text: str = "1.0") -> str:
    value = numeric(row, column)
    if value is None:
        return "nincs adat"
    if abs(value) < 1e-12:
        return zero_text
    if abs(value - 1.0) < 1e-12:
        return one_text
    return f"{value:.6f}".rstrip("0").rstrip(".")


def fmt_rate(row, column: str) -> str:
    return fmt_metric(row, column, zero_text="0.0", one_text="1.0")


def fmt_ttd(row) -> str:
    value = numeric(row, "mean_ttd")
    return "nincs adat" if value is None else f"{value:.4f}".rstrip("0").rstrip(".")


def confusion_text(row) -> str:
    return (
        f"TP={fmt_count(row, 'TP')}, FP={fmt_count(row, 'FP')}, "
        f"TN={fmt_count(row, 'TN')}, FN={fmt_count(row, 'FN')}"
    )


def metric_text(row) -> str:
    return (
        f"precision={fmt_metric(row, 'precision')}, recall={fmt_metric(row, 'recall')}, "
        f"f1={fmt_metric(row, 'f1')}, false_positive_rate={fmt_rate(row, 'false_positive_rate')}, "
        f"false_negative_rate={fmt_rate(row, 'false_negative_rate')}, alert_count={fmt_count(row, 'alert_count')}"
    )


def comparison_markdown_table(comparison) -> str:
    if comparison is None or comparison.empty:
        return "Nincs elérhető összehasonlító tábla.\n"
    present = [column for column in COMPARISON_COLUMNS if column in comparison.columns]
    lines = [
        "| " + " | ".join(COMPARISON_LABELS.get(column, column) for column in present) + " |",
        "| " + " | ".join(["---"] * len(present)) + " |",
    ]
    for _, row in comparison.iterrows():
        cells: list[str] = []
        for column in present:
            if column in {"TP", "FP", "TN", "FN", "alert_count"}:
                cells.append(fmt_count(row, column))
            elif column in {"false_positive_rate", "false_negative_rate"}:
                cells.append(fmt_rate(row, column))
            elif column == "mean_ttd":
                cells.append(fmt_ttd(row))
            elif column == "configuration":
                cells.append(str(row.get(column)))
            else:
                cells.append(fmt_metric(row, column))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def metric_improved(row, baseline, column: str) -> bool:
    current = numeric(row, column)
    base = numeric(baseline, column)
    return current is not None and base is not None and current > base


def hybrid_vs_wazuh_sentence(name: str, row, wazuh) -> str:
    if row is None or wazuh is None:
        return f"A {name} és a Wazuh-only összevetése a rendelkezésre álló adatokból nem dönthető el."
    recall_better = metric_improved(row, wazuh, "recall")
    f1_better = metric_improved(row, wazuh, "f1")
    if recall_better and f1_better:
        return (
            f"A {name} recall={fmt_metric(row, 'recall')} és f1={fmt_metric(row, 'f1')} értéke javulást mutat "
            f"a Wazuh-only recall={fmt_metric(wazuh, 'recall')} és f1={fmt_metric(wazuh, 'f1')} értékéhez képest."
        )
    if recall_better:
        return (
            f"A {name} recall={fmt_metric(row, 'recall')} értéke magasabb, mint a Wazuh-only recall={fmt_metric(wazuh, 'recall')} értéke, "
            f"de F1 alapján nem igazolható egyértelmű javulás."
        )
    if f1_better:
        return (
            f"A {name} f1={fmt_metric(row, 'f1')} értéke magasabb, mint a Wazuh-only f1={fmt_metric(wazuh, 'f1')} értéke, "
            f"de recall alapján nem igazolható egyértelmű javulás."
        )
    return (
        f"A {name} esetében nem igazolható egyértelmű javulás a Wazuh-only baseline-hoz képest: "
        f"recall={fmt_metric(row, 'recall')}, f1={fmt_metric(row, 'f1')}."
    )


def all_positive_comparison_sentence(*, all_positive, hybrid_or, hybrid_priority, ae) -> str:
    if all_positive is None:
        return "Az All-positive kontroll baseline nem áll rendelkezésre, ezért a triviális pozitív kontrollhoz viszonyított értelmezés nem dönthető el."
    equal_or = hybrid_or is not None and confusion_text(hybrid_or) == confusion_text(all_positive)
    equal_priority = hybrid_priority is not None and confusion_text(hybrid_priority) == confusion_text(all_positive)
    equal_ae = ae is not None and confusion_text(ae) == confusion_text(all_positive)
    matched = [name for name, equal in [("AE-Minimal", equal_ae), ("Hybrid OR", equal_or), ("Hybrid priority", equal_priority)] if equal]
    if matched:
        return (
            "A kontroll baseline fontos értelmezési korlátot ad: "
            + ", ".join(matched)
            + " ugyanazokat a bináris metrikákat produkálta, mint az All-positive baseline. "
            "Ezért ezek a stratégiák a vizsgált mérésben nem teljesítettek jobban a triviális, minden eseményt pozitívnak jelölő döntésnél."
        )
    return "A vizsgált stratégiák bináris eredménye eltért az All-positive baseline-tól, ezért a kontrollsor külön viszonyítási alapként értelmezendő."


def research_question_sentence(answer: dict, wazuh, hybrid_or, hybrid_priority, all_positive) -> str:
    delta = answer.get("f1_delta_vs_wazuh")
    if delta is None:
        return "A Wazuh-only baseline-hoz viszonyított F1 változás a rendelkezésre álló adatokból nem dönthető el."
    if delta > 0:
        text = (
            "a vizsgált lab mérésben F1 alapján javulás figyelhető meg a Wazuh-only baseline-hoz képest. "
            f"A Hybrid OR és a Hybrid priority esetében recall={fmt_metric(hybrid_or, 'recall')} és f1={fmt_metric(hybrid_or, 'f1')}, "
            f"míg a Wazuh-only esetében recall={fmt_metric(wazuh, 'recall')} és f1={fmt_metric(wazuh, 'f1')}."
        )
        if all_positive is not None:
            text += (
                f" Ugyanakkor ezek a hibrid stratégiák nem teljesítették túl az All-positive baseline-t, "
                f"amely szintén recall={fmt_metric(all_positive, 'recall')} és f1={fmt_metric(all_positive, 'f1')} értéket adott."
            )
        return text
    return (
        "a vizsgált lab mérésben F1 alapján nem igazolható egyértelmű javulás a Wazuh-only baseline-hoz képest. "
        f"A legjobb hibrid konfiguráció f1={fmt_metric(hybrid_or, 'f1')} körüli értéke nem haladja meg a Wazuh-only f1={fmt_metric(wazuh, 'f1')} értékét."
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
    metric_table = comparison_markdown_table(comparison)
    hybrid_table = comparison_markdown_table(comparison[comparison["configuration"].astype(str).str.startswith("Hybrid")]) if comparison is not None and not comparison.empty else "Nincs elérhető hibrid metrikatábla.\n"
    best_name = answer.get("best_hybrid_by_f1") or "nincs adat"
    _ = hybrid  # kept so the CLI contract still validates that the hybrid metrics input exists.

    wazuh = row_for(comparison, "Wazuh-only")
    ae = row_for(comparison, "AE-Minimal lab")
    all_positive = row_for(comparison, "All-positive baseline")
    hybrid_or = row_for(comparison, "Hybrid OR")
    hybrid_weighted = row_for(comparison, "Hybrid weighted")
    hybrid_priority = row_for(comparison, "Hybrid priority")
    hybrid_or_vs_wazuh = hybrid_vs_wazuh_sentence("Hybrid OR", hybrid_or, wazuh)
    hybrid_priority_vs_wazuh = hybrid_vs_wazuh_sentence("Hybrid priority", hybrid_priority, wazuh)
    all_positive_note = all_positive_comparison_sentence(
        all_positive=all_positive,
        hybrid_or=hybrid_or,
        hybrid_priority=hybrid_priority,
        ae=ae,
    )
    research_answer_text = research_question_sentence(answer, wazuh, hybrid_or, hybrid_priority, all_positive)

    body = f"""# 6. Eredmények és értékelés

## 6.1 Értékelési cél és módszertan

Az értékelés célja annak vizsgálata, hogy ugyanazon címkézett real-lab eseménykészleten hogyan viselkedik a Wazuh-only baseline, az AE-Minimal lab pontozás, az All-positive kontroll baseline és a három hibrid döntési stratégia. A végleges mérés 100 eseményt tartalmazott, ebből 40 attack és 60 benign címkéjű volt. A fő metrikák a konfúziós mátrix elemei, a precision, recall, F1, false positive rate, false negative rate, riasztásszám és ahol értelmezhető, az átlagos time-to-detection.

Az All-positive baseline külön naiv kontrollsor: minden eseményt pozitívnak jelöl, ezért nem Wazuh-, Zeek-, AE- vagy hibrid detektor, hanem triviális viszonyítási alap. Ennek szerepe annak ellenőrzése, hogy az AE- és hibrid stratégiák valóban többet adnak-e, mint a minden eseményre riasztó döntési szabály.

## 6.2 Mérési környezet és inputok

{provenance_note} A mérés bemeneteit a ground truth eseményablakok, a Zeek-alapú lab feature tábla és a Wazuh alert export adja. A Wazuh és a hibrid TTD értelmezése az első illeszkedő Wazuh alert és a ground truth kezdete közötti különbségen alapul. Az eredmények kizárólag a vizsgált `real-lab-001` mérésre vonatkoznak.

## 6.3 Wazuh-only baseline eredményei

A Wazuh-only konfiguráció konfúziós mátrixa: {confusion_text(wazuh)}. A fő metrikák: {metric_text(wazuh)}, mean_ttd={fmt_ttd(wazuh)}. Ez azt jelenti, hogy a Wazuh a 40 támadó eseményből 13-at jelzett helyesen, 27 támadó esemény viszont riasztás nélkül maradt. A 60 benign eseményből 39-et helyesen benignként hagyott, 21 esetben viszont téves riasztás keletkezett.

A Wazuh-only eredmény ezért mérsékelten konzervatív baseline-ként értelmezhető: a false_positive_rate=0.35 alacsonyabb, mint a minden eseményre riasztó stratégiáknál, ugyanakkor a recall=0.325 és a false_negative_rate=0.675 azt mutatja, hogy a támadó események jelentős része kimaradt.

## 6.4 AE-Minimal lab eredményei

Az AE-Minimal lab konfiguráció konfúziós mátrixa: {confusion_text(ae)}. A fő metrikák: {metric_text(ae)}. A konfiguráció minden eseményt pozitívként jelölt, ezért a 40 támadó eseményt mind megtalálta, de a 60 benign esemény mindegyikét téves pozitívként kezelte.

Ennek következtében az AE-Minimal recall=1.0 és f1=0.571429 értéke magasabb, mint a Wazuh-only recall=0.325 és f1=0.351351 értéke, de a false_positive_rate=1.0 és az alert_count=100 azt mutatja, hogy a döntési küszöb ebben a mérésben nem választotta szét érdemben a benign és attack eseményeket. Az AE-only eredmény offline scoring döntést jelent, ezért natív Wazuh-szerű TTD nem értelmezhető.

## 6.5 All-positive kontroll baseline értelmezése

Az All-positive baseline konfúziós mátrixa: {confusion_text(all_positive)}. A fő metrikák: {metric_text(all_positive)}. Ez a kontrollsor pontosan azt a viselkedést reprezentálja, amikor a rendszer minden eseményre riaszt. A precision=0.4 közvetlenül a mérési készlet 40 attack / 60 benign arányából következik, a recall=1.0 pedig abból, hogy nincs negatív predikció.

{all_positive_note}

## 6.6 Hibrid stratégiák eredményei

A Hybrid OR konfiguráció konfúziós mátrixa: {confusion_text(hybrid_or)}. A fő metrikák: {metric_text(hybrid_or)}, mean_ttd={fmt_ttd(hybrid_or)}. {hybrid_or_vs_wazuh} Ugyanakkor a false_positive_rate={fmt_rate(hybrid_or, 'false_positive_rate')}, az alert_count={fmt_count(hybrid_or, 'alert_count')} és az All-positive baseline-nal azonos konfúziós mátrix miatt ez a javulás nem tekinthető valódi diszkriminációs javulásnak.

A Hybrid weighted konfiguráció konfúziós mátrixa: {confusion_text(hybrid_weighted)}. A fő metrikák: {metric_text(hybrid_weighted)}. Ez a stratégia csak egy benign eseményre adott pozitív jelzést, miközben mind a 40 támadó eseményt kihagyta. Ennek megfelelően precision=0, recall=0, f1=0 és false_negative_rate=1.0. A false_positive_rate=0.016667 és az alert_count=1 alacsony riasztási terhelést mutat, de ez a támadások teljes elvesztésével jár, ezért detekciós szempontból nem használható eredmény.

A Hybrid priority konfiguráció konfúziós mátrixa: {confusion_text(hybrid_priority)}. A fő metrikák: {metric_text(hybrid_priority)}, mean_ttd={fmt_ttd(hybrid_priority)}. {hybrid_priority_vs_wazuh} A Hybrid priority a Hybrid OR-ral és az All-positive baseline-nal azonos bináris eredményt adott. A false positive rate nem csökkent, hanem false_positive_rate={fmt_rate(hybrid_priority, 'false_positive_rate')} lett.

## 6.7 Konfúziós mátrix szintű értelmezés

Konfúziós mátrix szinten három eltérő viselkedés látható. A Wazuh-only sor 13 true positive és 39 true negative mellett 27 false negative és 21 false positive hibát tartalmaz; ez részleges támadáslefedettséget és mérsékelt benign megkülönböztetést jelent. Az AE-Minimal, az All-positive baseline, a Hybrid OR és a Hybrid priority sorok 40 true positive mellett 60 false positive eredményt adtak, true negative és false negative nélkül; ezek maximális támadáslefedettséget, de nulla benign szelektivitást mutatnak. A Hybrid weighted ezzel szemben 59 true negative és 1 false positive mellett 40 false negative eredményt adott, vagyis szinte minden eseményt negatívnak tekintett, beleértve az összes támadó eseményt is.

Ez alapján a Hybrid OR és a Hybrid priority bináris metrikái a Wazuh-only baseline-hoz képest recall és F1 szerint akkor értelmezhetők javulásként, ha a magasabb támadáslefedettséget önmagában vizsgáljuk. A jelen mérésben azonban ezt a javulást a false positive rate jelentős romlása árán érik el. Mivel eredményük megegyezik az All-positive baseline eredményével, a mérés nem igazolja, hogy ezek a hibrid stratégiák ebben a konfigurációban ténylegesen jobb döntési határt tanultak vagy alkalmaztak volna.

## 6.8 Téves riasztások és riasztási terhelés

A hamis pozitív arány és a riasztásszám együtt értelmezendő. A Wazuh-only false_positive_rate=0.35 és alert_count=34 értéket adott. Ezzel szemben az AE-Minimal, az All-positive baseline, a Hybrid OR és a Hybrid priority false_positive_rate=1.0 és alert_count=100 értéket mutatott. Ezeknél a stratégiáknál a recall javulása azzal jár, hogy minden benign esemény téves pozitívként jelenik meg.

A Hybrid weighted riasztási terhelése alacsony, alert_count=1, de ez nem előnyös detekciós eredmény, mert TP=0 és FN=40. A mérés ezért azt mutatja, hogy önmagában sem a magas recall, sem az alacsony riasztásszám nem elegendő; a használható működéshez mindkét hibatípust kiegyensúlyozó küszöb- és stratégiahangolás szükséges.

## 6.9 Time-to-detection értelmezése

A Wazuh-only mean_ttd=37.2308, a Hybrid OR és a Hybrid priority esetében szintén mean_ttd=37.2308 szerepel, mert ezekben a hibrid stratégiákban a Wazuh riasztásokhoz illeszthető támadó események detektálási ideje öröklődik. AE-only és All-positive baseline esetben a TTD nem értelmezhető ugyanilyen módon, mert ezek nem natív Wazuh alert időpontra épülő detektorok.

## 6.10 Válasz a kutatási kérdésre

A kutatási kérdésre adott válasz óvatosan, korlátokkal fogalmazható meg: {research_answer_text}

Ezért a mérés nem igazolja, hogy a jelenlegi AE-Minimal vagy hibrid döntés production SOC környezetben hatékonyabb lenne. A prototípus értéke ebben az állapotban inkább az, hogy reprodukálhatóan összekapcsolja a Wazuh alert feldolgozást, a Zeek-alapú feature előállítást, az AE scoringot és a hibrid döntési logikát. Detekciós teljesítmény szempontjából további küszöbhangolásra, feature-vizsgálatra és nagyobb, változatosabb validációs mérésre van szükség.

## 6.11 Korlátok

A mérés kontrollált lab környezetben készült, 100 esemény alapján, ezért nem tekinthető hosszú idejű éles SOC-validációnak. Az események eloszlása, a támadási és benign forgatókönyvek száma, valamint a lab hálózati környezet sajátosságai korlátozzák az általánosíthatóságot. A Wazuh logadat és a Zeek/flow feature adatmodellje eltérő reprezentációt használ, ezért a korreláció és az időablak-választás közvetlenül befolyásolhatja a metrikákat.

További korlát, hogy az AE-Minimal és több hibrid stratégia ebben a mérésben minden eseményt pozitívnak jelölt. Ez threshold- vagy feature mapping problémára utalhat, amelyet külön kísérletekkel kell vizsgálni. A dolgozatban ezért az eredmény nem production teljesítményígéretként, hanem egy működő, mérhető pipeline és egy korlátokkal értelmezhető real-lab validáció eredményeként szerepeltethető.
"""
    tables = f"""# 6. fejezet táblázatai

## Összehasonlító metrikák

{metric_table}
## Hibrid stratégiák

{hybrid_table}
"""
    limitations = """# 6. fejezet korlátai

- A mérés kontrollált, 100 eseményes lab eseménykészletre vonatkozik.
- A Wazuh alert és a flow feature adatmodell eltérő reprezentáció.
- Az AE-only pontozás offline scoring, ezért TTD szempontból nem azonos a Wazuh riasztási idővel.
- A Hybrid OR és Hybrid priority recall/F1 javulása nem teljesíti túl az All-positive baseline-t.
- A hibrid döntés üzemeltetési hatását a riasztásszám és a hamis pozitív arány alapján kell értelmezni.
- További küszöbhangolásra és nagyobb validációs mérésre van szükség.
- A mérés nem hosszú idejű éles SOC-validáció.
"""
    research = f"""# Kutatási kérdésre adott válasz

{research_answer_text}

Legjobb F1 szerinti hibrid konfiguráció: `{best_name}`.

## Mérnöki értelmezés

A javulás nem jelent production SOC hatékonysági bizonyítékot, mert a Hybrid OR és a Hybrid priority nem teljesítette túl az All-positive baseline-t, és false_positive_rate=1.0 mellett működött. A prototípus a vizsgált lab mérésben lehetővé teszi a Wazuh-only baseline, az AE-Minimal lab pontozás és a hibrid stratégiák azonos ground truth eseményeken történő összehasonlítását, de a gyakorlati detekciós teljesítményhez további threshold tuning és nagyobb validáció szükséges.
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
