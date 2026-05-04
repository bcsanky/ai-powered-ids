from __future__ import annotations

import argparse
from pathlib import Path

from ml.src.thesis_integration.common import (
    cautious_improvement_sentence,
    compute_research_answer,
    ensure_output_dir,
    read_csv_optional,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", default="results/real_comparison/metrics_comparison.csv")
    parser.add_argument("--output-dir", default="reports/thesis_integration")
    return parser.parse_args()


def qna(answer: dict) -> list[tuple[str, str]]:
    improvement = cautious_improvement_sentence(answer)
    return [
        ("Miért Wazuh?", "A Wazuh elterjedt SIEM/IDS jellegű platform, amely szabályalapú riasztásokat ad, ezért alkalmas Wazuh-only baseline kialakítására."),
        ("Miért autoencoder?", "Az autoencoder benign minták rekonstrukcióját tanulja, így magas rekonstrukciós hiba esetén anomáliát jelezhet címkézett támadási minta nélkül is."),
        ("Mit jelent a reconstruction error?", "A bemenet és a modell által rekonstruált kimenet közötti eltérést jelenti; nagyobb érték nagyobb anomáliagyanút jelez."),
        ("Miért kell Wazuh-only baseline?", "A baseline adja meg, hogy a szabályalapú rendszer önmagában hogyan teljesít ugyanazon lab eseményeken."),
        ("Mi a különbség AE-only és hibrid között?", "Az AE-only csak a modell pontszámára támaszkodik, a hibrid stratégia pedig a Wazuh jelzést és az AE döntést együtt értelmezi."),
        ("Javult-e a rendszer?", improvement),
        ("Miért lehet magas a false positive rate?", "Oka lehet érzékeny küszöb, pontatlan feature mapping vagy a lab forgalom korlátozott reprezentativitása."),
        ("Mitől MSc szintű a munka?", "A munka több alrendszert kapcsol össze: adatelőkészítés, ML scoring, SIEM alert korreláció, hibrid döntés, provenance és QA."),
        ("Miért nem production SOC?", "A mérés kontrollált lab környezetben történt, kis elemszámmal és nem hosszú idejű üzemi terheléssel."),
        ("Mi a legfontosabb mérnöki hozzájárulás?", "A reprodukálható Wazuh-only, AE-only és hibrid összehasonlító pipeline provenance védelemmel."),
        ("Hogyan biztosítottad, hogy nem demonstrációs adatból van az eredmény?", "A no-demo guard tiltja az examples, templates és tests eredetű inputokat, a provenance pedig SHA256 azonosítókkal rögzíti a bemeneteket."),
        ("Mi a provenance szerepe?", "A provenance az inputok és kimenetek azonosságát dokumentálja, és megakadályozza, hogy nem igazolt fájlok végleges real-lab eredményként szerepeljenek."),
        ("Hogyan lehetne továbbfejleszteni?", "Nagyobb lab eseménykészlettel, több támadástípussal, stabilabb feature mappinggel és hosszabb mérési ablakkal."),
        ("Miért kell hibrid priority stratégia?", "Mert nemcsak bináris riasztást ad, hanem üzemeltetési prioritást rendel a Wazuh és ML jelzés kombinációjához."),
        ("Mit jelent a riasztásszám növekedése?", "Nagyobb elemzői terhelést jelenthet, ezért a recall növekedésével együtt kell értelmezni."),
        ("Miért fontos a false negative rate?", "A kihagyott támadó eseményeket jelzi, ami biztonsági kockázatot jelent."),
        ("Miért nem elég az F1?", "Az F1 nem mutatja külön a riasztási terhelést, a TTD-t és az üzemeltetési kompromisszumokat."),
        ("Hogyan kapcsolódik a live integration a méréshez?", "A live integration mérnöki demonstráció: alertet gazdagít ML pontszámmal és dashboard-ready kimenetet ad, de nem önálló benchmark."),
        ("Mit jelent az AE lab TTD korlát?", "Az AE scoring offline feldolgozásban történik, ezért nem natív alert időként kell értelmezni."),
        ("Mi a legfontosabb korlát?", "A kontrollált lab mérés nem általánosítható közvetlenül éles SOC környezetre."),
    ]


def generate_defense_questions(comparison_path: Path, output_dir: Path) -> Path:
    output_dir = ensure_output_dir(output_dir)
    answer = compute_research_answer(read_csv_optional(comparison_path))
    lines = ["# Védési kérdés-válasz segédlet", ""]
    for index, (question, response) in enumerate(qna(answer), start=1):
        lines.append(f"## {index}. {question}")
        lines.append("")
        lines.append(response)
        lines.append("")
    output = output_dir / "defense_questions_generated.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    output = generate_defense_questions(Path(args.comparison), Path(args.output_dir))
    print(f"[OK] Kimenet: {output}")


if __name__ == "__main__":
    main()

