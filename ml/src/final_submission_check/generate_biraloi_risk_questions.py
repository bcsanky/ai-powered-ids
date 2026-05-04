from __future__ import annotations

import argparse
from pathlib import Path

from ml.src.final_submission_check.common import ensure_output_dir


QUESTIONS = [
    ("Teljesíti-e a feladatlapot?", "A rendszer a fő mérnöki pontokat lefedi, de a végleges minősítéshez a Word dokumentumban is ellenőrizni kell a fejezeteket."),
    ("Hol van a saját mérnöki munka?", "A saját munka a pipeline-ok, guardok, validátorok, scoring és hibrid értékelési rétegek megvalósításában jelenik meg."),
    ("Miért nem elég csak Wazuh?", "A Wazuh szabályalapú baseline jó viszonyítási alap, de az AE komponens más típusú anomáliákat is jelezhet."),
    ("Miért autoencoder?", "Az autoencoder benign mintázatok rekonstrukcióját tanulja, így rekonstrukciós hibából képez anomáliapontszámot."),
    ("Miért nem TensorFlow, ha korábban az szerepelt?", "A prototípus végül sklearn alapú autoencodert használ, mert az illeszkedett a reprodukálható mérési lánchoz és a dolgozat hatóköréhez."),
    ("Milyen adatból tanult a modell?", "Az offline modell a dokumentált CIC-IDS2017 feldolgozási lánc benign tanító részén tanul."),
    ("Miért nem fake/generált mérés?", "Real-lab eredmény csak provenance fájllal, input hash-ekkel és no-demo guard mellett használható."),
    ("Mi bizonyítja a provenance-t?", "A measurement_provenance.json rögzíti az input útvonalakat és SHA256 azonosítókat."),
    ("Mi a különbség offline CIC-IDS2017 és real-lab mérés között?", "Az offline ág dataset-alapú modellvalidáció, a real-lab ág tényleges Wazuh exportot és lab ground truth-t használ."),
    ("Mi a hibrid stratégia előnye?", "Képes együtt értelmezni a szabályalapú és ML jelzést, így prioritási döntést is adhat."),
    ("Mi a hibrid stratégia hátránya?", "Növelheti a riasztásszámot vagy a hamis pozitív arányt, ezért metrikák alapján kell értékelni."),
    ("Mit jelent a false positive rate?", "A benign események közül tévesen riasztott arányt jelenti."),
    ("Miért lehet magas a riasztásszám?", "Érzékeny küszöb, OR logika vagy pontatlan feature mapping növelheti."),
    ("Miért nem production SOC?", "A prototípus laboratóriumi mérési és integrációs rendszer, nem hosszú idejű éles üzemű platform."),
    ("Mi az adatvédelmi kockázat?", "Wazuh logokban IP-címek, hostnevek és felhasználói nyomok lehetnek."),
    ("Hogyan anonimizáltad az érzékeny adatokat?", "A redaction réteg determinisztikus tokenekre cseréli az IP-címeket és hostneveket."),
    ("Hogyan reprodukálható a mérés?", "Makefile célok, runbookok, session marker, provenance és manifest alapján."),
    ("Mit tartalmaz a melléklet?", "Konfigurációkat, manifestet, provenance kivonatot, runbook kivonatokat és ellenőrzött riportokat."),
    ("Mi kerülhet Gitbe és mi nem?", "Forráskód, dokumentáció, sablon igen; raw lab input és futási output nem."),
    ("Mit fejlesztenél tovább?", "Nagyobb lab mérés, több támadástípus, stabilabb feature mapping és hosszabb megfigyelési idő."),
    ("Mit bizonyít a live integration?", "Azt, hogy Wazuh alertből ML scoringgal gazdagított, dashboard-ready kimenet készíthető."),
    ("Mit nem bizonyít a live integration?", "Nem benchmark és nem bizonyít detektálási javulást önmagában."),
    ("Miért fontos a no-overclaiming check?", "Megakadályozza, hogy a dokumentáció túlzó vagy nem igazolt állítást tartalmazzon."),
    ("Miért fontos a failure mode check?", "Bizonyítja, hogy a rendszer hiányzó vagy tiltott input esetén nem készít látszólagos eredményt."),
    ("Mikor mondható beadásra késznek?", "Akkor, ha a valós mérés, provenance, QA, dolgozati integráció és kézi Word ellenőrzés is megtörtént."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/final_submission_check")
    return parser.parse_args()


def generate_questions(output_dir: Path) -> Path:
    output_dir = ensure_output_dir(output_dir)
    lines = ["# Bírálói kockázati kérdések", ""]
    for index, (question, answer) in enumerate(QUESTIONS, start=1):
        lines.extend([f"## {index}. {question}", "", answer, ""])
    output = output_dir / "biraloi_risk_questions.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    output = generate_questions(Path(args.output_dir))
    print(f"[OK] Kimenet: {output}")


if __name__ == "__main__":
    main()

