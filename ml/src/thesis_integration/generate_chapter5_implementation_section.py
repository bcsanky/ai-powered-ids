from __future__ import annotations

import argparse
from pathlib import Path

from ml.src.thesis_integration.common import (
    DEFAULT_PROVENANCE,
    ensure_output_dir,
    provenance_warning_text,
    read_csv_optional,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/thesis_integration")
    parser.add_argument("--provenance", default=str(DEFAULT_PROVENANCE))
    parser.add_argument("--live-summary", default="reports/live_integration/enrichment_summary.csv")
    return parser.parse_args()


def live_summary_sentence(path: Path) -> str:
    df = read_csv_optional(path)
    if df is None or df.empty:
        return "A live integration kimenet még nem áll rendelkezésre, ezért ez a rész a megvalósított komponens szerepét írja le."
    row = df.iloc[0].to_dict()
    total = row.get("total_alerts", "nincs adat")
    scored = row.get("scored_alerts", "nincs adat")
    unmatched = row.get("unmatched_alerts", "nincs adat")
    return (
        f"A rendelkezésre álló integrációs összefoglaló {total} alert feldolgozását jelzi, "
        f"amelyből {scored} kapott ML pontszámot, míg {unmatched} alerthez nem volt illeszthető feature rekord."
    )


def generate_chapter5(output_dir: Path, provenance_path: Path, live_summary_path: Path) -> dict[str, Path]:
    output_dir = ensure_output_dir(output_dir)
    provenance_note = provenance_warning_text(provenance_path)
    live_sentence = live_summary_sentence(live_summary_path)
    body = f"""# 5. Implementáció

## 5.1 A prototípus áttekintése

A megvalósított rendszer laboratóriumi prototípus, amely a Wazuh/SIEM jellegű riasztások és az autoencoder-alapú anomáliadetektálás mérnöki összekapcsolását vizsgálja. A cél nem éles üzemi SOC-platform létrehozása, hanem egy reprodukálható feldolgozási lánc, amelyben ugyanazon lab eseményekre értelmezhető a Wazuh-only baseline, az AE-Minimal pontozás és a hibrid döntés.

{provenance_note}

## 5.2 Fejlesztési és futtatási környezet

A prototípus Python-alapú feldolgozási komponensekből áll. A mérési lánc Makefile célokon keresztül futtatható, így a lab inputok validálása, a Wazuh alert feldolgozás, az AE lab scoring, a hibrid kiértékelés, a provenance rögzítés és a dolgozati riportkészítés külön lépésekben is ellenőrizhető. A Wazuh és OpenSearch réteg a valós lab méréshez kapcsolódik; a kód nem hoz létre helyettesítő alertet vagy mérési bemenetet.

## 5.3 Adatfeldolgozási lánc

Az offline CIC-IDS2017 feldolgozás külön ágon marad: CSV bemenetből oszlopkanonizálás, címkeképzés, train/validation/calibration/test bontás és előfeldolgozó illesztés történik. A real-lab ág ettől elkülönül: a `data/lab/lab_ground_truth.csv`, a `data/lab/lab_features.csv` és a `data/wazuh/alerts.jsonl` fájlok tényleges lab futásból származó bemenetként kezelhetők, ha a provenance ezt igazolja.

## 5.4 Autoencoder modell

Az AE-Minimal komponens a korábban validált autoencoder modellfájlt és előfeldolgozó állományt használja. A pontozás a rekonstrukciós hibából számított `anomaly_score` értékre épül, amelyet a küszöbstratégia `ml_alert` döntéssé alakít. A scoring során nem szerepel helyettesítő pontszám: hiányzó modellfájl vagy preprocess állomány esetén a folyamat hibával áll le.

## 5.5 Wazuh-only baseline feldolgozása

A Wazuh-only baseline a Wazuh alert exportot normalizált CSV-vé alakítja, majd a ground truth eseményablakokhoz korrelálja. Egy esemény Wazuh pozitívnak számít, ha a konfigurált időablakon belül releváns alert illeszkedik hozzá. A baseline metrikái így ugyanarra a címkézett eseménykészletre számíthatók, mint az AE és a hibrid döntések.

## 5.6 AE lab scoring

Az AE lab scoring a validált lab feature táblát event_id alapján összekapcsolja a ground truth állománnyal, majd az AE-Minimal runtime logikával minden eseményhez anomáliapontszámot rendel. A kimenet tartalmazza az `ae_pred`, `anomaly_score`, `threshold_name`, `threshold_value`, `risk_level` és `reason` mezőket.

## 5.7 Hibrid döntési stratégiák

A hibrid értékelés három döntési módot valósít meg. A Hybrid OR stratégia riaszt, ha a Wazuh vagy az AE pozitív. A Hybrid weighted stratégia normalizált AE pontszámot és Wazuh rule level értéket kombinál. A Hybrid priority stratégia kockázati szintet rendel a két forrás együttes vagy különálló pozitív jelzéséhez.

## 5.8 Provenance és no-demo guard

A dolgozati real-lab eredmény kizárólag akkor tekinthető véglegesnek, ha a `measurement_provenance.json` tartalmazza a bemeneti állományok útvonalát, SHA256 azonosítóját és a `real_lab` eredetet. A guard réteg tiltja, hogy `examples`, `templates`, `tests`, illetve sample vagy fixture jellegű fájlok real-lab bizonyítékként kerüljenek feldolgozásra.

## 5.9 Live integration és dashboard-ready output

Az end-to-end integrációs réteg a Wazuh alertet feature mapping segítségével AE pontozással gazdagítja, majd hibrid prioritási szintet rendel hozzá. {live_sentence} Ez a kimenet dashboard-ready formátumot biztosít, de önmagában nem benchmark, hanem mérnöki integrációs bizonyíték.

## 5.10 Reprodukálhatóság és futtatási célok

A real-lab mérési útvonal fő futtatási céljai a `final-real-measurement-package-with-provenance`, a `final-live-integration` és a `final-thesis-integration`. Ezek egymásra épülnek: előbb a mérési csomag és provenance készül el, ezt követi az integrációs kimenet, végül a dolgozati beemelésre szánt szöveges anyag.

## 5.11 Korlátok

A prototípus laboratóriumi mérési környezetre készült. Nem helyettesít éles SOC-rendszert, nem modellezi teljes körűen a Wazuh indexelési láncot, és nem bizonyít hosszú idejű üzemi teljesítményt. Az AE lab scoring minősége a feature mapping megbízhatóságától, a Wazuh-only baseline pedig az alert export és az időablakos korreláció pontosságától függ.
"""
    notes = """# 5. fejezet beillesztési megjegyzések

- A fejezet implementációs leírás, ezért ne kerüljön bele metrikai javulási állítás.
- A Word dokumentumban ellenőrizni kell az ábra- és táblázatszámokat.
- Provenance hiányában a real-lab részeket csak jövőbeli mérési helykitöltőként szabad használni.
- A live integration kimenet integrációs demonstráció, nem benchmark.
"""
    outputs = {
        "chapter5": output_dir / "chapter5_implementation_generated.md",
        "notes": output_dir / "chapter5_implementation_insert_notes.md",
    }
    outputs["chapter5"].write_text(body, encoding="utf-8")
    outputs["notes"].write_text(notes, encoding="utf-8")
    return outputs


def main() -> None:
    args = parse_args()
    outputs = generate_chapter5(Path(args.output_dir), Path(args.provenance), Path(args.live_summary))
    for path in outputs.values():
        print(f"[OK] Kimenet: {path}")


if __name__ == "__main__":
    main()

