# Május 8-i lab/replay validáció és esettanulmányok

## Dátum és branch

- Dátum: 2026-05-08
- Branch: `thesis/final`

## Napi cél

A napi cél laborkörnyezeti, replay-alapú demonstrációs réteg készítése volt a validált AE-Minimal modellre épülő scoring folyamathoz. A cél nem új modell fejlesztése, hanem normál, port scan, SSH brute force jellegű és kombináltan gyanús események pontozása, esettanulmányos összefoglalása és dolgozatba beemelhető ábrák rendezése.

## Előzetes ellenőrzés

- A branch ellenőrzése sikeres volt: `thesis/final`.
- A május 7-i dokumentációk, batch scoring script, riportgenerátor és sample scoring bemenetek rendelkezésre álltak.

## Futtatott parancsok

```bash
make final-validate
make score-lab-events
make generate-case-studies
make collect-thesis-figures
make final-day8
```

Eredmény:

- `make final-validate`: sikeres, 39 teszt lefutott, 1 nem blokkoló sklearn konvergencia figyelmeztetéssel.
- `make score-lab-events`: sikeres, 20 kontrollált lab/replay esemény pontozása megtörtént.
- `make generate-case-studies`: sikeres, case study Markdown fájlok, CSV összesítések és PNG ábrák létrejöttek.
- `make collect-thesis-figures`: sikeres, 7 ábra átmásolva, hiányzó ábra nem volt.
- `make final-day8`: sikeres, a validációs, scoring, case study, ábragyűjtő és riportfrissítő lépések egymás után lefutottak.

## Létrejött bemeneti fájlok

- `examples/lab/lab_events.jsonl`
- `examples/lab/lab_events.csv`
- `examples/lab/README.md`

A bemenet 20 kontrollált replay eseményt tartalmaz a következő szcenáriókkal:

- `benign_activity`
- `port_scan`
- `ssh_bruteforce`
- `combined_suspicious`

## Létrejött pontozott fájlok

- `reports/lab/lab_scored_events.jsonl`
- `reports/lab/lab_scored_events.csv`

A batch scoring 20 eseményt pontozott. A kimenet megőrzi a `timestamp`, `scenario`, `description` és `expected_behavior` mezőket.

## Létrejött case study fájlok

- `reports/lab/case_study_summary.md`
- `reports/lab/case_study_benign_activity.md`
- `reports/lab/case_study_port_scan.md`
- `reports/lab/case_study_ssh_bruteforce.md`
- `reports/lab/case_study_combined_suspicious.md`
- `reports/lab/scenario_summary.csv`
- `reports/lab/scenario_risk_matrix.csv`

## Létrejött ábrák

- `reports/lab/lab_timeline.png`
- `reports/lab/risk_level_distribution.png`

## Thesis figures manifest

Létrejött:

- `reports/final/thesis_figures/figure_manifest.csv`

Az ábragyűjtés 7 ábrát másolt át és nem jelzett hiányzó ábrát. A manifest rögzíti a forrásfájlt, a javasolt dolgozati fejezetet és az átvétel státuszát.

## Korlátok

- A validáció replay-alapú demonstráció.
- Nem natív Wazuh teljesítménymérés.
- Nem éles SOC-validáció.
- Kis elemszámú bemutató eseménysor, ezért nem benchmark mérés.
- A benign_activity események magasabb kockázati prioritást kaptak; ez az AE-Minimal végleges küszöbválasztásának óvatos értelmezését igényli.

## Következő napra átadott feladatok

- Latency és throughput jellegű teljesítménymérés előkészítése.
- Az implementációs fejezet scoring és esettanulmány részeinek kidolgozása.
- A dolgozat 6. fejezetéhez végleges ábra- és táblázathivatkozások rögzítése.
