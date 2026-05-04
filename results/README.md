# Eredménykönyvtárak felépítése

A `results/` könyvtár a mérési lánc által előállított kimenetek egységes helye. Az itt megjelenő mérési állományokat alapértelmezetten nem szabad kézzel módosítani, és nem kell Git verziókezelésbe commitolni.

Real-lab eredmény csak verified real_lab provenance és manifest mellett használható dolgozati bizonyítékként. Demo, replay vagy offline output nem real-lab bizonyíték, és a beadási mellékletben is egyértelmű eredetmegjelöléssel kezelendő.

## Végleges struktúra

A végleges szakdolgozati eredmények elvárt gyökérkönyvtára:

```text
results/final/
  final-ae-minimal-v1/
  final-ae-context-v1/
  final-baseline-stat-v1/
  final-rule-proxy-v1/
  final-baseline-wazuh-v1/
  final-hybrid-v1/
```

Az egyes alkönyvtárak egy-egy összehasonlított konfiguráció konfigurált eredménygyökerei. Amennyiben egy futtatás időbélyeges alkönyvtárat hoz létre, a szakdolgozatban felhasznált végleges futtatást egyértelműen meg kell jelölni a `run_metadata.json` és a kapcsolódó dokumentáció alapján.

A prototípus jelen változatában a `train_ae.py` minden autoencoder futtatásnál időbélyeges alkönyvtárat hoz létre a konfigurált `paths.results_dir` alatt. Példa:

```text
results/final/final-ae-minimal-v1/ae_v1_YYYYMMDD_HHMMSS/
```

A prototípus jelen változatában az `ml/src/eval.py` viszonyítási alap (baseline) futtatásoknál szintén időbélyeges alkönyvtárat hoz létre a megadott `--results-dir` alatt, például:

```text
results/final/final-baseline-stat-v1/baseline_stat_YYYYMMDD_HHMMSS/
```

## Könyvtárankénti elvárt fájlok

### `results/final/final-ae-minimal-v1/`

Az autoencoder minimális jellemzőkészlettel futtatott végleges konfigurációjának eredményei.

Elvárt fájlok:

- `metrics_summary.csv`
- `predictions.csv`
- `threshold_curve.csv`
- `top_feature_errors.csv`
- `confusion_matrix.png`, ha külön ábrageneráló lépés előállítja
- `score_distribution.png`, ha külön ábrageneráló lépés előállítja
- `roc_curve.png`, ha külön ábrageneráló lépés előállítja
- `top_feature_frequency.png`, ha külön ábrageneráló lépés előállítja
- `run_metadata.json`

Megjegyzés: az AE tanítási folyamat elsősorban CSV eredményeket és rekonstrukciós hiba alapú magyarázati fájlokat állít elő. A PNG ábrák AE esetén a `plot_final_results.py` ábrageneráló lépéssel állíthatók elő a `predictions.csv`, `threshold_curve.csv` és kapcsolódó eredményfájlok alapján.

### `results/final/final-ae-context-v1/`

Az AE-Context mérési ág eredményei. Ez a konfiguráció a minimális CIC-IDS2017 flow jellemzők mellett egyszerű, timestamp nélküli kontextusjellemzőket használ: tanító adatrészből illesztett célport- és protokollgyakoriságot, ritka célport jelzőt, valamint soronként számított csomag- és forgalmi arányokat. Ezek nem időablakos, hostalapú vagy CTI-alapú kontextusjellemzők.

Elvárt fájlok:

- `metrics_summary.csv`
- `predictions.csv`
- `threshold_curve.csv`
- `top_feature_errors.csv`
- `confusion_matrix.png`, ha külön ábrageneráló lépés előállítja
- `score_distribution.png`, ha külön ábrageneráló lépés előállítja
- `roc_curve.png`, ha külön ábrageneráló lépés előállítja
- `top_feature_frequency.png`, ha külön ábrageneráló lépés előállítja
- `run_metadata.json`

### `results/final/final-baseline-stat-v1/`

A statisztikai baseline eredményei. Ez a konfiguráció a tanítóhalmaz középpontjától mért távolság alapján képez anomáliapontszámot.

Elvárt fájlok:

- `metrics_summary.csv`
- `predictions.csv`
- `threshold_curve.csv`
- `confusion_matrix.png`
- `score_distribution.png`
- `roc_curve.png`
- `run_metadata.json`

Nem elvárt fájl:

- `top_feature_errors.csv`, mivel a statisztikai baseline nem autoencoder rekonstrukciós jellemzőhibák alapján működik.

### `results/final/final-baseline-wazuh-v1/`

A natív Wazuh exportált riasztások baseline kiértékelésének eredményei. Ez a könyvtár csak akkor tartalmaz validált mérési eredményt, ha rendelkezésre áll megfelelő címkézett Wazuh export.

Elvárt fájlok:

- `metrics_summary.csv`
- `predictions.csv`
- `threshold_curve.csv`
- `confusion_matrix.png`
- `score_distribution.png`
- `roc_curve.png`
- `run_metadata.json`

Nem elvárt fájl:

- `top_feature_errors.csv`, mivel ez a baseline nem autoencoder modellből származó jellemző-rekonstrukciós hibát mér.

### `results/final/final-rule-proxy-v1/`

A szabályalapú proxy baseline eredményei. Ez kontrollált, flow-alapú Wazuh-szerű szabályproxy, nem natív Wazuh teljesítménymérés.

Elvárt fájlok:

- `metrics_summary.csv`
- `predictions.csv`
- `threshold_curve.csv`
- `threshold_curve.png`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `score_distribution.png`
- `roc_curve.png`
- `run_metadata.json`

Nem elvárt fájl:

- `top_feature_errors.csv`, mivel ez a baseline nem autoencoder modellből származó jellemző-rekonstrukciós hibát mér.

### `results/final/final-hybrid-v1/`

A hibrid AE + rule_proxy kiértékelés eredményei. Ez kontrollált offline kiértékelés, amely az AE-Minimal és a szabályalapú proxy baseline predikcióit azonos teszthalmaz-sorrend mellett kombinálja.

Elvárt fájlok:

- `metrics_summary.csv`
- `predictions.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `score_distribution.png`
- `roc_curve.png`
- `run_metadata.json`

Opcionális vagy nem alkalmazható fájl:

- `top_feature_errors.csv`, csak akkor releváns, ha a hibrid eredményben az AE rekonstrukciós hibák magyarázati célból továbbvezetésre kerülnek.
- `threshold_curve.csv`, mivel a jelenlegi hibrid döntés fix unióalapú szabályt használ.

## A szakdolgozatban felhasznált fájlok

A szakdolgozat táblázataihoz és ábráihoz elsősorban az alábbi fájlok használhatók:

- `metrics_summary.csv`: konfigurációk összehasonlítása precision, recall, F1, false positive rate és kapcsolódó metrikák alapján.
- `predictions.csv`: mintaszintű elemzés, alert count számítás, hibás besorolások vizsgálata.
- `threshold_curve.csv`: küszöbérzékenységi táblázatok és ábrák forrása.
- `confusion_matrix.png`: konfigurációnkénti confusion matrix ábra.
- `score_distribution.png`: anomáliapontszámok eloszlásának bemutatása.
- `roc_curve.png`: ROC-görbe, ahol a pontszám és a bináris címke alapján értelmezhető.
- `top_feature_errors.csv`: AE-alapú magyarázhatósági táblázatok forrása.
- `top_feature_frequency.png`: AE-alapú magyarázhatósági ábra a leggyakoribb elsődleges jellemzőkről.
- `run_metadata.json`: reprodukálhatósági adatok, például bemeneti könyvtárak, futtatási azonosítók és konfigurációs hivatkozások.

## Automatikusan előállított eredményfájlok kezelése

A következő fájlok automatikusan előállított eredményfájlok, ezért nem szabad őket kézzel szerkeszteni:

- `metrics_summary.csv`
- `predictions.csv`
- `threshold_curve.csv`
- `top_feature_errors.csv`
- `confusion_matrix.png`
- `score_distribution.png`
- `roc_curve.png`
- `top_feature_frequency.png`
- `run_metadata.json`

Ha egy mérés hibás konfigurációval vagy hibás bemenettel készült, a helyes eljárás az adott mérési parancs újrafuttatása, nem pedig az eredményfájlok kézi javítása.

Kézzel szerkeszthető kiegészítő fájl csak akkor kerüljön az eredménykönyvtárba, ha egyértelműen dokumentációs célú, például:

- `notes.md`
- `selected_run.md`

Ezekben jelezni kell, hogy melyik futtatás került be a szakdolgozat végleges ábráiba és táblázataiba.

## Végleges ábrák névkonvenciója

A szakdolgozatba átemelt vagy abból hivatkozott végleges ábrák javasolt névkonvenciója:

```text
fig_<config>_<content>.png
```

Példák:

- `fig_ae_minimal_confusion_matrix.png`
- `fig_ae_minimal_score_distribution.png`
- `fig_ae_minimal_threshold_curve.png`
- `fig_baseline_stat_confusion_matrix.png`
- `fig_rule_proxy_confusion_matrix.png`
- `fig_baseline_wazuh_roc_curve.png`
- `fig_hybrid_confusion_matrix.png`

Összehasonlító ábrák esetén:

```text
fig_comparison_<content>.png
```

Példák:

- `fig_comparison_f1.png`
- `fig_comparison_alert_count.png`
- `fig_comparison_false_positive_rate.png`

A május 4-i összehasonlító gyűjtőlépés a `results/final/comparison/` könyvtárban a következő végleges összehasonlító állományokat állítja elő:

- `metrics_comparison.csv`
- `metrics_comparison.md`
- `fig_comparison_precision_recall_f1.png`
- `fig_comparison_false_positive_rate.png`
- `fig_comparison_alert_count.png`

A névben a konfiguráció legyen kisbetűs, aláhúzással tagolt, és egyezzen a végleges mérési konfiguráció rövid nevével: `ae_minimal`, `ae_context`, `baseline_stat`, `rule_proxy`, `baseline_wazuh`, `hybrid`.

## Reprodukálhatósági megjegyzés

Minden végleges eredményhez meg kell őrizni a kapcsolódó konfigurációt, a futtatási metaadatokat és a parancsot, amellyel az eredmény készült. A szakdolgozatban szereplő táblázatok és ábrák csak olyan futtatásból származzanak, amelyhez rendelkezésre áll:

- a használt YAML konfiguráció,
- a `run_metadata.json`,
- a releváns `metrics_summary.csv`,
- a kapcsolódó `predictions.csv`,
- szükség esetén a modellfájl és a küszöbértékeket tartalmazó fájl.

Az eredmények összehasonlításakor azonos adathalmazverziót, azonos tanító, validációs, kalibrációs és teszt adatrészekre bontást, valamint dokumentált random seedet kell használni. A végleges konfigurációkban a random seed értéke `42`.

## Demonstrációs riportkimenetek

A május 7-i prototípus-réteg a végleges mérési eredményektől elkülönített, kis méretű demonstrációs riportokat is előállíthat:

```text
reports/scored_events.jsonl
reports/final/security_report.md
reports/final/security_report.html
reports/final/dashboard_summary.csv
```

Ezek a fájlok a scoring szolgáltatás, a batch scoring és a szakértői jelentés bemutatását szolgálják. Nem helyettesítik a `results/final/` alatti validált mérési eredményeket, és nem tekintendők külön benchmark futásnak. A dolgozatban felhasználhatók az implementációs feldolgozási lánc szemléltetésére, ha egyértelműen demonstrációs kimenetként szerepelnek.

## Lab/replay esettanulmányos kimenetek

A lab/replay validáció kimenetei a `reports/lab/` könyvtárban találhatók:

```text
reports/lab/lab_scored_events.jsonl
reports/lab/lab_scored_events.csv
reports/lab/case_study_summary.md
reports/lab/case_study_<scenario>.md
reports/lab/scenario_summary.csv
reports/lab/scenario_risk_matrix.csv
reports/lab/lab_timeline.png
reports/lab/risk_level_distribution.png
```

A dolgozatba szánt ábrák rendezett másolatai:

```text
reports/final/thesis_figures/
reports/final/thesis_figures/figure_manifest.csv
```

Ezek demonstrációs validációs kimenetek. Nem helyettesítik a végleges mérési benchmarkot, és nem jelentenek natív Wazuh teljesítménymérést.

## Teljesítménymérési kimenetek

A batch scoring lokális/labor teljesítménymérési kimenetei a `reports/performance/` könyvtárban találhatók:

```text
reports/performance/benchmark_results.csv
reports/performance/benchmark_summary.md
reports/performance/system_info.json
reports/performance/performance_report.md
reports/performance/performance_report.html
reports/performance/latency_by_batch_size.png
reports/performance/throughput_by_batch_size.png
reports/performance/scoring_time_distribution.png
```

Ezek a fájlok a batch scoring feldolgozási idő, késleltetés és áteresztőképesség lokális/labor mérését dokumentálják. Nem éles üzemi benchmarkot, nem natív Wazuh indexelési teljesítményt és nem teljes SIEM end-to-end mérést jelentenek.
