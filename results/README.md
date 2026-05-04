# Eredménykönyvtárak felépítése

A `results/` könyvtár a szakdolgozat végleges mérési eredményeinek egységes tárolására szolgál. Az itt megjelenő mérési állományok a pipeline által előállított kimenetek; ezeket alapértelmezetten nem szabad kézzel módosítani.

## Végleges struktúra

A végleges szakdolgozati eredmények elvárt gyökérkönyvtára:

```text
results/final/
  final-ae-minimal-v1/
  final-ae-context-v1/
  final-baseline-stat-v1/
  final-baseline-wazuh-v1/
  final-hybrid-v1/
```

Az egyes alkönyvtárak egy-egy összehasonlított konfiguráció konfigurált eredménygyökerei. Amennyiben egy futtatás időbélyeges alkönyvtárat hoz létre, a szakdolgozatban felhasznált végleges futtatást egyértelműen meg kell jelölni a `run_metadata.json` és a kapcsolódó dokumentáció alapján.

A prototípus jelen változatában a `train_ae.py` minden autoencoder futtatásnál időbélyeges alkönyvtárat hoz létre a konfigurált `paths.results_dir` alatt. Példa:

```text
results/final/final-ae-minimal-v1/ae_v1_YYYYMMDD_HHMMSS/
```

A prototípus jelen változatában az `ml/src/eval.py` baseline futtatásoknál szintén időbélyeges alkönyvtárat hoz létre a megadott `--results-dir` alatt, például:

```text
results/final/final-baseline-stat-v1/baseline_stat_YYYYMMDD_HHMMSS/
```

## Könyvtárankénti elvárt fájlok

### `results/final/final-ae-minimal-v1/`

Az autoencoder minimális feature-készlettel futtatott végleges konfigurációjának eredményei.

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

Megjegyzés: az AE tanító pipeline elsősorban CSV eredményeket és rekonstrukciós hiba alapú magyarázati fájlokat állít elő. A PNG ábrák AE esetén a `plot_final_results.py` ábrageneráló lépéssel állíthatók elő a `predictions.csv`, `threshold_curve.csv` és kapcsolódó eredményfájlok alapján.

### `results/final/final-ae-context-v1/`

Az AE-Context mérési ág eredményei. Ez a konfiguráció a minimális CIC-IDS2017 flow feature-k mellett egyszerű, timestamp nélküli context feature-öket használ: train splitből illesztett célport- és protokollgyakoriságot, ritka célport jelzőt, valamint soronként számított csomag- és forgalmi arányokat. Ezek nem időablakos, hostalapú vagy CTI-alapú context feature-ök.

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

- `top_feature_errors.csv`, mivel a statisztikai baseline nem autoencoder rekonstrukciós feature-hibák alapján működik.

### `results/final/final-baseline-wazuh-v1/`

A Wazuh vagy Wazuh-szerű exportált riasztások baseline kiértékelésének eredményei.

Elvárt fájlok:

- `metrics_summary.csv`
- `predictions.csv`
- `threshold_curve.csv`
- `confusion_matrix.png`
- `score_distribution.png`
- `roc_curve.png`
- `run_metadata.json`

Nem elvárt fájl:

- `top_feature_errors.csv`, mivel ez a baseline nem autoencoder modellből származó feature-rekonstrukciós hibát mér.

### `results/final/final-hybrid-v1/`

A hibrid AE + Wazuh kiértékelés eredményei. Ez a lépés tervezett implementációhoz kötött, ezért a könyvtár csak akkor tartalmaz végleges eredményeket, ha a hibrid kiértékelő pipeline már elkészült és validált.

Elvárt fájlok a hibrid implementáció elkészülte után:

- `metrics_summary.csv`
- `predictions.csv`
- `threshold_curve.csv`, ha a hibrid döntés küszöbfüggő pontszámot is használ
- `confusion_matrix.png`
- `score_distribution.png`, ha hibrid pontszám is rendelkezésre áll
- `roc_curve.png`, ha folytonos hibrid pontszám alapján értelmezhető
- `run_metadata.json`

Opcionális vagy nem alkalmazható fájl:

- `top_feature_errors.csv`, csak akkor releváns, ha a hibrid eredményben az AE rekonstrukciós hibák magyarázati célból továbbvezetésre kerülnek.

## A szakdolgozatban felhasznált fájlok

A szakdolgozat táblázataihoz és ábráihoz elsősorban az alábbi fájlok használhatók:

- `metrics_summary.csv`: konfigurációk összehasonlítása precision, recall, F1, false positive rate és kapcsolódó metrikák alapján.
- `predictions.csv`: mintaszintű elemzés, alert count számítás, hibás besorolások vizsgálata.
- `threshold_curve.csv`: küszöbérzékenységi táblázatok és ábrák forrása.
- `confusion_matrix.png`: konfigurációnkénti confusion matrix ábra.
- `score_distribution.png`: anomáliapontszámok eloszlásának bemutatása.
- `roc_curve.png`: ROC-görbe, ahol a pontszám és a bináris címke alapján értelmezhető.
- `top_feature_errors.csv`: AE-alapú magyarázhatósági táblázatok forrása.
- `top_feature_frequency.png`: AE-alapú magyarázhatósági ábra a leggyakoribb top feature értékekről.
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

A névben a konfiguráció legyen kisbetűs, aláhúzással tagolt, és egyezzen a végleges mérési konfiguráció rövid nevével: `ae_minimal`, `ae_context`, `baseline_stat`, `baseline_wazuh`, `hybrid`.

## Reprodukálhatósági megjegyzés

Minden végleges eredményhez meg kell őrizni a kapcsolódó konfigurációt, a futtatási metaadatokat és a parancsot, amellyel az eredmény készült. A szakdolgozatban szereplő táblázatok és ábrák csak olyan futtatásból származzanak, amelyhez rendelkezésre áll:

- a használt YAML konfiguráció,
- a `run_metadata.json`,
- a releváns `metrics_summary.csv`,
- a kapcsolódó `predictions.csv`,
- szükség esetén a modellfájl és a küszöbértékeket tartalmazó fájl.

Az eredmények összehasonlításakor azonos adathalmazverziót, azonos train/validation/calibration/test felosztást és dokumentált random seedet kell használni. A végleges konfigurációkban a random seed értéke `42`.
