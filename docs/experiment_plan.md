# Végleges mérési munkafolyamat

A szakdolgozat végleges mérési folyamata a projektben rögzített parancsokra, a `Makefile` célokra, az `ml/src/build_dataset.py`, `ml/src/train_ae.py` és `ml/src/eval.py` belépési pontokra, valamint az `experiments/final/` alatti végleges konfigurációkra épül.

A cél egy reprodukálható mérési lánc kialakítása, amelyben az autoencoder-alapú konfigurációk, a statisztikai baseline, a Wazuh-stílusú baseline és a tervezett hibrid kiértékelés egységesen dokumentált eredményfájlokat állít elő.

## 1. Szükséges bemeneti adatok

A mérési folyamat elsődleges bemenete a CIC-IDS2017 nyers CSV formátumú adathalmaza:

```text
data/raw/cicids2017/*.csv
```

Az adatépítő pipeline a nyers CSV fájlokból a következő feldolgozott szeleteket állítja elő:

```text
train.parquet
val.parquet
calib.parquet
test.parquet
preprocess.pkl
dataset_metadata.json
```

Az AE-Minimal és AE-Context futtatások a saját konfigurációjukban megadott `dataset.output_dir` könyvtárba írnak. A statisztikai baseline a már feldolgozott AE-Minimal adatokra épül.

A végleges szakdolgozati konfigurációk teljes adatos futtatást használnak. A `dev_sample` kapcsoló ezekben explicit módon kikapcsolt állapotú:

```yaml
dev_sample:
  enabled: false
  max_rows_total: null
```

A `dev_sample` kizárólag technikai validációra szolgál. Bekapcsolt állapotban az adatépítés a tisztítás után, de a train/validation/calibration/test felosztás előtt determinisztikus mintavételt végez a `random_seed` alapján, és lehetőség szerint megőrzi mind a benign, mind a támadó osztályt. Az így készült gyors futtathatósági ellenőrzés eredményei nem használhatók végleges szakdolgozati mérési eredményként.

A Wazuh baseline külön bemenetet igényel: egy Wazuh vagy Wazuh-szerű exportot, amely tartalmazza a valós címkét és a Wazuh riasztási vagy predikciós mezőit. Támogatott formátumok a prototípus jelen változatában az `ml/src/eval.py` alapján:

```text
.parquet
.csv
.jsonl
.ndjson
```

Példa bemeneti útvonal:

```text
data/processed/final/wazuh/wazuh_eval.csv
```

## 2. Dataset build parancs

Az AE-Minimal végleges adatkészletének előállítása:

```bash
make dataset CONFIG=experiments/final/ae_minimal.yaml
```

Ekvivalens közvetlen Python parancs:

```bash
python3 ml/src/build_dataset.py --config experiments/final/ae_minimal.yaml
```

Elvárt fő kimeneti könyvtár:

```text
data/processed/final/ae_minimal/
```

## 3. AE-Minimal tanítási parancs

Az AE-Minimal a végleges minimális feature-készletre épül. A tanítás parancsa:

```bash
make train-ae CONFIG=experiments/final/ae_minimal.yaml
```

Ekvivalens közvetlen Python parancs:

```bash
python3 -m ml.src.train_ae --config experiments/final/ae_minimal.yaml
```

A teljes AE-Minimal adatépítés és tanítás rövidített Makefile célon keresztül:

```bash
make final-ae-minimal
```

Gyors futtathatósági ellenőrzés kisebb mintán:

```bash
make dataset CONFIG=experiments/final/ae_minimal_smoke.yaml
make train-ae CONFIG=experiments/final/ae_minimal_smoke.yaml
```

Az `ae_minimal_smoke.yaml` konfiguráció csak a pipeline futtathatóságának gyors ellenőrzésére szolgál. A szakdolgozati összehasonlító táblázatokban és ábrákban a teljes adatos `ae_minimal.yaml` vagy egy külön egyértelműen jelölt végleges futtatás eredményei használhatók.

Elvárt fő kimeneti gyökerek:

```text
artifacts/final/final-ae-minimal-v1/
results/final/final-ae-minimal-v1/
```

Megjegyzés: a prototípus jelen változatában a `train_ae.py` minden futtatásnál időbélyeges alkönyvtárat hoz létre a konfigurációban megadott modellkimeneti és eredménykönyvtár alatt.

## 4. AE-Context tanítási parancs

Az AE-Context mérési ág a minimális CIC-IDS2017 flow feature-ök mellett egyszerű, timestamp nélküli context feature-öket is használ. Az `experiments/final/ae_context.yaml` konfigurációban a `features.context_enabled: true` kapcsoló aktiválja ezeket az előállított numerikus jellemzőket:

- `destination_port_frequency`
- `protocol_frequency`
- `is_rare_destination_port`
- `packet_ratio`
- `bytes_packets_ratio`

A `destination_port_frequency`, `protocol_frequency` és `is_rare_destination_port` train splitből illesztett gyakorisági statisztikákon alapul. A validation, calibration és test splitben megjelenő, trainben nem látott célportok vagy protokollok `0.0` gyakoriságot kapnak. A `packet_ratio` és `bytes_packets_ratio` soronként számított arányok, ezért nem igényelnek globális statisztikát.

Futtatás:

```bash
make dataset CONFIG=experiments/final/ae_context.yaml
make train-ae CONFIG=experiments/final/ae_context.yaml
```

Ekvivalens közvetlen Python parancsok:

```bash
python3 ml/src/build_dataset.py --config experiments/final/ae_context.yaml
python3 -m ml.src.train_ae --config experiments/final/ae_context.yaml
```

Elvárt fő kimeneti gyökerek:

```text
data/processed/final/ae_context/
artifacts/final/final-ae-context-v1/
results/final/final-ae-context-v1/
```

Fontos értelmezési szabály: az AE-Context nem használ timestamphez kötött időablakos aggregációkat. A prototípus jelen változatában a context feature engineering train-alapú gyakorisági és soronkénti arányalapú jellemzőket használ, ezért stabilan futtatható a meglévő CIC-IDS2017 flow adatokon.

## 5. Statisztikai baseline kiértékelési parancs

A statisztikai baseline az AE-Minimal feldolgozott adatait használja. A Makefile cél:

```bash
make final-eval-stat
```

Ekvivalens közvetlen Python parancs:

```bash
python3 -m ml.src.eval \
  --baseline stat \
  --data-dir data/processed/final/ae_minimal \
  --results-dir results/final/final-baseline-stat-v1 \
  --threshold-quantile 0.95
```

A baseline a tanítóhalmaz középpontjától mért távolság alapján képez anomáliapontszámot, majd a validációs pontszámok megadott kvantilise szerint választ küszöböt.

## 6. Wazuh baseline kiértékelési parancs

A Wazuh baseline kiértékelését a meglévő `ml/src/eval.py` támogatja a következő argumentumokkal:

```bash
python3 -m ml.src.eval \
  --baseline wazuh \
  --data-dir data/processed/final/wazuh \
  --results-dir results/final/final-baseline-wazuh-v1 \
  --wazuh-input data/processed/final/wazuh/wazuh_eval.csv
```

A `--wazuh-input` útvonalat a tényleges exportált fájl helyére kell állítani. Ha ez az argumentum nincs megadva, a prototípus jelen változata a `--data-dir` könyvtárban keresi a támogatott nevű Wazuh bemeneteket, például:

```text
wazuh_eval.parquet
wazuh_eval.csv
wazuh_eval.jsonl
wazuh_alerts.parquet
wazuh_alerts.csv
wazuh_alerts.jsonl
```

A Wazuh baseline eredményeinek értelmezésekor jelezni kell, hogy ez riasztás- vagy logorientált baseline, míg az AE-Minimal CIC-IDS2017 flow feature-ökön tanul.

## 7. Tervezett hibrid kiértékelési lépés

A hibrid kiértékelés tervezett elem. A projektben nincs külön hibrid kiértékelő modul, és az `ml/src/eval.py` nem tartalmaz külön hibrid fúziós módot.

A tervezett hibrid lépés célja az AE predikciók és a Wazuh predikciók összekapcsolása egy stabil esemény- vagy flow-azonosító alapján. A kezdeti döntési logika:

```text
hybrid_alert = ae_prediction == 1 OR wazuh_prediction == 1
```

A későbbi hibrid kiértékelő bemenete várhatóan az AE és Wazuh `predictions.csv` állománya lesz, kimenete pedig a többi mérési ággal azonos szerkezetű `results/final/final-hybrid-v1/` eredménykönyvtárba kerülhet. A szakdolgozatban ezt a lépést addig tervezett hibrid értékelésként kell megnevezni, amíg a megfelelő implementáció és validált kimenet nem készül el.

## 8. Elvárt eredményfájlok

Az alábbi eredményfájlok szolgálnak a szakdolgozati mérés alapjául. Nem minden fájlt ugyanaz a script állít elő minden konfiguráció esetén.

| Fájl | Előállító lépés | Tartalom | Megjegyzés |
|---|---|---|---|
| `metrics_summary.csv` | AE tanítás és baseline eval | Fő mérőszámok: precision, recall, F1, ROC-AUC, mintaszámok, küszöbinformációk | Az összehasonlító táblázatok elsődleges forrása. |
| `predictions.csv` | AE tanítás és baseline eval | Mintaszintű pontszámok, címkék és predikciók | Alert count és részletes hibaelemzés számítható belőle. |
| `threshold_curve.csv` | AE tanítás és baseline eval | Küszöbértékekhez tartozó precision, recall és F1 | Küszöbérzékenységi elemzéshez. |
| `top_feature_errors.csv` | AE tanítás | Feature-csoportonkénti rekonstrukciós hiba | Autoencoder magyarázhatósági kiegészítés; baseline eval nem állítja elő. |
| `confusion_matrix.png` | Baseline eval és AE ábragenerálás | Confusion matrix ábra | Baseline futtatásokhoz az `eval.py`, AE futtatásokhoz a `plot_final_results.py` állítja elő. |
| `score_distribution.png` | Baseline eval és AE ábragenerálás | Benign és támadó pontszámeloszlások | Baseline futtatásokhoz az `eval.py`, AE futtatásokhoz a `plot_final_results.py` állítja elő. |
| `roc_curve.png` | Baseline eval és AE ábragenerálás | ROC-görbe | Csak akkor értelmezhető, ha mindkét osztály jelen van. |
| `top_feature_frequency.png` | AE ábragenerálás | A leggyakoribb top feature értékek oszlopdiagramja | Autoencoder magyarázhatósági kiegészítés. |
| `run_metadata.json` | AE tanítás és baseline eval | Futtatási metaadatok, input- és output-útvonalak | Reprodukálhatósági dokumentációhoz. |

Az AE tanítás CSV-alapú eredményeket és modellfájlokat ment. A baseline eval ezen felül több PNG ábrát is előállít. Az AE eredményekhez a `plot_final_results.py` készíti el a `confusion_matrix.png`, `score_distribution.png`, `threshold_curve.png`, `roc_curve.png` és `top_feature_frequency.png` ábrákat a futtatási eredménykönyvtárban.

## 9. Május 4-i értékelési és validációs lépések

A május 4-i kiegészítések célja, hogy a szakdolgozati értékelés metrikái, ábrái és összehasonlító táblázatai egységes formában álljanak elő. A metrikaszámítás az AE tanításban és a baseline kiértékelésben azonos kiegészítő mezőket tartalmaz:

- `false_positive_rate`
- `false_negative_rate`
- `true_positive_rate`
- `true_negative_rate`
- `alert_count`

Az AE eredménykönyvtárakhoz az ábragenerálás külön parancsként futtatható:

```bash
python3 -m ml.src.plot_final_results --run-dir results/final/final-ae-minimal-v1/<run_id>
```

A legfrissebb AE-Minimal futtatás ábrái Makefile célon keresztül készíthetők el:

```bash
make final-plot-ae-minimal
```

Az AE-Context ábragenerálása csak akkor futtatható, ha már létezik időbélyeges AE-Context eredménykönyvtár:

```bash
make final-plot-ae-context
```

A végleges összehasonlító táblázat és az összehasonlító ábrák előállítása:

```bash
make final-compare
```

Ez a cél a következő fájlokat állítja elő a `results/final/comparison/` könyvtárban:

- `metrics_comparison.csv`
- `metrics_comparison.md`
- `fig_comparison_precision_recall_f1.png`
- `fig_comparison_false_positive_rate.png`
- `fig_comparison_alert_count.png`

A gyors, nem tanító jellegű validációs cél:

```bash
make final-validate
```

Ez YAML-ellenőrzést, Python fordítási ellenőrzést és unit teszteket futtat, de nem indít hosszú dataset buildet vagy autoencoder tanítást. A napi összefoglaló cél:

```bash
make final-day4
```

Ez a `final-validate`, `final-plot-ae-minimal` és `final-compare` lépéseket futtatja. Az AE-Context ábragenerálást csak akkor kapcsolja be, ha már van AE-Context eredménykönyvtár.

## 10. Kapcsolat a szakdolgozati ábrákkal és táblázatokkal

A kimeneti fájlok az alábbi módon használhatók fel a szakdolgozatban:

| Thesis elem | Forrásfájl | Felhasználás |
|---|---|---|
| Konfigurációk összehasonlító táblázata | `metrics_summary.csv` | Precision, recall, F1, ROC-AUC és mintaszámok konfigurációnkénti bemutatása. |
| Confusion matrix ábra | `confusion_matrix.png` vagy `confusion_matrix.csv` | A true positive, false positive, true negative és false negative értékek szemléltetése. |
| Küszöbérzékenységi ábra | `threshold_curve.csv`, illetve baseline esetén `threshold_curve.png` | A küszöbválasztás hatásának bemutatása precision, recall és F1 szerint. |
| Pontszámeloszlás ábra | `score_distribution.png` vagy `predictions.csv` | Benign és támadó minták anomáliapontszám-eloszlásának összehasonlítása. |
| ROC-görbe | `roc_curve.png` | Detektálási képesség vizuális értékelése különböző küszöbök mellett. |
| Alert count táblázat | `predictions.csv` | A pozitív predikciók számának összesítése konfigurációnként. |
| Autoencoder magyarázhatósági táblázat | `top_feature_errors.csv` | A legnagyobb rekonstrukciós hibát adó feature-csoportok bemutatása. |
| Reprodukálhatósági melléklet | `run_metadata.json`, `train_config.json`, `thresholds.json` | Konfigurációk, futtatási útvonalak, küszöbök és metaadatok dokumentálása. |
| Végleges összehasonlító táblázat | `results/final/comparison/metrics_comparison.md` | A fő konfigurációk egységes metrikáinak szakdolgozatba átemelhető táblázata. |
| Végleges összehasonlító ábrák | `results/final/comparison/fig_comparison_*.png` | Precision/recall/F1, false positive rate és alert count összehasonlítása. |

A szakdolgozatban az eredményeket óvatosan kell értelmezni: a CIC-IDS2017 flow-alapú mérés, a Wazuh logalapú baseline és a tervezett hibrid fúzió eltérő adatmodellre épülhet. Emiatt az összehasonlítás célja elsősorban a módszertani és architekturális különbségek bemutatása, nem pedig általános érvényű production IDS teljesítménygarancia megfogalmazása.
