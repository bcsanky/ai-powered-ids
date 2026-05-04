# Végleges mérési munkafolyamat

A szakdolgozat végleges mérési folyamata a projektben rögzített parancsokra, a `Makefile` célokra, az `ml/src/build_dataset.py`, `ml/src/train_ae.py` és `ml/src/eval.py` belépési pontokra, valamint az `experiments/final/` alatti végleges konfigurációkra épül.

A cél egy reprodukálható mérési lánc kialakítása, amelyben az autoencoder-alapú konfigurációk, a statisztikai viszonyítási alap (baseline), a szabályalapú proxy baseline, a natív Wazuh exporttal futtatható opcionális baseline és az offline hibrid kiértékelés egységesen dokumentált eredményfájlokat állít elő.

## 1. Szükséges bemeneti adatok

A mérési folyamat elsődleges bemenete a CIC-IDS2017 nyers CSV formátumú adathalmaza:

```text
data/raw/cicids2017/*.csv
```

Az adatépítési folyamat a nyers CSV fájlokból a következő feldolgozott szeleteket állítja elő:

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

A `dev_sample` kizárólag technikai validációra szolgál. Bekapcsolt állapotban az adatépítés a tisztítás után, de a tanító, validációs, kalibrációs és teszt adatrészekre bontás előtt determinisztikus mintavételt végez a `random_seed` alapján, és lehetőség szerint megőrzi mind a benign, mind a támadó osztályt. Az így készült gyors futtathatósági ellenőrzés eredményei nem használhatók végleges szakdolgozati mérési eredményként.

A natív Wazuh baseline külön bemenetet igényel: egy Wazuh exportot, amely tartalmazza a valós címkét és a Wazuh riasztási vagy predikciós mezőit. Támogatott formátumok a prototípus jelen változatában az `ml/src/eval.py` alapján:

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

Az AE-Minimal a végleges minimális jellemzőkészletre épül. A tanítás parancsa:

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

Az `ae_minimal_smoke.yaml` konfiguráció csak a mérési lánc futtathatóságának gyors ellenőrzésére szolgál. A szakdolgozati összehasonlító táblázatokban és ábrákban a teljes adatos `ae_minimal.yaml` vagy egy külön egyértelműen jelölt végleges futtatás eredményei használhatók.

Elvárt fő kimeneti gyökerek:

```text
results/final/final-ae-minimal-v1/
```

Megjegyzés: a prototípus jelen változatában a `train_ae.py` minden futtatásnál időbélyeges alkönyvtárat hoz létre a konfigurációban megadott modellkimeneti és eredménykönyvtár alatt.

## 4. AE-Context tanítási parancs

Az AE-Context mérési ág a minimális CIC-IDS2017 flow jellemzők mellett egyszerű, timestamp nélküli kontextusjellemzőket is használ. Az `experiments/final/ae_context.yaml` konfigurációban a `features.context_enabled: true` kapcsoló aktiválja ezeket az előállított numerikus jellemzőket:

- `destination_port_frequency`
- `protocol_frequency`
- `is_rare_destination_port`
- `packet_ratio`
- `bytes_packets_ratio`

A `destination_port_frequency`, `protocol_frequency` és `is_rare_destination_port` tanító adatrészből illesztett gyakorisági statisztikákon alapul. A validációs, kalibrációs és teszt adatrészben megjelenő, tanító adatrészben nem látott célportok vagy protokollok `0.0` gyakoriságot kapnak. A `packet_ratio` és `bytes_packets_ratio` soronként számított arányok, ezért nem igényelnek globális statisztikát.

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
results/final/final-ae-context-v1/
```

Fontos értelmezési szabály: az AE-Context nem használ timestamphez kötött időablakos aggregációkat. A prototípus jelen változatában a kontextusjellemzők képzése tanító adatrészből illesztett gyakorisági és soronkénti arányalapú jellemzőket használ, ezért stabilan futtatható a meglévő CIC-IDS2017 flow adatokon.

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

## 6. Natív Wazuh baseline kiértékelési parancs

A natív Wazuh baseline kiértékelését a meglévő `ml/src/eval.py` támogatja a következő argumentumokkal:

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

A natív Wazuh baseline eredményeinek értelmezésekor jelezni kell, hogy ez riasztás- vagy logorientált baseline, míg az AE-Minimal CIC-IDS2017 flow jellemzőkön tanul. Ha nincs megfelelő címkézett Wazuh export, ez az ág nem tekinthető validált mérési eredménynek.

## 7. Szabályalapú proxy baseline

Ha nem áll rendelkezésre címkézett natív Wazuh export, a mérési lánc egy kontrollált, flow-alapú szabályproxy baseline-t használ. Ez nem natív Wazuh teljesítménymérés, hanem Wazuh-szerű szabályalapú export, amely a meglévő `ml/src/eval.py --baseline wazuh` kiértékelési útvonalon futtatható.

A proxy export előállítása:

```bash
python3 -m ml.src.create_rule_proxy_export \
  --data-dir data/processed/final/ae_minimal \
  --output-dir data/processed/final/rule_proxy \
  --threshold-quantile 0.95
```

Kiértékelés:

```bash
python3 -m ml.src.eval \
  --baseline wazuh \
  --data-dir data/processed/final/rule_proxy \
  --results-dir results/final/final-rule-proxy-v1 \
  --wazuh-input data/processed/final/rule_proxy/wazuh_like_rule_eval.csv
```

Makefile cél:

```bash
make final-rule-proxy
```

A szabályproxy pontszáma soronként a legnagyobb abszolút standardizált numerikus jellemzőérték. A küszöb a validációs adatrész pontszámeloszlásának 0,95 kvantilise, így a teszt címkéi nem vesznek részt a szabály illesztésében.

## 8. Offline hibrid kiértékelési lépés

A hibrid kiértékelés az AE-Minimal predikciókat és a rule_proxy predikciókat kombinálja. A döntési logika:

```text
hybrid_pred = ae_pred == 1 OR rule_pred == 1
```

A hibrid pontszám a min-max normalizált AE-pontszám és a min-max normalizált rule_proxy pontszám maximuma. A kiértékelés azonos teszthalmaz-sorrendet feltételez az AE-Minimal és rule_proxy `predictions.csv` fájlokban; ez kontrollált offline mérés, nem éles eseménykorreláció.

Futtatás:

```bash
make final-hybrid
```

Közvetlen parancs:

```bash
python3 -m ml.src.hybrid_eval \
  --ae-root results/final/final-ae-minimal-v1 \
  --rule-root results/final/final-rule-proxy-v1 \
  --output-dir results/final/final-hybrid-v1
```

## 9. Elvárt eredményfájlok

Az alábbi eredményfájlok szolgálnak a szakdolgozati mérés alapjául. Nem minden fájlt ugyanaz a script állít elő minden konfiguráció esetén.

| Fájl | Előállító lépés | Tartalom | Megjegyzés |
|---|---|---|---|
| `metrics_summary.csv` | AE tanítás, baseline kiértékelés és hibrid kiértékelés | Fő mérőszámok: precision, recall, F1, ROC-AUC, mintaszámok, küszöbinformációk | Az összehasonlító táblázatok elsődleges forrása. |
| `predictions.csv` | AE tanítás, baseline kiértékelés és hibrid kiértékelés | Mintaszintű pontszámok, címkék és predikciók | Alert count és részletes hibaelemzés számítható belőle. |
| `threshold_curve.csv` | AE tanítás és baseline kiértékelés | Küszöbértékekhez tartozó precision, recall és F1 | Küszöbérzékenységi elemzéshez. |
| `top_feature_errors.csv` | AE tanítás | Jellemzőcsoportonkénti rekonstrukciós hiba | Autoencoder magyarázhatósági kiegészítés; baseline kiértékelés nem állítja elő. |
| `confusion_matrix.png` | Baseline kiértékelés és AE ábragenerálás | Konfúziós mátrix ábra | Baseline futtatásokhoz az `eval.py`, AE futtatásokhoz a `plot_final_results.py` állítja elő. |
| `score_distribution.png` | Baseline kiértékelés és AE ábragenerálás | Benign és támadó pontszámeloszlások | Baseline futtatásokhoz az `eval.py`, AE futtatásokhoz a `plot_final_results.py` állítja elő. |
| `roc_curve.png` | Baseline kiértékelés és AE ábragenerálás | ROC-görbe | Csak akkor értelmezhető, ha mindkét osztály jelen van. |
| `top_feature_frequency.png` | AE ábragenerálás | A leggyakoribb elsődleges jellemzők oszlopdiagramja | Autoencoder magyarázhatósági kiegészítés. |
| `run_metadata.json` | AE tanítás és baseline kiértékelés | Futtatási metaadatok, input- és output-útvonalak | Reprodukálhatósági dokumentációhoz. |

Az AE tanítás CSV-alapú eredményeket és modellfájlokat ment. A baseline kiértékelés ezen felül több PNG ábrát is előállít. Az AE eredményekhez a `plot_final_results.py` készíti el a `confusion_matrix.png`, `score_distribution.png`, `threshold_curve.png`, `roc_curve.png` és `top_feature_frequency.png` ábrákat a futtatási eredménykönyvtárban.

## 10. Május 4-6. értékelési és validációs lépések

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

A `final-day4` gyors validációs és ábrafrissítő cél: nem épít új adatkészletet és nem indít új autoencoder tanítást. A `final-day5` ezzel szemben hosszú futtatási cél, amely az AE-Context adatépítést, teljes tanítást, AE ábragenerálást és összehasonlítást egymás után futtatja:

```bash
make final-day5
```

Ezt a célt csak akkor célszerű indítani, ha rendelkezésre áll a teljes AE-Context tanításhoz szükséges idő és számítási erőforrás.

A május 6-i cél nem indít új AE-Minimal vagy AE-Context teljes tanítást; csak a szabályproxy exportot, a proxy baseline kiértékelést, a hibrid offline kiértékelést és az összehasonlítást frissíti:

```bash
make final-day6
```

## 11. Kapcsolat a szakdolgozati ábrákkal és táblázatokkal

A kimeneti fájlok az alábbi módon használhatók fel a szakdolgozatban:

| Dolgozati elem | Forrásfájl | Felhasználás |
|---|---|---|
| Konfigurációk összehasonlító táblázata | `metrics_summary.csv` | Precision, recall, F1, ROC-AUC és mintaszámok konfigurációnkénti bemutatása. |
| Confusion matrix ábra | `confusion_matrix.png` vagy `confusion_matrix.csv` | A true positive, false positive, true negative és false negative értékek szemléltetése. |
| Küszöbérzékenységi ábra | `threshold_curve.csv`, illetve baseline esetén `threshold_curve.png` | A küszöbválasztás hatásának bemutatása precision, recall és F1 szerint. |
| Pontszámeloszlás ábra | `score_distribution.png` vagy `predictions.csv` | Benign és támadó minták anomáliapontszám-eloszlásának összehasonlítása. |
| ROC-görbe | `roc_curve.png` | Detektálási képesség vizuális értékelése különböző küszöbök mellett. |
| Alert count táblázat | `predictions.csv` | A pozitív predikciók számának összesítése konfigurációnként. |
| Autoencoder magyarázhatósági táblázat | `top_feature_errors.csv` | A legnagyobb rekonstrukciós hibát adó jellemzőcsoportok bemutatása. |
| Reprodukálhatósági melléklet | `run_metadata.json`, `train_config.json`, `thresholds.json` | Konfigurációk, futtatási útvonalak, küszöbök és metaadatok dokumentálása. |
| Végleges összehasonlító táblázat | `results/final/comparison/metrics_comparison.md` | A fő konfigurációk egységes metrikáinak szakdolgozatba átemelhető táblázata. |
| Végleges összehasonlító ábrák | `results/final/comparison/fig_comparison_*.png` | Precision/recall/F1, false positive rate és alert count összehasonlítása. |

A szakdolgozatban az eredményeket óvatosan kell értelmezni: a CIC-IDS2017 flow-alapú mérés, a szabályalapú proxy baseline, az opcionális natív Wazuh baseline és az offline hibrid fúzió eltérő adatmodellre és eltérő feltételezésekre épülhet. Emiatt az összehasonlítás célja elsősorban a módszertani és architekturális különbségek bemutatása, nem pedig általános érvényű éles üzemi IDS teljesítménygarancia megfogalmazása.

## 12. Május 7-i demonstrációs scoring és riportlépések

A május 7-i kiegészítés célja, hogy a validált AE-Minimal modellre építve bemutatható prototípus-réteg készüljön a diplomamunka implementációs fejezetéhez. Ez nem indít új autoencoder tanítást, hanem a meglévő modellfájlt, preprocess állományt és küszöbfájlt használja pontozásra.

A FastAPI scoring szolgáltatás `/score` végpontja nem fix mintaértéket ad vissza. Ha a modell, preprocess vagy küszöbfájl nem érhető el, a `/health` válaszban a `model_loaded: false` státusz jelenik meg, a `/score` pedig hibával tér vissza. Így a szolgáltatás nem állít elő félrevezető anomáliapontszámot hiányzó modell mellett.

A parancssori batch scoring ugyanazt a modellalapú pontozási logikát használja:

```bash
make score-sample-events
```

Elvárt kimenet:

```text
reports/scored_events.jsonl
```

A szakértői jelentés és dashboard összefoglaló előállítása:

```bash
make generate-security-report
```

Elvárt kimenetek:

```text
reports/final/security_report.md
reports/final/security_report.html
reports/final/dashboard_summary.csv
```

A teljes május 7-i demonstrációs cél:

```bash
make final-day7
```

Ez a `final-validate`, `score-sample-events` és `generate-security-report` lépéseket futtatja. A cél nem épít új adatkészletet, nem indít AE-Minimal vagy AE-Context tanítást, és nem módosítja a végleges mérési eredményeket. A riport demonstrációs és dolgozati összefoglaló célú; nem éles SOC incidensjelentés és nem új benchmark mérés.

## 13. Május 8-i lab/replay esettanulmányos lépések

A május 8-i réteg kontrollált replay eseménysort használ a scoring és riportkészítési lánc bemutatására. A bemeneti események nem natív Wazuh exportból származnak, és nem tekinthetők éles SOC-validációnak.

Bemeneti fájlok:

```text
examples/lab/lab_events.jsonl
examples/lab/lab_events.csv
```

Lab események pontozása:

```bash
make score-lab-events
```

Case study riportok és ábrák előállítása:

```bash
make generate-case-studies
```

Dolgozatba szánt ábrák rendezése:

```bash
make collect-thesis-figures
```

Teljes május 8-i cél:

```bash
make final-day8
```

A `final-day8` nem indít új autoencoder tanítást és nem épít új végleges adatkészletet. Csak a lab/replay scoringot, az esettanulmányos riportokat, az ábrák rendezését és a szakértői riport frissítését futtatja.

Fő kimenetek:

```text
reports/lab/lab_scored_events.jsonl
reports/lab/lab_scored_events.csv
reports/lab/case_study_summary.md
reports/lab/lab_timeline.png
reports/lab/risk_level_distribution.png
reports/final/thesis_figures/figure_manifest.csv
```

Ezek a kimenetek elsősorban az 5. fejezet implementációs és demonstrációs részeihez használhatók. A 6. fejezet benchmark jellegű összehasonlítása továbbra is a `results/final/` alatti validált futtatási könyvtárakra épül.

## 14. Május 9-i teljesítménymérési lépések

A május 9-i mérési réteg a meglévő batch scoring feldolgozási lánc lokális/labor teljesítményét vizsgálja. Nem indít új modelltanítást, nem épít új adatkészletet, és nem tekinthető éles üzemi vagy natív Wazuh teljesítménymérésnek.

Benchmark futtatása:

```bash
make benchmark-scoring
```

Performance ábrák előállítása:

```bash
make plot-performance
```

Teljesítményriport készítése:

```bash
make generate-performance-report
```

Teljes május 9-i cél:

```bash
make final-day9
```

Fő kimenetek:

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

A `collect-thesis-figures` cél a performance ábrákat is átmásolja a dolgozatba rendezett ábrakönyvtárba.
