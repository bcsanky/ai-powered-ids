# Végleges mérési munkafolyamat

Ez a dokumentum a szakdolgozat végleges mérési folyamatát rögzíti. A terv az aktuális repository parancsaira, a `Makefile` célokra, az `ml/src/build_dataset.py`, `ml/src/train_ae.py` és `ml/src/eval.py` belépési pontokra, valamint az `experiments/final/` alatti végleges konfigurációkra épül.

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

A Wazuh baseline külön bemenetet igényel: egy Wazuh vagy Wazuh-szerű exportot, amely tartalmazza a valós címkét és a Wazuh riasztási vagy predikciós mezőit. Támogatott formátumok az aktuális `ml/src/eval.py` alapján:

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

Elvárt fő kimeneti gyökerek:

```text
artifacts/final/final-ae-minimal-v1/
results/final/final-ae-minimal-v1/
```

Megjegyzés: az aktuális `train_ae.py` minden futtatásnál időbélyeges alkönyvtárat hoz létre a konfigurációban megadott artifact és result gyökér alatt.

## 4. AE-Context tanítási parancs

Az AE-Context jelenleg opcionális mérési ág. A repository aktuális kódja alapján a valódi kontextusfeature-ök még nem részei az adatépítő pipeline implementációjának. Ezért az `experiments/final/ae_context.yaml` konfiguráció futtatható, de jelenleg a minimális feature-készlettel kompatibilis változatként kezelendő.

Futtatás, amennyiben a cél az opcionális, jelenlegi kóddal kompatibilis AE-Context mérés:

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

Fontos értelmezési szabály: amíg a kontextusfeature-ök nincsenek implementálva az adatépítő kódban, az AE-Context eredmény nem állítható be tényleges kontextusmodellezésként. Ebben az állapotban opcionális, kontrollkonfigurációként használható.

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

A `--wazuh-input` útvonalat a tényleges exportált fájl helyére kell állítani. Ha ez az argumentum nincs megadva, az aktuális kód a `--data-dir` könyvtárban keresi a támogatott nevű Wazuh bemeneteket, például:

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

A hibrid kiértékelés **tervezett május 6-i implementációs feladat**. A repository jelenlegi állapotában nincs `hybrid_eval.py`, és az aktuális `ml/src/eval.py` nem tartalmaz külön hibrid fúziós módot.

A tervezett hibrid lépés célja az AE predikciók és a Wazuh predikciók összekapcsolása egy stabil esemény- vagy flow-azonosító alapján. A kezdeti döntési logika:

```text
hybrid_alert = ae_prediction == 1 OR wazuh_prediction == 1
```

Tervezett parancsforma, amely csak az implementáció elkészülte után tekinthető futtathatónak:

```bash
# tervezett, jelenleg nem futtatható
python3 -m ml.src.hybrid_eval \
  --ae-predictions results/final/final-ae-minimal-v1/<run_id>/predictions.csv \
  --wazuh-predictions results/final/final-baseline-wazuh-v1/<run_id>/predictions.csv \
  --results-dir results/final/final-hybrid-v1
```

A szakdolgozatban ezt a lépést addig tervezett hibrid értékelésként kell megnevezni, amíg a megfelelő implementáció és validált kimenet nem készül el.

## 8. Elvárt eredményfájlok

Az alábbi eredményfájlok szolgálnak a szakdolgozati mérés alapjául. Nem minden fájlt ugyanaz a script állít elő minden konfiguráció esetén.

| Fájl | Előállító lépés | Tartalom | Megjegyzés |
|---|---|---|---|
| `metrics_summary.csv` | AE tanítás és baseline eval | Fő mérőszámok: precision, recall, F1, ROC-AUC, mintaszámok, küszöbinformációk | Az összehasonlító táblázatok elsődleges forrása. |
| `predictions.csv` | AE tanítás és baseline eval | Mintaszintű pontszámok, címkék és predikciók | Alert count és részletes hibaelemzés számítható belőle. |
| `threshold_curve.csv` | AE tanítás és baseline eval | Küszöbértékekhez tartozó precision, recall és F1 | Küszöbérzékenységi elemzéshez. |
| `top_feature_errors.csv` | AE tanítás | Feature-csoportonkénti rekonstrukciós hiba | Autoencoder magyarázhatósági kiegészítés; baseline eval nem állítja elő. |
| `confusion_matrix.png` | Baseline eval | Confusion matrix ábra | Az aktuális `eval.py` állítja elő baseline futtatásokhoz. |
| `score_distribution.png` | Baseline eval | Benign és támadó pontszámeloszlások | Az aktuális `eval.py` állítja elő baseline futtatásokhoz. |
| `roc_curve.png` | Baseline eval | ROC-görbe | Csak akkor értelmezhető, ha mindkét osztály jelen van. |
| `run_metadata.json` | AE tanítás és baseline eval | Futtatási metaadatok, input- és output-útvonalak | Reprodukálhatósági dokumentációhoz. |

Az AE tanítás aktuálisan CSV-alapú eredményeket és model artifactokat ment. A baseline eval ezen felül több PNG ábrát is generál. Amennyiben az AE eredményekhez is szükséges `confusion_matrix.png`, `score_distribution.png` vagy `roc_curve.png`, azt külön ábrageneráló lépésben vagy későbbi kódbővítéssel kell előállítani.

## 9. Kapcsolat a szakdolgozati ábrákkal és táblázatokkal

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

A szakdolgozatban az eredményeket óvatosan kell értelmezni: a CIC-IDS2017 flow-alapú mérés, a Wazuh logalapú baseline és a tervezett hibrid fúzió eltérő adatmodellre épülhet. Emiatt az összehasonlítás célja elsősorban a módszertani és architekturális különbségek bemutatása, nem pedig általános érvényű production IDS teljesítménygarancia megfogalmazása.

## 10. May 3 validation checklist

A május 3-i végleges kísérleti beállítás validálásához az alábbi parancsok használhatók. A `py_compile` és YAML-ellenőrző parancsok gyors statikus validációt adnak, míg a `make dataset`, `make train-ae` és `make final-eval-stat` tényleges pipeline-futtatások, ezért ezek hosszabb ideig tarthatnak.

```bash
python3 -m py_compile ml/src/build_dataset.py
python3 -m py_compile ml/src/train_ae.py
python3 -m py_compile ml/src/eval.py
```

```bash
python3 - <<'PY'
import yaml
from pathlib import Path
for p in Path("experiments/final").glob("*.yaml"):
    with open(p, "r", encoding="utf-8") as f:
        yaml.safe_load(f)
    print("[OK]", p)
PY
```

```bash
make dataset CONFIG=experiments/final/ae_minimal.yaml
make train-ae CONFIG=experiments/final/ae_minimal.yaml
make final-eval-stat
```
