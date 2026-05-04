# May 3 final experiment setup validáció

Validáció dátuma: 2026-05-03  
Branch: `thesis/final`

A május 3-i végleges kísérleti beállítás futtathatósági ellenőrzése során a final konfigurációk, a fő Python belépési pontok, az AE-Minimal adatépítés, az AE-Minimal tanítás és a statisztikai baseline futtathatósága került ellenőrzésre. A későbbi lokális validáció alapján a teljes AE-Minimal tanítás sikeresen befejeződött, és létrejöttek a hozzá tartozó artifact és result könyvtárak.

## Lefuttatott parancsok

### Branch és munkakönyvtár ellenőrzése

```bash
git branch --show-current
git status --short
```

Eredmény:

- Branch: `thesis/final`
- A validáció indításakor a munkafa tiszta volt.

### Final YAML konfigurációk ellenőrzése

```bash
python3 - <<'PY'
import yaml
from pathlib import Path
for p in sorted(Path("experiments/final").glob("*.yaml")):
    with open(p, "r", encoding="utf-8") as f:
        yaml.safe_load(f)
    print("[OK]", p)
PY
```

Eredmény: sikeres.

Ellenőrzött konfigurációk:

- `experiments/final/ae_context.yaml`
- `experiments/final/ae_minimal.yaml`
- `experiments/final/ae_minimal_smoke.yaml`
- `experiments/final/baseline_stat.yaml`
- `experiments/final/baseline_wazuh.yaml`
- `experiments/final/final_experiment.yaml`
- `experiments/final/hybrid.yaml`

### Python fordítási ellenőrzés

```bash
python3 -m py_compile ml/src/build_dataset.py ml/src/train_ae.py ml/src/eval.py
```

Eredmény: sikeres.

### Nyers adat ellenőrzése

```bash
find data/raw/cicids2017 -maxdepth 1 -type f -name "*.csv" | wc -l
```

Eredmény: `8` nyers CIC-IDS2017 CSV fájl elérhető.

### AE-Minimal dataset build

```bash
make dataset CONFIG=experiments/final/ae_minimal.yaml
```

Eredmény: sikeres.

A létrejött feldolgozott adatkészlet metaadatai:

- `experiment_id`: `final-ae-minimal-v1`
- `rows_total`: `2827876`
- `rows_train`: `1589922`
- `rows_val`: `340698`
- `rows_calib`: `448628`
- `rows_test`: `448628`
- `rows_calib_attacks`: `278278`
- `rows_calib_benign`: `170350`
- `rows_test_attacks`: `278278`
- `rows_test_benign`: `170350`
- `seed`: `42`

Létrejött fő kimeneti fájlok:

- `data/processed/final/ae_minimal/train.parquet`
- `data/processed/final/ae_minimal/val.parquet`
- `data/processed/final/ae_minimal/calib.parquet`
- `data/processed/final/ae_minimal/test.parquet`
- `data/processed/final/ae_minimal/preprocess.pkl`
- `data/processed/final/ae_minimal/preprocess_final-ae-minimal-v1.pkl`
- `data/processed/final/ae_minimal/dataset_metadata.json`

### AE-Minimal training

```bash
make train-ae CONFIG=experiments/final/ae_minimal.yaml
```

Eredmény: a későbbi lokális futtatás sikeresen befejeződött teljes AE-Minimal konfigurációval.

Ellenőrzött artifact könyvtár:

```text
artifacts/final/final-ae-minimal-v1/ae_v1_20260504_113852/
```

Ellenőrzött result könyvtár:

```text
results/final/final-ae-minimal-v1/ae_v1_20260504_113852/
```

A futás `run_metadata.json` állománya alapján:

- `run_id`: `20260504_113852`
- `experiment_id`: `final-ae-minimal-v1`
- `rows_train`: `1589922`
- `rows_val`: `340698`
- `rows_calib`: `448628`
- `rows_test`: `448628`
- `artifact_dir`: `artifacts/final/final-ae-minimal-v1/ae_v1_20260504_113852`
- `result_dir`: `results/final/final-ae-minimal-v1/ae_v1_20260504_113852`

A mentett `train_config.json` alapján a futás teljes adatos final konfigurációval készült:

- `dev_sample.enabled`: `false`
- `dev_sample.max_rows_total`: `null`
- `random_seed`: `42`

Létrejött artifact fájlok:

- `artifacts/final/final-ae-minimal-v1/ae_v1_20260504_113852/model.joblib`
- `artifacts/final/final-ae-minimal-v1/ae_v1_20260504_113852/history.json`
- `artifacts/final/final-ae-minimal-v1/ae_v1_20260504_113852/thresholds.json`
- `artifacts/final/final-ae-minimal-v1/ae_v1_20260504_113852/train_config.json`

Létrejött eredményfájlok:

- `results/final/final-ae-minimal-v1/ae_v1_20260504_113852/metrics_summary.csv`
- `results/final/final-ae-minimal-v1/ae_v1_20260504_113852/predictions.csv`
- `results/final/final-ae-minimal-v1/ae_v1_20260504_113852/threshold_curve.csv`
- `results/final/final-ae-minimal-v1/ae_v1_20260504_113852/top_feature_errors.csv`
- `results/final/final-ae-minimal-v1/ae_v1_20260504_113852/run_metadata.json`

### Statisztikai baseline kiértékelés

```bash
make final-eval-stat
```

Eredmény: sikeres.

A futás az alábbi eredménykönyvtárat hozta létre:

```text
results/final/final-baseline-stat-v1/baseline_stat_20260503_233227/
```

Létrejött kimeneti fájlok:

- `results/final/final-baseline-stat-v1/baseline_stat_20260503_233227/confusion_matrix.csv`
- `results/final/final-baseline-stat-v1/baseline_stat_20260503_233227/confusion_matrix.png`
- `results/final/final-baseline-stat-v1/baseline_stat_20260503_233227/metrics_summary.csv`
- `results/final/final-baseline-stat-v1/baseline_stat_20260503_233227/predictions.csv`
- `results/final/final-baseline-stat-v1/baseline_stat_20260503_233227/roc_curve.png`
- `results/final/final-baseline-stat-v1/baseline_stat_20260503_233227/run_metadata.json`
- `results/final/final-baseline-stat-v1/baseline_stat_20260503_233227/score_distribution.png`
- `results/final/final-baseline-stat-v1/baseline_stat_20260503_233227/threshold_curve.csv`
- `results/final/final-baseline-stat-v1/baseline_stat_20260503_233227/threshold_curve.png`

Fő baseline metrikák a `metrics_summary.csv` alapján:

- `tn`: `161952`
- `fp`: `8398`
- `fn`: `269371`
- `tp`: `8907`
- `precision`: `0.5147067321583357`
- `recall`: `0.0320075607845392`
- `f1`: `0.06026733607819123`
- `roc_auc`: `0.47581617769653184`
- `threshold`: `3.5242483973503114`
- `feature_mode`: `already_preprocessed`

Megjegyzés: a futás közben a Matplotlib ideiglenes cache könyvtárra vonatkozó figyelmeztetést írt ki, mert a default Matplotlib cache útvonal nem írható. Ez nem blokkolta az ábrák létrehozását.

## AE-Context státusz

Az `AE-Context` aktuális állapotban már nem pusztán kompatibilis konfiguráció: az adatépítő pipeline egyszerű, timestamp nélküli context feature-öket tud előállítani. Ezek port- és protokollgyakoriságon, ritka célport jelzőn, valamint forgalmi arányokon alapulnak.

Az implementált context feature-ök:

- `destination_port_frequency`
- `protocol_frequency`
- `is_rare_destination_port`
- `packet_ratio`
- `bytes_packets_ratio`

Ez továbbra sem jelent időablakos, hostalapú vagy CTI-alapú context feature engineeringet.

Az AE-Context dataset build lokálisan lefutott, és létrejött a következő metaadatfájl:

```text
data/processed/final/ae_context/dataset_metadata.json
```

A rögzített context metaadatok:

- `context_enabled`: `true`
- `context_features`:
  - `destination_port_frequency`
  - `protocol_frequency`
  - `is_rare_destination_port`
  - `packet_ratio`
  - `bytes_packets_ratio`
- `context_fit_split`: `train`
- `unknown_context_frequency`: `0.0`

Az AE-Context gyakorisági feature-ök train splitből illesztett statisztikákon alapulnak; a validation, calibration és test splitben ismeretlen portok vagy protokollok `0.0` gyakoriságot kapnak.

## Nyitva maradt hiba vagy feladat

- A hibrid kiértékelő pipeline továbbra is tervezett elem; külön hibrid kiértékelő modul és validált hibrid eredmény még nem áll rendelkezésre.
- A Wazuh baseline végleges kiértékeléséhez címkézett Wazuh vagy Wazuh-szerű export szükséges.
- Az AE-Context tanítási futás teljes befejezése külön validációs körben rögzíthető, ha a context konfiguráció eredményei is bekerülnek az összehasonlításba.

## Összegzés

Sikeresen validált elemek:

- Branch ellenőrzés.
- Final YAML konfigurációk szintaktikai érvényessége.
- Fő Python belépési pontok fordíthatósága.
- CIC-IDS2017 raw adat elérhetősége.
- Final AE-Minimal dataset build.
- Final AE-Minimal autoencoder tréning teljes befejezése.
- Final AE-Minimal artifact fájlok létrejötte.
- Final AE-Minimal eredményfájlok létrejötte.
- AE-Context dataset build és context metaadatok létrejötte.
- Final statisztikai baseline kiértékelés és ábragenerálás.

Nem lezárt elemek:

- Hibrid kiértékelés implementációja és validált kimenete.
- Wazuh baseline végleges futtatása megfelelő exportált bemenettel.

## Git hygiene update

A generált `artifacts/` és `results/final/` alatti mérési fájlok nem maradnak verziókezelve, mert a `.gitignore` kizárja ezeket az útvonalakat. Ezek futtatási artefaktok, amelyek egy adott mérési futás konkrét kimeneteit tartalmazzák, például modelleket, küszöböket, predikciókat, metrikákat, ábrákat és futtatási metaadatokat.

A repository-ban a hosszú távon karbantartandó elemek maradnak verziókezelve:

- konfigurációk,
- dokumentációk,
- futtatási leírások,
- validációs jegyzőkönyvek.

A végleges mérési eredmények a szakdolgozati ZIP mellékletben adhatók át, nem pedig a Git repository részeként. Ez nem érinti a reprodukálhatóságot, mert a szükséges parancsok, konfigurációk és leírások továbbra is a repository-ban maradnak, így az eredmények azonos bemeneti adatok mellett újrafuttathatók.
