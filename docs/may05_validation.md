# Május 5-i AE-Context és összehasonlító validáció

Validáció dátuma: 2026-05-05  
Branch: `thesis/final`

## Napi cél

A napi cél az AE-Context teljes adatos futtatásának lezárása, az AE-Context ábrák előállítása, az AE-Minimal ábrák újraellenőrzése, valamint a végleges összehasonlító eredmények frissítése volt. A validáció célja annak rögzítése, hogy mely mérési eredmények használhatók a diplomamunka 6. fejezetében.

## Futtatott parancsok

```bash
git branch --show-current
git status --short
make final-validate
make dataset CONFIG=experiments/final/ae_context.yaml
make train-ae CONFIG=experiments/final/ae_context.yaml
make final-plot-ae-context
make final-plot-ae-minimal
make final-compare
```

A `final-day5` Makefile cél dokumentált hosszú futtatási célként rendelkezésre áll, de a validáció során a lépések külön parancsokkal futottak le, hogy az egyes ellenőrzési pontok külön rögzíthetők legyenek.

## AE-Context dataset build eredménye

Az AE-Context adatépítés sikeresen lefutott:

```bash
make dataset CONFIG=experiments/final/ae_context.yaml
```

Ellenőrzött metaadatfájl:

```text
data/processed/final/ae_context/dataset_metadata.json
```

Fő ellenőrzött értékek:

- `experiment_id`: `final-ae-context-v1`
- `context_enabled`: `true`
- `context_fit_split`: `train`
- `unknown_context_frequency`: `0.0`
- `dev_sample.enabled`: `false`
- `rows_train`: `1589922`
- `rows_val`: `340698`
- `rows_calib`: `448628`
- `rows_test`: `448628`
- `rows_calib_attacks`: `278278`
- `rows_calib_benign`: `170350`
- `rows_test_attacks`: `278278`
- `rows_test_benign`: `170350`

Ellenőrzött kontextusjellemzők:

- `destination_port_frequency`
- `protocol_frequency`
- `is_rare_destination_port`
- `packet_ratio`
- `bytes_packets_ratio`

A gyakorisági kontextusjellemzők kizárólag a tanító adatrészen illesztett statisztikákból származnak.

## AE-Context training eredménye

Az AE-Context teljes adatos tanítás sikeresen lefutott:

```bash
make train-ae CONFIG=experiments/final/ae_context.yaml
```

AE-Context modellkimeneti könyvtár:

```text
final-ae-context-v1/ae_v1_20260504_131741/
```

AE-Context eredménykönyvtár:

```text
results/final/final-ae-context-v1/ae_v1_20260504_131741/
```

Fő modellfájlok:

- `model.joblib`
- `history.json`
- `thresholds.json`
- `train_config.json`

Fő eredményfájlok:

- `metrics_summary.csv`
- `predictions.csv`
- `threshold_curve.csv`
- `top_feature_errors.csv`
- `run_metadata.json`

A `metrics_summary.csv` tartalmazza a szakdolgozati összehasonlításhoz szükséges fő metrikákat, beleértve a `false_positive_rate`, `false_negative_rate`, `true_positive_rate`, `true_negative_rate` és `alert_count` mezőket. A `predictions.csv` tartalmazza a pontszámot, a valós címkét, a küszöbönkénti predikciókat és az első öt magyarázó jellemzőmezőt.

## AE-Context ábrák

Az AE-Context ábragenerálás sikeresen lefutott:

```bash
make final-plot-ae-context
```

Létrejött ábrák:

- `confusion_matrix.png`
- `score_distribution.png`
- `threshold_curve.png`
- `roc_curve.png`
- `top_feature_frequency.png`

## AE-Minimal ábrák újraellenőrzése

Az AE-Minimal ábragenerálás ismét sikeresen lefutott:

```bash
make final-plot-ae-minimal
```

Ellenőrzött AE-Minimal eredménykönyvtár:

```text
results/final/final-ae-minimal-v1/ae_v1_20260504_113852/
```

Ellenőrzött ábrák:

- `confusion_matrix.png`
- `score_distribution.png`
- `threshold_curve.png`
- `roc_curve.png`
- `top_feature_frequency.png`

## Final comparison eredménye

Az összehasonlító eredmények frissítése sikeresen lefutott:

```bash
make final-compare
```

Frissített comparison könyvtár:

```text
results/final/comparison/
```

Fő kimenetek:

- `metrics_comparison.csv`
- `metrics_comparison.md`
- `fig_comparison_precision_recall_f1.png`
- `fig_comparison_false_positive_rate.png`
- `fig_comparison_alert_count.png`

## Metrics comparison státuszösszefoglaló

| Konfiguráció | Státusz | Megjegyzés |
|---|---|---|
| `ae_minimal` | ok | Validált AE-Minimal futtatás alapján. |
| `ae_context` | ok | Validált AE-Context futtatás alapján. |
| `baseline_stat` | ok | Validált statisztikai baseline futtatás alapján. |
| `baseline_wazuh` | missing | Nem áll rendelkezésre megfelelő címkézett Wazuh export. |
| `hybrid` | missing | Nincs végleges hibrid kiértékelő lépés. |

Az `ae_minimal`, `ae_context` és `baseline_stat` sorok tartalmazzák a fő összehasonlító metrikákat: precision, recall, F1, false positive rate, false negative rate, alert count, ROC-AUC, mintaszám, támadó minták száma és benign minták száma.

## Nyitva maradt feladatok május 6-ra

- Wazuh baseline export vagy szintetikus Wazuh-szerű export előkészítése.
- Hibrid kiértékelési döntés: csak akkor érdemes véglegesíteni, ha stabil összekapcsolási kulcs áll rendelkezésre az AE és Wazuh eredmények között.
- A diplomamunka 5-6. fejezetének írása a validált AE-Minimal, AE-Context és statisztikai baseline eredmények alapján.
- A dolgozatba beemelt ábrák és táblázatok forrásfájljainak végleges rögzítése.
