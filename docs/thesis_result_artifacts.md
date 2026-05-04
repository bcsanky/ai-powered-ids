# Dolgozatba beemelhető eredményjegyzék

Ez a dokumentum rögzíti, hogy a validált mérési eredmények közül mely futtatási könyvtárak, CSV állományok és PNG ábrák használhatók a diplomamunka 6. fejezetében. A felsorolt eredmények a konfiguráció és a kód alapján reprodukálható futtatásokból származnak.

## Végleges futtatási könyvtárak

| Mérési ág | Futtatási könyvtár | Státusz |
|---|---|---|
| AE-Minimal | `results/final/final-ae-minimal-v1/ae_v1_20260504_113852/` | validált |
| AE-Context | `results/final/final-ae-context-v1/ae_v1_20260504_131741/` | validált |
| Statisztikai baseline | `results/final/final-baseline-stat-v1/baseline_stat_20260504_120120/` | validált |
| Wazuh baseline | nincs végleges futtatási könyvtár | hiányzik |
| Hibrid konfiguráció | nincs végleges futtatási könyvtár | tervezett |
| Összehasonlító eredmények | `results/final/comparison/` | validált |

## Táblázatokhoz használható CSV állományok

| Elem | Forrásfájl | Dolgozatbeli felhasználás | Státusz |
|---|---|---|---|
| AE-Minimal fő metrikák | `results/final/final-ae-minimal-v1/ae_v1_20260504_113852/metrics_summary.csv` | AE-Minimal mérőszámok bemutatása | validált |
| AE-Context fő metrikák | `results/final/final-ae-context-v1/ae_v1_20260504_131741/metrics_summary.csv` | AE-Context mérőszámok bemutatása | validált |
| Statisztikai baseline fő metrikák | `results/final/final-baseline-stat-v1/baseline_stat_20260504_120120/metrics_summary.csv` | Egyszerű statisztikai baseline összehasonlítása | validált |
| Végleges összehasonlító táblázat | `results/final/comparison/metrics_comparison.csv` | AE-Minimal, AE-Context és baseline_stat egységes összehasonlítása | validált |
| Dolgozatba közvetlenül átemelhető Markdown táblázat | `results/final/comparison/metrics_comparison.md` | Összehasonlító táblázat a 6. fejezethez | validált |
| AE-Minimal mintaszintű eredmények | `results/final/final-ae-minimal-v1/ae_v1_20260504_113852/predictions.csv` | Riasztásszám és hibás besorolások elemzése | validált |
| AE-Context mintaszintű eredmények | `results/final/final-ae-context-v1/ae_v1_20260504_131741/predictions.csv` | Riasztásszám és hibás besorolások elemzése | validált |
| Wazuh baseline metrikák | `results/final/final-baseline-wazuh-v1/<run>/metrics_summary.csv` | Wazuh vagy Wazuh-szerű kiértékelés | hiányzik |
| Hibrid metrikák | `results/final/final-hybrid-v1/<run>/metrics_summary.csv` | AE és Wazuh eredmények kombinált kiértékelése | tervezett |

## Ábrákhoz használható PNG állományok

| Elem | Forrásfájl | Dolgozatbeli felhasználás | Státusz |
|---|---|---|---|
| AE-Minimal konfúziós mátrix | `results/final/final-ae-minimal-v1/ae_v1_20260504_113852/confusion_matrix.png` | Predikciós hibák szemléltetése | validált |
| AE-Minimal pontszámeloszlás | `results/final/final-ae-minimal-v1/ae_v1_20260504_113852/score_distribution.png` | Benign és támadó minták anomáliapontszámának összehasonlítása | validált |
| AE-Minimal küszöbgörbe | `results/final/final-ae-minimal-v1/ae_v1_20260504_113852/threshold_curve.png` | Küszöbválasztás hatásának bemutatása | validált |
| AE-Minimal ROC-görbe | `results/final/final-ae-minimal-v1/ae_v1_20260504_113852/roc_curve.png` | Detektálási képesség vizuális értékelése | validált |
| AE-Minimal magyarázó jellemzők | `results/final/final-ae-minimal-v1/ae_v1_20260504_113852/top_feature_frequency.png` | Rekonstrukciós hiba magyarázhatósági kiegészítése | validált |
| AE-Context konfúziós mátrix | `results/final/final-ae-context-v1/ae_v1_20260504_131741/confusion_matrix.png` | Predikciós hibák szemléltetése | validált |
| AE-Context pontszámeloszlás | `results/final/final-ae-context-v1/ae_v1_20260504_131741/score_distribution.png` | Benign és támadó minták anomáliapontszámának összehasonlítása | validált |
| AE-Context küszöbgörbe | `results/final/final-ae-context-v1/ae_v1_20260504_131741/threshold_curve.png` | Küszöbválasztás hatásának bemutatása | validált |
| AE-Context ROC-görbe | `results/final/final-ae-context-v1/ae_v1_20260504_131741/roc_curve.png` | Detektálási képesség vizuális értékelése | validált |
| AE-Context magyarázó jellemzők | `results/final/final-ae-context-v1/ae_v1_20260504_131741/top_feature_frequency.png` | Rekonstrukciós hiba magyarázhatósági kiegészítése | validált |
| Precision, recall és F1 összehasonlítása | `results/final/comparison/fig_comparison_precision_recall_f1.png` | Konfigurációk fő metrikáinak összehasonlítása | validált |
| Hamis pozitív arány összehasonlítása | `results/final/comparison/fig_comparison_false_positive_rate.png` | Üzemeltetési terheléshez kapcsolódó hibaarány bemutatása | validált |
| Riasztásszám összehasonlítása | `results/final/comparison/fig_comparison_alert_count.png` | Riasztási mennyiség összehasonlítása | validált |

## Validált és hiányzó konfigurációk

Validált konfigurációk:

- `ae_minimal`
- `ae_context`
- `baseline_stat`

Hiányzó vagy tervezett konfigurációk:

- `baseline_wazuh`: még nem validált, mert nem áll rendelkezésre megfelelő címkézett Wazuh vagy Wazuh-szerű export, amelyből a `metrics_summary.csv` előállítható lenne.
- `hybrid`: még nem validált, mert nincs végleges hibrid kiértékelő lépés és nincs stabil összekapcsolási kulcs az AE predikciók és a Wazuh riasztások között.

## Beemelési megjegyzés

A diplomamunka 6. fejezetében az AE-Minimal, AE-Context és statisztikai baseline eredmények szerepeltethetők validált mérési eredményként. A Wazuh baseline és a hibrid konfiguráció csak akkor kerülhet be mérési eredményként, ha később létrejön hozzájuk validált `metrics_summary.csv`, `predictions.csv` és futtatási metaadat.
