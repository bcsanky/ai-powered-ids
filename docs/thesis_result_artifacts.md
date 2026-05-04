# Dolgozatba beemelhető eredményjegyzék

Ez a dokumentum rögzíti, hogy a validált mérési eredmények közül mely futtatási könyvtárak, CSV állományok és PNG ábrák használhatók a diplomamunka 6. fejezetében. A felsorolt eredmények dokumentált konfigurációval, azonos futtatási beállítások mellett reprodukálhatók.

## Végleges futtatási könyvtárak

| Mérési ág | Futtatási könyvtár | Státusz |
|---|---|---|
| AE-Minimal | `results/final/final-ae-minimal-v1/ae_v1_20260504_113852/` | validált |
| AE-Context | `results/final/final-ae-context-v1/ae_v1_20260504_131741/` | validált |
| Statisztikai baseline | `results/final/final-baseline-stat-v1/baseline_stat_20260504_120120/` | validált |
| Szabályalapú proxy baseline | `results/final/final-rule-proxy-v1/baseline_wazuh_20260504_144645/` | validált |
| Hibrid konfiguráció | `results/final/final-hybrid-v1/hybrid_20260504_144659/` | validált, offline proxy-alapú |
| Natív Wazuh baseline | nincs végleges futtatási könyvtár | hiányzik |
| Összehasonlító eredmények | `results/final/comparison/` | validált |

## Táblázatokhoz használható CSV állományok

| Elem | Forrásfájl | Dolgozatbeli felhasználás | Státusz |
|---|---|---|---|
| AE-Minimal fő metrikák | `results/final/final-ae-minimal-v1/ae_v1_20260504_113852/metrics_summary.csv` | AE-Minimal mérőszámok bemutatása | validált |
| AE-Context fő metrikák | `results/final/final-ae-context-v1/ae_v1_20260504_131741/metrics_summary.csv` | AE-Context mérőszámok bemutatása | validált |
| Statisztikai baseline fő metrikák | `results/final/final-baseline-stat-v1/baseline_stat_20260504_120120/metrics_summary.csv` | Egyszerű statisztikai baseline összehasonlítása | validált |
| Szabályalapú proxy baseline fő metrikák | `results/final/final-rule-proxy-v1/baseline_wazuh_20260504_144645/metrics_summary.csv` | Kontrollált flow-alapú szabályproxy összehasonlítása | validált |
| Hibrid fő metrikák | `results/final/final-hybrid-v1/hybrid_20260504_144659/metrics_summary.csv` | AE-Minimal és rule_proxy offline kombinációjának értékelése | validált, offline proxy-alapú |
| Végleges összehasonlító táblázat | `results/final/comparison/metrics_comparison.csv` | AE-Minimal, AE-Context, baseline_stat, rule_proxy és hybrid egységes összehasonlítása | validált |
| Dolgozatba közvetlenül átemelhető Markdown táblázat | `results/final/comparison/metrics_comparison.md` | Összehasonlító táblázat a 6. fejezethez | validált |
| AE-Minimal mintaszintű eredmények | `results/final/final-ae-minimal-v1/ae_v1_20260504_113852/predictions.csv` | Riasztásszám és hibás besorolások elemzése | validált |
| AE-Context mintaszintű eredmények | `results/final/final-ae-context-v1/ae_v1_20260504_131741/predictions.csv` | Riasztásszám és hibás besorolások elemzése | validált |
| Rule_proxy mintaszintű eredmények | `results/final/final-rule-proxy-v1/baseline_wazuh_20260504_144645/predictions.csv` | Riasztásszám és hibás besorolások elemzése | validált |
| Hibrid mintaszintű eredmények | `results/final/final-hybrid-v1/hybrid_20260504_144659/predictions.csv` | AE és szabályproxy kombinált döntéseinek elemzése | validált, offline proxy-alapú |
| Natív Wazuh baseline metrikák | `results/final/final-baseline-wazuh-v1/<run>/metrics_summary.csv` | Natív Wazuh export kiértékelése | hiányzik |
| Pontozott demonstrációs események | `reports/scored_events.jsonl` | Scoring feldolgozási lánc bemutatása az 5. fejezetben | demonstrációs kimenet, nem mérési benchmark |
| Szakértői jelentés Markdown formában | `reports/final/security_report.md` | Riportkészítési prototípus bemutatása az 5. fejezetben | dolgozatba beemelhető demonstrációs kimenet |
| Szakértői jelentés HTML formában | `reports/final/security_report.html` | Dashboard jellegű összefoglaló bemutatása | demonstrációs kimenet |
| Dashboard összefoglaló CSV | `reports/final/dashboard_summary.csv` | Fő metrikák riportoldali összesítése | demonstrációs kimenet, nem mérési benchmark |
| Lab/replay pontozott események CSV | `reports/lab/lab_scored_events.csv` | Eseményalapú demonstráció táblázatos bemutatása | demonstrációs validáció, nem benchmark mérés |
| Lab/replay case study összefoglaló | `reports/lab/case_study_summary.md` | Esettanulmányos összefoglaló az 5. fejezethez | dolgozatba beemelhető |
| Port scan esettanulmány | `reports/lab/case_study_port_scan.md` | Port scan jellegű mintázat bemutatása | dolgozatba beemelhető |
| SSH brute force esettanulmány | `reports/lab/case_study_ssh_bruteforce.md` | SSH brute force jellegű mintázat bemutatása | dolgozatba beemelhető |
| Lab/replay scenario summary | `reports/lab/scenario_summary.csv` | Szcenáriónkénti kockázati összesítés | demonstrációs validáció, nem benchmark mérés |
| Thesis figures manifest | `reports/final/thesis_figures/figure_manifest.csv` | Dolgozatba szánt ábrák forrásjegyzéke | dolgozatba beemelhető |

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
| Rule_proxy konfúziós mátrix | `results/final/final-rule-proxy-v1/baseline_wazuh_20260504_144645/confusion_matrix.png` | Szabályproxy predikciós hibáinak szemléltetése | validált |
| Rule_proxy pontszámeloszlás | `results/final/final-rule-proxy-v1/baseline_wazuh_20260504_144645/score_distribution.png` | Rule_proxy pontszámeloszlás bemutatása | validált |
| Rule_proxy küszöbgörbe | `results/final/final-rule-proxy-v1/baseline_wazuh_20260504_144645/threshold_curve.png` | Szabályproxy küszöbérzékenységének bemutatása | validált |
| Rule_proxy ROC-görbe | `results/final/final-rule-proxy-v1/baseline_wazuh_20260504_144645/roc_curve.png` | Szabályproxy pontszám alapú vizuális értékelése | validált |
| Hibrid konfúziós mátrix | `results/final/final-hybrid-v1/hybrid_20260504_144659/confusion_matrix.png` | Hibrid döntés predikciós hibáinak szemléltetése | validált, offline proxy-alapú |
| Hibrid pontszámeloszlás | `results/final/final-hybrid-v1/hybrid_20260504_144659/score_distribution.png` | Hibrid pontszámeloszlás bemutatása | validált, offline proxy-alapú |
| Hibrid ROC-görbe | `results/final/final-hybrid-v1/hybrid_20260504_144659/roc_curve.png` | Hibrid pontszám alapú vizuális értékelése | validált, offline proxy-alapú |
| Precision, recall és F1 összehasonlítása | `results/final/comparison/fig_comparison_precision_recall_f1.png` | Konfigurációk fő metrikáinak összehasonlítása | validált |
| Hamis pozitív arány összehasonlítása | `results/final/comparison/fig_comparison_false_positive_rate.png` | Üzemeltetési terheléshez kapcsolódó hibaarány bemutatása | validált |
| Riasztásszám összehasonlítása | `results/final/comparison/fig_comparison_alert_count.png` | Riasztási mennyiség összehasonlítása | validált |
| Lab/replay timeline ábra | `reports/lab/lab_timeline.png` | Kontrollált eseménysor időbeli szemléltetése az 5. fejezetben | dolgozatba beemelhető demonstrációs ábra |
| Lab/replay kockázati eloszlás | `reports/lab/risk_level_distribution.png` | Kockázati szintek eloszlásának bemutatása | dolgozatba beemelhető demonstrációs ábra |

## Demonstrációs prototípuskimenetek

A scoring szolgáltatás, a batch scoring és a szakértői jelentés a diplomamunka 5. fejezetében az implementált prototípus működését szemlélteti. Ezek a kimenetek a validált AE-Minimal modellre és az összehasonlító eredménytáblára épülnek, de nem helyettesítik a 6. fejezet mérési benchmark táblázatait.

| Elem | Forrásfájl | Dolgozatbeli felhasználás | Státusz |
|---|---|---|---|
| Batch scoring eredmény | `reports/scored_events.jsonl` | Eseményszintű pontozás bemutatása | demonstrációs kimenet |
| Szakértői riport | `reports/final/security_report.md` | Implementációs fejezetben bemutatható riport | dolgozatba beemelhető |
| HTML riport | `reports/final/security_report.html` | Dashboard jellegű megjelenítés demonstrálása | demonstrációs kimenet |
| Dashboard CSV | `reports/final/dashboard_summary.csv` | Riportoldali metrikaösszesítés | demonstrációs kimenet, nem mérési benchmark |
| Lab/replay pontozott események | `reports/lab/lab_scored_events.jsonl` | Kontrollált eseménysor pontozásának bemutatása | demonstrációs validáció |
| Lab/replay esettanulmányok | `reports/lab/case_study_*.md` | Port scan, SSH brute force és kombinált mintázatok elemzése | dolgozatba beemelhető |
| Dolgozati ábrajegyzék | `reports/final/thesis_figures/figure_manifest.csv` | Ábrák forrásának és fejezeti helyének rögzítése | dolgozatba beemelhető |

## Validált és hiányzó konfigurációk

Validált konfigurációk:

- `ae_minimal`
- `ae_context`
- `baseline_stat`
- `rule_proxy`
- `hybrid` kontrollált offline, proxy-alapú kiértékelésként

Hiányzó vagy tervezett konfigurációk:

- `baseline_wazuh_real`: még nem validált, mert nem áll rendelkezésre megfelelő címkézett natív Wazuh export, amelyből a `metrics_summary.csv` előállítható lenne.

## Beemelési megjegyzés

A diplomamunka 6. fejezetében az AE-Minimal, AE-Context, statisztikai baseline, rule_proxy és offline hibrid eredmények szerepeltethetők validált mérési eredményként. A rule_proxy kontrollált, flow-alapú szabályproxy, nem natív Wazuh teljesítménymérés. A hibrid eredmény az AE-Minimal és a rule_proxy azonos sorrendű teszthalmaz-predikcióinak offline kombinációja, nem éles eseménykorreláció. Natív Wazuh mérés csak akkor kerülhet be eredményként, ha később létrejön hozzá validált `metrics_summary.csv`, `predictions.csv` és futtatási metaadat.
