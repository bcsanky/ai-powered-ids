# Dolgozatba beemelhető eredményjegyzék

Ez a dokumentum rögzíti, hogy a validált mérési eredmények milyen futtatási könyvtárakban, CSV állományokban és PNG ábrákban állnak elő. A felsorolt eredmények dokumentált konfigurációval, azonos futtatási beállítások mellett reprodukálhatók.

Repository clean state megjegyzés: a `reports/final`, `reports/lab`, `reports/performance`, `results/performance` és `figures/final` alatti korábbi futási kimenetek nem verziózott forrásként hivatkozandók. A dolgozatba kerülő ábrákat és táblázatokat a tényleges mérés után kell előállítani, majd provenance és manifest alapján kell beadási mellékletként kezelni. A dokumentumban szereplő ilyen útvonalak várható kimeneti helyek vagy korábbi validációs hivatkozások, nem Gitben tartandó kutatási bizonyítékok.

Fontos adateredeti megkötés: a `reports/lab`, `reports/final` és `examples/lab` tartalma demonstrációs vagy offline jellegű, ezért nem használható real-lab mérési eredményként. Real-lab eredmény csak `reports/real_measurement/measurement_provenance.json` megléte és `verified_real_lab` provenance státusz esetén emelhető be a dolgozat mérési eredményei közé.

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
| Teljesítménymérési eredménytábla | `reports/performance/benchmark_results.csv` | Batch scoring throughput és latency táblázat | teljesítménymérési eredmény, lokális/labor mérés |
| Kompatibilis teljesítménymérési eredménytábla | `results/performance/performance_metrics.csv` | CPU-idővel és memóriaoszlopokkal kiegészített teljesítménymérési táblázat | dolgozatba beemelhető, lokális/labor mérés |
| Teljesítményriport Markdown formában | `reports/performance/performance_report.md` | Teljesítménymérési alfejezet forrása | dolgozatba beemelhető |
| Teljesítményriport HTML formában | `reports/performance/performance_report.html` | Dashboard jellegű teljesítmény-összefoglaló | lokális/labor mérés |
| Május 10-i performance validáció | `docs/may10_performance_validation.md` | CPU/RAM kiegészítés és kompatibilis kimenetek validációs jegyzőkönyve | dolgozatba beemelhető háttérdokumentáció |

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
| Teljesítménymérési késleltetés ábra | `reports/performance/latency_by_batch_size.png` | Batch size és p95 késleltetés bemutatása | dolgozatba beemelhető, lokális/labor mérés |
| Teljesítménymérési throughput ábra | `reports/performance/throughput_by_batch_size.png` | Áteresztőképesség bemutatása esemény/másodpercben | dolgozatba beemelhető, lokális/labor mérés |
| Scoring időeloszlás ábra | `reports/performance/scoring_time_distribution.png` | Átlagos scoring késleltetés összehasonlítása | dolgozatba beemelhető, lokális/labor mérés |
| CPU- és memóriahasználati ábra | `reports/performance/resource_usage_by_batch_size.png` | CPU-idő és memóriahasználat batch size szerinti bemutatása, ha az adatok elérhetők | dolgozatba beemelhető, lokális/labor mérés |
| Kompatibilis késleltetési ábra | `figures/final/latency_by_load.png` | Végleges néven hivatkozható p95 késleltetési ábra | dolgozatba beemelhető |
| Kompatibilis throughput ábra | `figures/final/throughput.png` | Végleges néven hivatkozható áteresztőképességi ábra | dolgozatba beemelhető |
| Dolgozatba rendezett performance ábrák | `reports/final/thesis_figures/performance_*.png` | Teljesítménymérési ábrák rendezett másolatai | dolgozatba beemelhető |

Megjegyzés: a teljesítménymérési eredmények batch scoring/inference mérések. Az eredeti event/perc terhelési célok helyett a prototípus jelen változata 100, 500, 1000, 5000 és 10000 esemény feldolgozási idejét vizsgálja több batch size mellett. Ez nem teljes SIEM/Wazuh end-to-end terhelhetőségi mérés.

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

## Valódi lab-alapú Wazuh+AE eredmények

Ez a blokk a natív Wazuh alert exporttal, lab ground truth eseményekkel és AE-Minimal lab feature-ökkel futtatható összehasonlító mérési lánc kimeneteit rögzíti. Ezek az eredmények csak akkor nevezhetők validált lab-alapú Wazuh+AE eredménynek, ha a bemeneti `data/wazuh/alerts.jsonl`, `data/lab/lab_ground_truth.csv` és `data/lab/lab_features.csv` állományok rendelkezésre állnak, és a `make final-real-hybrid` parancs sikeresen lefutott.

| Elem | Forrásfájl | Dolgozatbeli felhasználás | Státusz |
|---|---|---|---|
| Natív Wazuh-only lab metrikák | `results/wazuh_real/metrics_summary.csv` | Natív Wazuh szabályalapú baseline értékelése címkézett lab eseményeken | bemenetfüggő, valódi Wazuh export szükséges |
| AE-Minimal lab metrikák | `results/ae_lab/metrics_summary.csv` | AE-Minimal offline lab scoring értékelése ugyanazon eseményeken | bemenetfüggő |
| Valós hibrid metrikák | `results/hybrid_real/metrics_summary.csv` | Hybrid OR, weighted és priority stratégiák összehasonlítása | bemenetfüggő |
| Valós lab összehasonlító táblázat | `results/real_comparison/metrics_comparison.csv` | Wazuh-only, AE-Minimal lab és hibrid stratégiák egységes táblázata | bemenetfüggő |
| Valós lab összehasonlító Markdown | `results/real_comparison/metrics_comparison.md` | Dolgozatba átemelhető összehasonlító táblázat | bemenetfüggő |
| Valós lab precision/recall/F1 ábra | `results/real_comparison/fig_precision_recall_f1.png` | Wazuh-only, AE-only és hibrid konfigurációk fő metrikáinak összehasonlítása | bemenetfüggő |
| Valós lab hamis pozitív arány ábra | `results/real_comparison/fig_false_positive_rate.png` | Téves riasztási arány összehasonlítása | bemenetfüggő |
| Valós lab riasztásszám ábra | `results/real_comparison/fig_alert_count.png` | Riasztási mennyiség összehasonlítása | bemenetfüggő |
| Valós lab TTD ábra | `results/real_comparison/fig_mean_ttd.png` | Átlagos Wazuh-alapú detektálási idő összehasonlítása, ha rendelkezésre áll | opcionális, Wazuh TTD szükséges |

## Valódi lab input-előállítás és validáció

Ez a blokk azokat az állományokat és dokumentumokat rögzíti, amelyek a `final-real-hybrid` mérési lánc valós lab bemeneteinek előállításához, ellenőrzéséhez és reprodukálásához szükségesek. A `templates/lab` alatti sablonok verziókezelésben tarthatók. A `data/lab` és `data/wazuh` alatti tényleges mérési állományok méretük és környezetfüggő jellegük miatt jellemzően nem kerülnek Git verziókezelésbe, hanem a beadási mellékletben vagy helyi mérési könyvtárban őrizhetők meg.

| Elem | Forrásfájl | Dolgozatbeli felhasználás | Státusz |
|---|---|---|---|
| Ground truth CSV sablon | `templates/lab/lab_ground_truth_template.csv` | Lab eseményablakok kézi ellenőrzéséhez és kitöltéséhez | verziókezelésben megtartható sablon |
| Lab feature CSV sablon | `templates/lab/lab_features_template.csv` | AE-Minimal kompatibilis lab feature struktúra bemutatása | verziókezelésben megtartható sablon |
| Lab szcenárió sablon | `templates/lab/lab_scenarios_template.yaml` | Mérési szcenáriók előzetes tervezése | verziókezelésben megtartható sablon |
| Event marker state fájl | `data/lab/session_events.json` | Folyamatban lévő lab eseményrögzítés állapota | lokális mérési állomány, Gitből kizárva |
| Valós lab ground truth | `data/lab/lab_ground_truth.csv` | Címkézett lab eseményablakok a Wazuh+AE méréshez | lokális mérési állomány, beadási mellékletbe tehető |
| Valós lab feature fájl | `data/lab/lab_features.csv` | AE-Minimal lab scoring bemenete | lokális mérési állomány, beadási mellékletbe tehető |
| Lab input validációs jelentés | `reports/lab_input_validation/input_validation_report.md` | Bemeneti konzisztencia ellenőrzésének dokumentálása | csak tényleges mérés után beemelendő |
| Lab támadásszimulációs runbook | `docs/lab_attack_scenarios_runbook.md` | Valós lab mérés előkészítése és reprodukálása | dolgozatba beemelhető háttérdokumentáció |
| Valós mérési checklist | `docs/final_real_measurement_checklist.md` | Mérési előfeltételek és kimenetek ellenőrzése | dolgozatba beemelhető háttérdokumentáció |

## Valódi lab mérési csomag és dolgozati riport

Ez a blokk a tényleges real-lab futtatás után előálló ellenőrző, riport- és csomagolási állományokat rögzíti. A `data/wazuh/*metadata.json` állományok jelszót nem tartalmazhatnak, de környezeti URL-t, indexmintát vagy mérési időablakot tartalmazhatnak, ezért beadás előtt tartalmi ellenőrzésük szükséges.

| Elem | Forrásfájl | Dolgozatbeli felhasználás | Státusz |
|---|---|---|---|
| Wazuh export összefoglaló | `reports/wazuh_export/wazuh_export_summary.md` | A natív Wazuh alert export ellenőrző leírása | tényleges Wazuh export után áll elő |
| Wazuh rule summary | `reports/wazuh_export/wazuh_export_rule_summary.csv` | Rule azonosítók és előfordulások táblázatos összesítése | tényleges Wazuh export után áll elő |
| Bundle validációs riport | `reports/real_measurement/bundle_validation_report.md` | Annak ellenőrzése, hogy a real-lab mérési csomag teljes-e | tényleges mérés után beemelendő |
| Real-lab eredményriport | `reports/real_measurement/real_lab_results_report.md` | Mérnöki értékelés a Wazuh-only, AE-only és hibrid eredményekről | dolgozatba beemelhető, ha a mérés lefutott |
| Dolgozati real-lab szakasz | `reports/real_measurement/thesis_real_lab_section.md` | Közvetlenül a 6. fejezetbe illeszthető real-lab eredményszöveg | dolgozatba beemelhető, ha a mérés lefutott |
| Mérési manifest CSV | `reports/real_measurement/measurement_manifest.csv` | Beadási melléklet fájljegyzéke SHA256 hash-ekkel | archiválási és beadási ellenőrzés |
| Mérési manifest Markdown | `reports/real_measurement/measurement_manifest.md` | Ember által olvasható mérési csomagjegyzék | archiválási és beadási ellenőrzés |
| Anonimizált real-lab riportok | `reports/real_measurement_redacted/` | IP-címek és hostnevek eltávolítása után mellékelhető riportok | opcionális, érzékeny adatok esetén szükséges |

## Real-lab QA és dolgozati beemelési segédletek

Ez a blokk a tényleges mérés előtti és utáni minőségbiztosítási kimeneteket sorolja fel. Ezek a fájlok nem helyettesítik a mérési eredményeket, hanem azt dokumentálják, hogy az eredmények alkalmasak-e a 6. fejezetbe történő beemelésre.

| Elem | Forrásfájl | Dolgozatbeli felhasználás | Státusz |
|---|---|---|---|
| Real-lab preflight riport | `reports/real_measurement_qa/preflight_report.md` | Mérés előtti feltétel-ellenőrzés dokumentálása | mérés előtt futtatandó |
| Post-run QA riport | `reports/real_measurement_qa/postrun_quality_report.md` | Eredmények dolgozati beemelhetőségének ellenőrzése | tényleges mérés után futtatandó |
| Kutatási kérdés összefoglaló JSON | `reports/real_measurement_qa/research_question_answer.json` | Wazuh-onlyhoz viszonyított hibrid változások géppel olvasható összegzése | tényleges metrikákból áll elő |
| Dolgozati beemelhetőség | `reports/real_measurement_qa/thesis_readiness.md` | READY / READY_WITH_LIMITATIONS / NOT_READY státusz magyarázata | 6. fejezet előtti ellenőrzés |
| Real-lab comparison táblázat | `reports/real_measurement_qa/thesis_table_real_comparison.md` | Wordbe másolható fő eredménytábla | dolgozatba beemelhető, ha QA szerint alkalmas |
| Dolgozati értelmezési pontok | `reports/real_measurement_qa/thesis_interpretation_bullets.md` | Rövid megállapítások a 6. fejezethez | tényleges metrikákból áll elő |
| Védési real-lab jegyzet | `reports/real_measurement_qa/defense_notes_real_measurement.md` | Védésre használható kérdés-válasz segédlet | mérési eredmények alapján frissítendő |

## End-to-end live integration demonstráció

Ez a blokk a Wazuh alert exportból induló, AE-Minimal scoringgal és hibrid prioritási logikával gazdagított integrációs kimeneteket sorolja fel. Ezek csak verified real-lab provenance mellett használhatók integrációs demonstrációként, és nem helyettesítik a Wazuh-only vs AE-only vs Hybrid benchmark táblát.

| Elem | Forrásfájl | Dolgozatbeli felhasználás | Státusz |
|---|---|---|---|
| Gazdagított alert JSONL | `reports/live_integration/enriched_alerts.jsonl` | Dashboard vagy OpenSearch bemenet eseményszintű JSONL formában | verified real-lab provenance mellett használható |
| Gazdagított alert CSV | `reports/live_integration/enriched_alerts.csv` | Eseményszintű áttekintés Wazuh, ML és hibrid mezőkkel | verified real-lab provenance mellett használható |
| Nem illesztett alert lista | `reports/live_integration/unmatched_alerts.csv` | Feature mapping lefedettségének ellenőrzése | értelmezési korlátként közlendő |
| Enrichment összefoglaló | `reports/live_integration/enrichment_summary.csv` | Scored, unmatched és hibrid pozitív darabszámok összesítése | dashboard-ready kimenet |
| Dashboard payload | `reports/live_integration/dashboard_payload.json` | Dashboard kártyák és top listák géppel olvasható formában | integrációs demonstráció |
| Dashboard összefoglaló | `reports/live_integration/dashboard_summary.md` | Dolgozatban röviden hivatkozható dashboard-jellegű áttekintés | integrációs demonstráció |
| Live integration validáció | `reports/live_integration/live_integration_validation.md` | READY / READY_WITH_LIMITATIONS / NOT_READY státusz indoklása | dolgozati beemelés előtti ellenőrzés |
| Live integration dolgozati szakasz | `reports/live_integration/thesis_live_integration_section.md` | 5. fejezetbe illeszthető integrációs leírás | verified real-lab provenance mellett használható |
| Live integration védési jegyzet | `reports/live_integration/live_integration_defense_notes.md` | Védési kérdés-válasz segédlet az integrációs lánchoz | mérési csomag után frissítendő |

## Lab session orchestration és operátori naplózás

Ez a blokk a tényleges mérés végrehajtását támogató operátori segédleteket sorolja fel. Ezek nem mérési eredmények, hanem a real-lab futás reprodukálhatóságát és auditálhatóságát támogató futási kimenetek. Gitbe nem kerülnek automatikusan, futási outputként kezelendők.

| Elem | Forrásfájl | Dolgozatbeli felhasználás | Státusz |
|---|---|---|---|
| Session doctor riport | `reports/lab_session/session_doctor_report.md` | Mérés előtti környezet- és modul-ellenőrzés dokumentálása | operátori segédlet |
| Session terv | `reports/lab_session/session_plan.md` | Tervezett szcenáriók és szükséges kimenetek rögzítése | operátori segédlet |
| Operátori parancsnapló sablon | `reports/lab_session/operator_command_log_template.md` | Mérés közbeni kézi parancsnapló vezetése | operátori segédlet |
| Post-session input check | `reports/lab_session/post_session_input_check.md` | A tényleges inputok meglétének ellenőrzése a pipeline előtt | operátori ellenőrzés |
| Futtatási parancslista | `reports/lab_session/run_commands.md` | A mérési pipeline kézi futtatási sorrendje | operátori segédlet |
| Session összefoglaló | `reports/lab_session/session_summary.md` | A session után rendelkezésre álló inputok és outputok áttekintése | operátori összegzés |

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
