# Táblázat- és ábraterv a diplomamunkához

Ez a dokumentum a diplomamunka 5. és 6. fejezetéhez javasolt táblázatokat és ábrákat foglalja össze. A státusz azt jelzi, hogy az adott elemhez rendelkezésre áll-e validált forrásfájl, vagy további kézi szerkesztés szükséges.

## Javasolt táblázatok

| Sorszám | Cím | Forrásfájl | Javasolt fejezet | Státusz | Megjegyzés |
|---|---|---|---|---|---|
| 5.1 | Implementált komponensek és szerepük | `docs/thesis_implementation_notes.md` | 5. fejezet | validált | A táblázat kézzel szerkesztendő a komponenslista alapján. |
| 5.2 | Jellemzőkészletek összehasonlítása | `docs/feature_catalog.md` | 5. fejezet | validált | AE-Minimal és AE-Context jellemzők, típusok és előfeldolgozás. |
| 5.3 | Futtatási konfigurációk | `experiments/final/*.yaml` | 5. fejezet | validált | `ae_minimal`, `ae_context`, `baseline_stat`, `rule_proxy`, `hybrid`; a natív Wazuh ág hiányzóként jelölendő. |
| 6.1 | Dataset split | `data/processed/final/ae_minimal/dataset_metadata.json`, `data/processed/final/ae_context/dataset_metadata.json` | 6. fejezet | validált | Tanító, validációs, kalibrációs és teszt adatrész méretei. |
| 6.2 | Fő metrikák konfigurációnként | `results/final/comparison/metrics_comparison.csv` | 6. fejezet | validált | Precision, recall, F1, FPR, FNR, alert count, ROC-AUC. |
| 6.3 | Lab/replay esettanulmányok összefoglalója | `reports/lab/scenario_summary.csv` | 6. fejezet | validált demonstrációs eredmény | Nem benchmark mérés, hanem kontrollált replay demonstráció. |
| 6.4 | Teljesítménymérés összefoglalója | `reports/performance/benchmark_results.csv`, `reports/performance/performance_report.md` | 6. fejezet | validált lokális/labor mérés | Áteresztőképesség, késleltetés, hibás események. |
| 6.5 | Korlátok és kezelésük | `docs/thesis_limitations_and_error_analysis.md` | 6. fejezet | elkészítve | A táblázat kézzel rövidíthető a dolgozati oldalszámhoz igazítva. |

## Javasolt ábrák

| Sorszám | Cím | Forrásfájl | Javasolt képaláírás | Javasolt fejezet | Státusz |
|---|---|---|---|---|---|
| 5.1 | Prototípus architektúrája | [IDE KERÜL: architektúraábra forrásfájlja] | A laboratóriumi prototípus fő komponensei és adatáramlása. | 5. fejezet | hiányzik |
| 5.2 | Adatfeldolgozási lánc | [IDE KERÜL: adatfeldolgozási ábra forrásfájlja] | CIC-IDS2017 bemenet, előfeldolgozás, tanítás és kiértékelés folyamata. | 5. fejezet | hiányzik |
| 5.3 | Scoring és riportgenerálási folyamat | [IDE KERÜL: scoring folyamatábra forrásfájlja] | Eseményszintű pontozás, kockázati kategória és szakértői riport előállítása. | 5. fejezet | hiányzik |
| 6.1 | AE-Minimal konfúziós mátrix | `reports/final/thesis_figures/ae_minimal_confusion_matrix.png` | Az AE-Minimal konfiguráció teszthalmazon számított konfúziós mátrixa. | 6. fejezet | validált |
| 6.2 | AE-Context konfúziós mátrix | `reports/final/thesis_figures/ae_context_confusion_matrix.png` | Az AE-Context konfiguráció teszthalmazon számított konfúziós mátrixa. | 6. fejezet | validált |
| 6.3 | Precision/Recall/F1 összehasonlítás | `reports/final/thesis_figures/comparison_precision_recall_f1.png` | A validált konfigurációk precision, recall és F1 értékeinek összehasonlítása. | 6. fejezet | validált |
| 6.4 | False positive rate összehasonlítás | `reports/final/thesis_figures/comparison_false_positive_rate.png` | A validált konfigurációk hamis pozitív arányának összehasonlítása. | 6. fejezet | validált |
| 6.5 | Alert count összehasonlítás | `reports/final/thesis_figures/comparison_alert_count.png` | A validált konfigurációk riasztásszámának összehasonlítása. | 6. fejezet | validált |
| 6.6 | Lab/replay timeline | `reports/final/thesis_figures/lab_timeline.png` | A kontrollált lab/replay eseménysor időbeli áttekintése. | 5. vagy 6. fejezet | validált demonstrációs ábra |
| 6.7 | Risk level eloszlás | `reports/final/thesis_figures/lab_risk_level_distribution.png` | A lab/replay események kockázati szint szerinti eloszlása. | 5. vagy 6. fejezet | validált demonstrációs ábra |
| 6.8 | Throughput batch size szerint | `reports/final/thesis_figures/performance_throughput_by_batch_size.png` | Batch scoring áteresztőképesség különböző batch size értékek mellett. | 6. fejezet | validált lokális/labor mérés |
| 6.9 | Latency batch size szerint | `reports/final/thesis_figures/performance_latency_by_batch_size.png` | Batch scoring p95 késleltetés különböző batch size értékek mellett. | 6. fejezet | validált lokális/labor mérés |
| 6.10 | Scoring időeloszlás | `reports/final/thesis_figures/performance_scoring_time_distribution.png` | A scoring időbeli jellemzőinek összefoglalása. | 6. fejezet | opcionális |

## Beillesztési megjegyzések

- A 6. fejezet fő eredménytáblájához a `metrics_comparison.csv` használható, de a natív Wazuh sort hiányzóként kell kezelni.
- A lab/replay ábrák demonstrációs célt szolgálnak, nem detektálási benchmarkot.
- A teljesítménymérési ábrák lokális/labor mérést mutatnak, ezért nem szabad éles üzemi teljesítménygaranciaként megfogalmazni.
- Az architektúra- és folyamatábrák hiányzóként szerepelnek, mert ezekhez célszerű külön, dolgozati stílusú vektoros ábrát készíteni.
