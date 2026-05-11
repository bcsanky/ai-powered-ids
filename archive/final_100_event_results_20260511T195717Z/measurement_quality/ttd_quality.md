# Time-to-detection minőségi ellenőrzés

Összesített státusz: **PASS**

A riport csak meglévő TTD mezőkből dolgozik. Hiányzó TTD esetén nem számol ki helyettesítő értéket.

| Ellenőrzés | Kategória | Státusz | Üzenet | Javaslat | Érték |
|---|---|---|---|---|---|
| wazuh_predictions_readable | Bemenet | PASS | wazuh_predictions olvasható | Futtasd a real-lab pipeline megfelelő lépését. | results/wazuh_real/predictions.csv |
| hybrid_predictions_readable | Bemenet | PASS | hybrid_predictions olvasható | Futtasd a real-lab pipeline megfelelő lépését. | results/hybrid_real/predictions.csv |
| ground_truth_readable | Bemenet | PASS | ground_truth olvasható | Futtasd a real-lab pipeline megfelelő lépését. | data/lab/lab_ground_truth.csv |
| ttd_numeric | TTD | PASS | TTD értékek numerikusak, ahol alert volt |  |  |
| ttd_negative | TTD | PASS | nincs negatív TTD | Ellenőrizd az időszinkront és a korrelációs időablakot. | 0 |
| ttd_large_values | TTD | PASS | 0 darab 300 másodpercnél nagyobb TTD | Túl nagy TTD esetén az időablak és az alert matching ellenőrizendő. | 0 |
| ttd_attack_coverage | TTD | PASS | attack TTD coverage: 0.3250 | Ha nincs TTD adat, a TTD nem használható eredményértelmezésre. | 0.3250 |
| ttd_summary_stats | TTD | PASS | TTD összefoglaló statisztikák |  | {'mean': '38.3824', 'median': '29.0000', 'p95': '87.3500'} |
| hybrid_ttd_available | TTD | PASS | hibrid predikciókban van TTD adat |  |  |
