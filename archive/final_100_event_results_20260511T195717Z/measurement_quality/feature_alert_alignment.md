# Feature, alert és predikció alignment ellenőrzés

Összesített státusz: **WARN**

A riport azt vizsgálja, hogy a ground truth, feature, Wazuh, AE és hibrid kimenetek ugyanarra az event_id készletre vonatkoznak-e.

| Ellenőrzés | Kategória | Státusz | Üzenet | Javaslat | Érték |
|---|---|---|---|---|---|
| measurement_provenance | Adateredet | PASS | verified real_lab provenance rendelkezésre áll |  |  |
| ground_truth_path_guard | Adateredet | PASS | ground_truth útvonal elfogadható |  | data/lab/lab_ground_truth.csv |
| lab_features_path_guard | Adateredet | PASS | lab_features útvonal elfogadható |  | data/lab/lab_features.csv |
| provenance_ground_truth_path | Adateredet | PASS | útvonal egyezik a provenance-szel | Ellenőrizd, hogy ugyanazt a mérési inputot használod-e. | data/lab/lab_ground_truth.csv |
| provenance_lab_features_path | Adateredet | PASS | útvonal egyezik a provenance-szel | Ellenőrizd, hogy ugyanazt a mérési inputot használod-e. | data/lab/lab_features.csv |
| ground_truth_readable | Bemenet | PASS | ground_truth olvasható | Futtasd a real-lab pipeline megfelelő lépését. | 100 |
| lab_features_readable | Bemenet | PASS | lab_features olvasható | Futtasd a real-lab pipeline megfelelő lépését. | 100 |
| wazuh_predictions_readable | Bemenet | PASS | wazuh_predictions olvasható | Futtasd a real-lab pipeline megfelelő lépését. | 100 |
| ae_predictions_readable | Bemenet | PASS | ae_predictions olvasható | Futtasd a real-lab pipeline megfelelő lépését. | 100 |
| hybrid_predictions_readable | Bemenet | PASS | hybrid_predictions olvasható | Futtasd a real-lab pipeline megfelelő lépését. | 100 |
| features_event_set | Event alignment | PASS | lab_features event_id készlet egyezik |  | missing=0, extra=0 |
| wazuh_event_set | Event alignment | PASS | Wazuh predictions event_id készlet egyezik |  | missing=0, extra=0 |
| ae_event_set | Event alignment | PASS | AE predictions event_id készlet egyezik |  | missing=0, extra=0 |
| hybrid_event_set | Event alignment | PASS | Hybrid predictions event_id készlet egyezik |  | missing=0, extra=0 |
| missing_ae_score_ratio | AE scoring | PASS | hiányzó AE score arány: 0.0000 | A lab feature mappinget és AE scoring kimenetet ellenőrizni kell. | 0.0000 |
| attack_without_wazuh_ratio | Wazuh matching | WARN | Wazuh alert nélküli attack események aránya: 0.6750 | Magas arány esetén a Wazuh export, időablak vagy IP-korreláció ellenőrzendő. | 0.6750 |
| ground_truth_source_ip_filled | Mezőkitöltöttség | PASS | ground_truth.source_ip kitöltöttsége: 1.0000 | Alacsony kitöltöttség ronthatja az alert-event korrelációt. | 1.0000 |
| ground_truth_target_ip_filled | Mezőkitöltöttség | PASS | ground_truth.target_ip kitöltöttsége: 1.0000 | Alacsony kitöltöttség ronthatja az alert-event korrelációt. | 1.0000 |
| lab_features_source_ip_filled | Mezőkitöltöttség | PASS | lab_features.source_ip kitöltöttsége: 1.0000 | Alacsony kitöltöttség ronthatja az alert-event korrelációt. | 1.0000 |
| lab_features_target_ip_filled | Mezőkitöltöttség | PASS | lab_features.target_ip kitöltöttsége: 1.0000 | Alacsony kitöltöttség ronthatja az alert-event korrelációt. | 1.0000 |
