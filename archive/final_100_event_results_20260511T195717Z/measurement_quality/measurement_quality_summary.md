# Mérési minőségi összefoglaló

Mérési minőségi státusz: **MEASUREMENT_USABLE_WITH_LIMITATIONS**

Kutatási állítás kategória: **TRADEOFF_ONLY**

Ez az összefoglaló nem új mérési eredmény, hanem a meglévő verified real-lab kimenetek minőségi ellenőrzése.

| Forrás | Ellenőrzés | Státusz | Üzenet |
|---|---|---|---|
| scenario_coverage.csv | measurement_provenance | PASS | verified real_lab provenance rendelkezésre áll |
| scenario_coverage.csv | ground_truth_path_guard | PASS | ground truth útvonal elfogadható |
| scenario_coverage.csv | ground_truth_provenance_path | PASS | ground truth útvonal egyezik a provenance-szel |
| scenario_coverage.csv | ground_truth_columns | PASS | kötelező oszlopok rendben |
| scenario_coverage.csv | event_id_unique | PASS | event_id értékek egyediek |
| scenario_coverage.csv | timestamps_valid | PASS | timestamp_start és timestamp_end értékek validak |
| scenario_coverage.csv | min_total_events | PASS | összes esemény: 100, elvárt minimum: 100 |
| scenario_coverage.csv | min_benign_events | PASS | benign események száma: 60, elvárt minimum: 60 |
| scenario_coverage.csv | min_attack_events | PASS | attack események száma: 40, elvárt minimum: 40 |
| scenario_coverage.csv | min_scenarios | PASS | különböző scenario-k száma: 7, elvárt minimum: 7 |
| scenario_coverage.csv | min_attack_scenarios | PASS | attack scenario-k száma: 5, elvárt minimum: 5 |
| scenario_coverage.csv | short_or_zero_duration | PASS | nincs nulla vagy negatív duration |
| scenario_coverage.csv | expected_benign_scenarios | PASS | várt benign scenario-k lefedve |
| scenario_coverage.csv | expected_attack_scenarios | PASS | várt attack scenario-k lefedve |
| feature_alert_alignment.csv | measurement_provenance | PASS | verified real_lab provenance rendelkezésre áll |
| feature_alert_alignment.csv | ground_truth_path_guard | PASS | ground_truth útvonal elfogadható |
| feature_alert_alignment.csv | lab_features_path_guard | PASS | lab_features útvonal elfogadható |
| feature_alert_alignment.csv | provenance_ground_truth_path | PASS | útvonal egyezik a provenance-szel |
| feature_alert_alignment.csv | provenance_lab_features_path | PASS | útvonal egyezik a provenance-szel |
| feature_alert_alignment.csv | ground_truth_readable | PASS | ground_truth olvasható |
| feature_alert_alignment.csv | lab_features_readable | PASS | lab_features olvasható |
| feature_alert_alignment.csv | wazuh_predictions_readable | PASS | wazuh_predictions olvasható |
| feature_alert_alignment.csv | ae_predictions_readable | PASS | ae_predictions olvasható |
| feature_alert_alignment.csv | hybrid_predictions_readable | PASS | hybrid_predictions olvasható |
| feature_alert_alignment.csv | features_event_set | PASS | lab_features event_id készlet egyezik |
| feature_alert_alignment.csv | wazuh_event_set | PASS | Wazuh predictions event_id készlet egyezik |
| feature_alert_alignment.csv | ae_event_set | PASS | AE predictions event_id készlet egyezik |
| feature_alert_alignment.csv | hybrid_event_set | PASS | Hybrid predictions event_id készlet egyezik |
| feature_alert_alignment.csv | missing_ae_score_ratio | PASS | hiányzó AE score arány: 0.0000 |
| feature_alert_alignment.csv | attack_without_wazuh_ratio | WARN | Wazuh alert nélküli attack események aránya: 0.6750 |
| feature_alert_alignment.csv | ground_truth_source_ip_filled | PASS | ground_truth.source_ip kitöltöttsége: 1.0000 |
| feature_alert_alignment.csv | ground_truth_target_ip_filled | PASS | ground_truth.target_ip kitöltöttsége: 1.0000 |
| feature_alert_alignment.csv | lab_features_source_ip_filled | PASS | lab_features.source_ip kitöltöttsége: 1.0000 |
| feature_alert_alignment.csv | lab_features_target_ip_filled | PASS | lab_features.target_ip kitöltöttsége: 1.0000 |
| metric_consistency.csv | wazuh_metrics_readable | PASS | wazuh_metrics olvasható |
| metric_consistency.csv | ae_metrics_readable | PASS | ae_metrics olvasható |
| metric_consistency.csv | hybrid_metrics_readable | PASS | hybrid_metrics olvasható |
| metric_consistency.csv | metrics_comparison_readable | PASS | metrics_comparison olvasható |
| metric_consistency.csv | comparison_configurations | PASS | minden kötelező konfiguráció szerepel |
| metric_consistency.csv | range_precision | PASS | precision 0 és 1 közötti |
| metric_consistency.csv | range_recall | PASS | recall 0 és 1 közötti |
| metric_consistency.csv | range_f1 | PASS | f1 0 és 1 közötti |
| metric_consistency.csv | range_false_positive_rate | PASS | false_positive_rate 0 és 1 közötti |
| metric_consistency.csv | range_false_negative_rate | PASS | false_negative_rate 0 és 1 közötti |
| metric_consistency.csv | alert_count_non_negative | PASS | alert_count nem negatív |
| metric_consistency.csv | n_samples_consistency | PASS | n_samples értékek: [100, 100, 100, 100, 100] |
| metric_consistency.csv | nan_metric_values | PASS | nincs NaN a fő metrikákban |
| metric_consistency.csv | wazuh_confusion_sum_0 | PASS | TP+FP+TN+FN=100, n_samples=100 |
| metric_consistency.csv | ae_confusion_sum_0 | PASS | TP+FP+TN+FN=100, n_samples=100 |
| metric_consistency.csv | hybrid_confusion_sum_0 | PASS | TP+FP+TN+FN=100, n_samples=100 |
| metric_consistency.csv | hybrid_confusion_sum_1 | PASS | TP+FP+TN+FN=100, n_samples=100 |
| metric_consistency.csv | hybrid_confusion_sum_2 | PASS | TP+FP+TN+FN=100, n_samples=100 |
| metric_consistency.csv | hybrid_or_logic | PASS | Hybrid OR predikció megfelel a Wazuh OR AE logikának |
| metric_consistency.csv | wazuh_mean_ttd_available | PASS | wazuh mean_ttd elérhető |
| metric_consistency.csv | wazuh_median_ttd_available | PASS | wazuh median_ttd elérhető |
| metric_consistency.csv | hybrid_mean_ttd_available | PASS | hybrid mean_ttd elérhető |
| metric_consistency.csv | hybrid_median_ttd_available | PASS | hybrid median_ttd elérhető |
| ttd_quality.csv | wazuh_predictions_readable | PASS | wazuh_predictions olvasható |
| ttd_quality.csv | hybrid_predictions_readable | PASS | hybrid_predictions olvasható |
| ttd_quality.csv | ground_truth_readable | PASS | ground_truth olvasható |
| ttd_quality.csv | ttd_numeric | PASS | TTD értékek numerikusak, ahol alert volt |
| ttd_quality.csv | ttd_negative | PASS | nincs negatív TTD |
| ttd_quality.csv | ttd_large_values | PASS | 0 darab 300 másodpercnél nagyobb TTD |
| ttd_quality.csv | ttd_attack_coverage | PASS | attack TTD coverage: 0.3250 |
| ttd_quality.csv | ttd_summary_stats | PASS | TTD összefoglaló statisztikák |
| ttd_quality.csv | hybrid_ttd_available | PASS | hibrid predikciókban van TTD adat |
