# Real-lab preflight ellenőrzés

Összesített státusz: **PASS**

| Ellenőrzés | Kategória | Státusz | Üzenet | Útvonal | Javaslat |
|---|---|---|---|---|---|
| project_makefile | Projektstruktúra | PASS | létezik | Makefile |  |
| ae_minimal_config | Projektstruktúra | PASS | létezik | experiments/final/ae_minimal.yaml |  |
| ae_minimal_model_dir | Projektstruktúra | PASS | létezik | artifacts/final/final-ae-minimal-v1 |  |
| ae_minimal_preprocess | Projektstruktúra | PASS | létezik | data/processed/final/ae_minimal/preprocess.pkl |  |
| template_lab_ground_truth_template | Lab template-ek | PASS | létezik | templates/lab/lab_ground_truth_template.csv |  |
| template_lab_features_template | Lab template-ek | PASS | létezik | templates/lab/lab_features_template.csv |  |
| template_lab_scenarios_template | Lab template-ek | PASS | létezik | templates/lab/lab_scenarios_template.yaml |  |
| doc_lab_attack_scenarios_runbook | Dokumentáció | PASS | létezik | docs/lab_attack_scenarios_runbook.md |  |
| doc_final_real_measurement_checklist | Dokumentáció | PASS | létezik | docs/final_real_measurement_checklist.md |  |
| doc_real_hybrid_lab_evaluation_runbook | Dokumentáció | PASS | létezik | docs/real_hybrid_lab_evaluation_runbook.md |  |
| doc_real_measurement_export_and_packaging_runbook | Dokumentáció | PASS | létezik | docs/real_measurement_export_and_packaging_runbook.md |  |
| dir_data_lab | Input könyvtárak | PASS | létezik | data/lab |  |
| dir_data_wazuh | Input könyvtárak | PASS | létezik | data/wazuh |  |
| dir_data_processed | Input könyvtárak | PASS | létezik | data/processed |  |
| dir_reports | Input könyvtárak | PASS | létezik | reports |  |
| optional_lab_ground_truth | Opcionális inputok | WARN | hiányzik | data/lab/lab_ground_truth.csv | A tényleges mérés előtt vagy közben kell előállítani. |
| optional_lab_features | Opcionális inputok | WARN | hiányzik | data/lab/lab_features.csv | A tényleges mérés előtt vagy közben kell előállítani. |
| optional_conn | Opcionális inputok | WARN | hiányzik | data/lab/zeek/conn.log | A tényleges mérés előtt vagy közben kell előállítani. |
| optional_flows | Opcionális inputok | WARN | hiányzik | data/lab/flows.csv | A tényleges mérés előtt vagy közben kell előállítani. |
| optional_alerts | Opcionális inputok | WARN | hiányzik | data/wazuh/alerts.jsonl | A tényleges mérés előtt vagy közben kell előállítani. |
| env_opensearch_url | Környezeti változók | WARN | nincs beállítva, de nem kötelező | OPENSEARCH_URL | Csak OpenSearch export futtatásakor szükséges. |
| env_opensearch_username | Környezeti változók | WARN | nincs beállítva, de nem kötelező | OPENSEARCH_USERNAME | Csak OpenSearch export futtatásakor szükséges. |
| env_opensearch_password | Környezeti változók | WARN | nincs beállítva, de nem kötelező | OPENSEARCH_PASSWORD | Csak OpenSearch export futtatásakor szükséges. |
