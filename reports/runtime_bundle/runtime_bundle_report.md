# Runtime reproduction bundle riport

Státusz: **WARN**
Csomag neve: `dist/submission/ai_powered_ids_runtime_bundle.zip`
Létrehozás ideje: `2026-05-15T00:55:17Z`
Összes becsomagolt fájl: **403**
Becsült összméret: **2.3 MiB**

## Kötelező elemek státusza

| path | category | status | note |
| --- | --- | --- | --- |
| README.md | project | OK |  |
| Makefile | project | OK |  |
| requirements-dev.txt | project | OK |  |
| ml/src | source | OK |  |
| ml/tests | tests | OK |  |
| experiments/final/ae_minimal.yaml | config | OK |  |
| experiments/final/ae_context.yaml | config | OK |  |
| docs | documentation | OK |  |
| templates/lab | lab_templates | OK |  |
| artifacts/final/final-ae-minimal-v1 | model_artifact | OK |  |
| data/processed/final/ae_minimal/preprocess.pkl | model_artifact | OK |  |
| reports/real_measurement | real_measurement_report | OK |  |
| reports/real_measurement_qa | qa_report | OK |  |
| reports/measurement_quality | quality_report | OK |  |
| results/wazuh_real | real_result | OK |  |
| results/ae_lab | real_result | OK |  |
| results/hybrid_real | real_result | OK |  |
| results/real_comparison | real_result | OK |  |
| data/lab/lab_ground_truth.csv | runtime_input | OK |  |
| data/lab/lab_features.csv | runtime_input | OK |  |
| data/lab/zeek/conn.log | runtime_input | OK |  |
| data/lab/flows.csv | runtime_input | MISSING | Optional flow-CSV input; Zeek conn.log is the measured input when this is absent. |
| data/wazuh/alerts.jsonl | runtime_input | OK |  |
| raw/wazuh_export.json | runtime_input | MISSING | Optional raw manual Wazuh export; normalized alerts.jsonl is sufficient when this is absent. |

## Hiányzó elemek

- `data/lab/flows.csv`
- `raw/wazuh_export.json`

## Kizárt elemek összefoglalása

- Kizárt fájlok száma: `260`
- `ml/src/__pycache__/__init__.cpython-311.pyc`
- `ml/src/__pycache__/__init__.cpython-38.pyc`
- `ml/src/__pycache__/__init__.cpython-39.pyc`
- `ml/src/__pycache__/autoencoder.cpython-311.pyc`
- `ml/src/__pycache__/autoencoder.cpython-38.pyc`
- `ml/src/__pycache__/autoencoder.cpython-39.pyc`
- `ml/src/__pycache__/benchmark_scoring.cpython-38.pyc`
- `ml/src/__pycache__/build_dataset.cpython-38.pyc`
- `ml/src/__pycache__/collect_thesis_figures.cpython-38.pyc`
- `ml/src/__pycache__/compare_final_results.cpython-38.pyc`
- `ml/src/__pycache__/create_rule_proxy_export.cpython-38.pyc`
- `ml/src/__pycache__/eval.cpython-38.pyc`
- `ml/src/__pycache__/eval.cpython-39.pyc`
- `ml/src/__pycache__/evaluate.cpython-38.pyc`
- `ml/src/__pycache__/explain.cpython-38.pyc`
- `ml/src/__pycache__/explain.cpython-39.pyc`
- `ml/src/__pycache__/export_performance_artifacts.cpython-38.pyc`
- `ml/src/__pycache__/export_performance_outputs.cpython-38.pyc`
- `ml/src/__pycache__/features.cpython-38.pyc`
- `ml/src/__pycache__/generate_case_studies.cpython-38.pyc`
- ... további 240 fájl

## Érzékenyadat-ellenőrzés

Nem találtam jelszó-, token-, privátkulcs- vagy tanúsítványtartalomra utaló mintát.

## Reprodukciós korlátok

- A csomag reprodukciós és újraellenőrzési célú, nem publikus forrás-only beadási csomag.
- A raw Wazuh/Zeek/lab bemenetek a runtime ellenőrzés miatt kerülnek bele.
- A csomag nem telepíti újra automatikusan a teljes Wazuh, Zeek vagy VirtualBox labor infrastruktúrát.
- Hiányzó elem esetén nem készül kézzel pótolt mérési adat; a hiány a manifestben és ebben a riportban szerepel.

## Javasolt futtatási parancsok

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m py_compile $(find ml/src -name "*.py")
python -m pytest ml/tests
make wazuh-parse-alerts
make wazuh-correlate
make wazuh-eval-real
make lab-ae-validate-features
make lab-ae-score
make lab-ae-eval
make hybrid-real-eval
make real-compare
make real-plot
make final-real-measurement-thesis-ready
make final-measurement-quality MEASUREMENT_QUALITY_THRESHOLDS=docs/measurement_quality_thresholds_real_lab_100.yaml
```
