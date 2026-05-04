# Real-lab operator command sheet

Ez a mérésnapi parancslap kézzel követendő sorrendet ad. Nem hoz létre mérési inputot, nem generál Wazuh alertet, és nem helyettesíti a provenance fájlt. Nyilvános vagy idegen IP-t tilos célozni; rövid szabály: tilos idegen IP.

Kötelező hivatkozások: provenance, Wazuh, `lab_ground_truth.csv`, `lab_features.csv`, `alerts.jsonl`, `final-real-measurement-package-with-provenance`, `final-measurement-quality`, tilos idegen IP.

## Változók

```bash
export SESSION_ID=<SESSION_ID>
export ATTACKER_IP=<ATTACKER_IP>
export TARGET_IP=<TARGET_IP>
export WAZUH_MANAGER_IP=<WAZUH_MANAGER_IP>
export OPENSEARCH_URL=<OPENSEARCH_URL>
export OPENSEARCH_USERNAME=<OPENSEARCH_USERNAME>
export OPENSEARCH_PASSWORD=<OPENSEARCH_PASSWORD>
export WAZUH_EXPORT_START=<YYYY-MM-DDTHH:MM:SSZ>
export WAZUH_EXPORT_END=<YYYY-MM-DDTHH:MM:SSZ>
```

Ne írj be konkrét jelszót dokumentumba vagy Gitbe. Az `OPENSEARCH_PASSWORD` csak shell sessionben legyen jelen.

## 1. Előkészítés

```bash
make live-smoke
make final-acceptance
make final-submission-check
make repo-hygiene-check
make final-validate
make lab-session-prep LAB_SESSION_ID=$SESSION_ID ATTACKER_IP=$ATTACKER_IP TARGET_IP=$TARGET_IP WAZUH_MANAGER_IP=$WAZUH_MANAGER_IP
```

## 2. Marker parancssablonok

Start:

```bash
python -m ml.src.lab_session.scenario_marker_helper start --scenario <SCENARIO> --event-id <EVENT_ID> --source-ip <SOURCE_IP> --target-ip <TARGET_IP>
```

End:

```bash
python -m ml.src.lab_session.scenario_marker_helper end --event-id <EVENT_ID>
```

Export:

```bash
python -m ml.src.lab_session.scenario_marker_helper export --output data/lab/lab_ground_truth.csv
```

## 3. Wazuh export

OpenSearchből:

```bash
make wazuh-export-opensearch WAZUH_EXPORT_START=$WAZUH_EXPORT_START WAZUH_EXPORT_END=$WAZUH_EXPORT_END OPENSEARCH_URL=$OPENSEARCH_URL OPENSEARCH_USERNAME=$OPENSEARCH_USERNAME OPENSEARCH_PASSWORD=$OPENSEARCH_PASSWORD
```

Manuális exportból:

```bash
make wazuh-export-from-file WAZUH_RAW_EXPORT=<RAW_EXPORT_JSON> WAZUH_EXPORT_START=$WAZUH_EXPORT_START WAZUH_EXPORT_END=$WAZUH_EXPORT_END
```

Az eredmény elvárt útvonala: `data/wazuh/alerts.jsonl`.

## 4. Feature build

Zeek conn.log esetén:

```bash
make lab-build-features-zeek
```

Flow CSV esetén:

```bash
make lab-build-features-flow-csv
```

Az eredmény elvárt útvonala: `data/lab/lab_features.csv`.

## 5. Input check és mérési pipeline

```bash
make lab-session-after-capture
make final-real-measurement-package-with-provenance
make final-live-integration
make final-measurement-quality
make final-thesis-integration
make final-submission-check
```

## 6. Beadási csomagolás

Valódi mérés és ellenőrzött provenance után:

```bash
make final-submission-bundle
```

## 7. Validáció

```bash
make repo-hygiene-check
make final-validate
```

## 8. Ellenőrzendő kimenetek

- `data/lab/lab_ground_truth.csv`
- `data/lab/lab_features.csv`
- `data/wazuh/alerts.jsonl`
- `reports/real_measurement/measurement_provenance.json`
- `results/real_comparison/metrics_comparison.csv`
- `reports/measurement_quality/measurement_quality_summary.md`
- `reports/thesis_integration/chapter6_results_generated.md`
- `dist/submission/ai_powered_ids_submission_bundle.zip`
