# End-to-end live integration runbook

Ez a runbook azt a feldolgozási láncot írja le, amelyben a Wazuh alert exportból származó események AE-Minimal scoringgal és hibrid kockázati döntéssel gazdagíthatók. A cél dashboard-ready kimenet előállítása, nem metrikai benchmark és nem éles üzemi SOC-rendszer igazolása.

## Cél

A live integration lánc a következő adatáramlást valósítja meg:

```text
Wazuh alert export -> lab feature lookup -> AE-Minimal scoring -> hibrid prioritás -> enriched alert output -> dashboard payload
```

A kimenet mérnöki integrációs bizonyíték: azt mutatja meg, hogy a Wazuh alert, az ML scoring és a hibrid kockázati priorizálás összekapcsolható ugyanazon real-lab mérési csomag alapján.

## Szükséges inputok

- `data/wazuh/alerts.jsonl`: tényleges Wazuh/OpenSearch alert export.
- `data/lab/lab_features.csv`: a lab eseményekhez tartozó AE-Minimal kompatibilis feature állomány.
- `data/lab/lab_ground_truth.csv`: event_id, időablak, source_ip és target_ip mezőket tartalmazó ground truth.
- `reports/real_measurement/measurement_provenance.json`: a real-lab inputok SHA256 azonosítóit rögzítő provenance fájl.
- `artifacts/final/final-ae-minimal-v1`: a validált AE-Minimal modell futtatási könyvtára.
- `data/processed/final/ae_minimal/preprocess.pkl`: az AE-Minimal preprocess állomány.

Az `examples`, `templates`, `tests`, `sample`, `demo` vagy `fixture` eredetű inputok real integration módban nem használhatók.

## Futtatás

Előfeltételként a real-lab mérési csomag provenance-szel együtt készüljön el:

```bash
make final-real-measurement-package-with-provenance
```

Ezután az end-to-end integráció futtatása:

```bash
make final-live-integration
```

A cél egymás után futtatja:

- `check-no-demo-real-results`
- `live-enrich-alerts`
- `live-dashboard-payload`
- `live-validate`
- `live-thesis-section`

## OpenSearchbe írás

Ha dashboard demóhoz OpenSearch indexbe kell írni a gazdagított alert eseményeket:

```bash
make final-live-integration-opensearch OPENSEARCH_PASSWORD=<jelszo>
```

Az OpenSearch jelszó nem kerül metadata fájlba. A `OPENSEARCH_VERIFY_TLS=false` csak izolált lab környezetben javasolt.

## Kimenetek

- `reports/live_integration/enriched_alerts.jsonl`
- `reports/live_integration/enriched_alerts.csv`
- `reports/live_integration/unmatched_alerts.csv`
- `reports/live_integration/enrichment_summary.csv`
- `reports/live_integration/dashboard_payload.json`
- `reports/live_integration/dashboard_summary.md`
- `reports/live_integration/dashboard_cards.csv`
- `reports/live_integration/live_integration_validation.md`
- `reports/live_integration/live_integration_readiness.json`
- `reports/live_integration/thesis_live_integration_section.md`
- `reports/live_integration/live_integration_defense_notes.md`

## Matching logika

A feature mapping sorrendje:

1. `event_id` alapú illesztés, ha az alert tartalmaz event_id mezőt.
2. Időbélyeg + source_ip + target_ip alapú illesztés a ground truth időablak alapján.
3. Csak időablak alapú illesztés kizárólag akkor, ha az `ALLOW_TIME_ONLY_MATCH=true` kapcsoló engedélyezett.

Ha nincs illesztés, az alert az `unmatched_alerts.csv` állományba kerül, és nem kap ML pontszámot.

## Hibrid kockázati döntés

- `critical`: Wazuh pozitív jelzés és ML pozitív döntés együtt.
- `high`: csak Wazuh pozitív jelzés.
- `medium`: csak ML pozitív döntés.
- `normal`: egyik ág sem pozitív.

Ez a prioritási logika dashboard és szakértői áttekintés céljára készült. A detektálási teljesítmény összehasonlítását továbbra is a real-lab benchmark táblázat adja.

## Dolgozati felhasználás

Az integrációs kimenet a diplomamunka 5. fejezetében mutatható be a prototípus végponttól végpontig tartó működéseként. A 6. fejezetben csak akkor használható real-lab bizonyítékként, ha a provenance verified real-lab státuszú, és a `live_integration_validation.md` READY vagy READY_WITH_LIMITATIONS státuszt ad.

## Korlátok

- A live integration output nem önálló benchmark.
- Nem igazolja éles üzemi SOC-rendszer teljesítményét.
- Az unmatched alert arány közvetlenül befolyásolja az ML scoring lefedettségét.
- A dashboard payload csak ugyanabból a verified real-lab mérésből származó inputokkal értelmezhető.
- A Wazuh alert mezők elérhetősége exportformátumtól és Wazuh szabálytól függ.

