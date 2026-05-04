# Valódi lab mérési checklist

Ez a checklist a natív Wazuh-only, AE-Minimal lab és hibrid Wazuh+AE mérés előkészítéséhez használható.

## Környezet

- [ ] A Wazuh manager és dashboard elérhető.
- [ ] A target gép Wazuh agentje látszik a manageren.
- [ ] A target gép naplói megjelennek a Wazuh felületen.
- [ ] A lab gépek órája szinkronban van.
- [ ] A támadásszimuláció kizárólag izolált saját lab hálózaton fut.

## Ground truth

- [ ] Minden szcenáriónál használtam az `event_marker.py start` parancsot.
- [ ] Minden szcenáriónál használtam az `event_marker.py end` parancsot.
- [ ] Nincs lezáratlan esemény a `data/lab/session_events.json` fájlban.
- [ ] Elkészült a `data/lab/lab_ground_truth.csv`.
- [ ] A ground truth tartalmaz benign és attack eseményt is.
- [ ] Legalább két különböző scenario szerepel a mérésben.

## Wazuh export

- [ ] A lab időablakra szűrt Wazuh alert export elkészült.
- [ ] Az export útvonala: `data/wazuh/alerts.jsonl` vagy `data/wazuh/alerts.json`.
- [ ] Az export metaadata elkészült: `data/wazuh/alerts_export_metadata.json` vagy `data/wazuh/alerts_file_export_metadata.json`.
- [ ] A Wazuh export összefoglaló elkészült: `reports/wazuh_export/wazuh_export_summary.md`.
- [ ] Az export tartalmaz timestamp, rule id, rule level és IP mezőket, ha ezek elérhetők.
- [ ] A Wazuh alert érkezését legalább egy ismert teszteseménnyel ellenőriztem.

## Flow / Zeek input

- [ ] Elkészült a Zeek `conn.log`, vagy rendelkezésre áll egy flow CSV.
- [ ] Zeek esetén az útvonal: `data/lab/zeek/conn.log`.
- [ ] Flow CSV esetén az útvonal: `data/lab/flows.csv`.
- [ ] A flow adatok időbélyege ugyanahhoz az időzónához vagy UTC értelmezéshez köthető.
- [ ] A flow adatok tartalmazzák a forrás IP, cél IP, célport, protokoll és csomagszám mezőket.

## Feature build és input validáció

- [ ] `make lab-templates` lefutott, ha sablonokra volt szükség.
- [ ] Zeek input esetén `make lab-build-features-zeek` lefutott.
- [ ] Flow CSV input esetén `make lab-build-features-flow-csv` lefutott.
- [ ] Elkészült a `data/lab/lab_features.csv`.
- [ ] Lefutott a `make lab-validate-real-inputs`.
- [ ] Elkészült a `reports/lab_input_validation/input_validation_report.md`.
- [ ] Elkészült a `reports/lab_input_validation/input_validation_summary.csv`.

## Teljes mérési lánc

- [ ] Lefutott a `make final-real-hybrid`.
- [ ] Lefutott a `make real-measurement-validate-bundle`.
- [ ] A bundle validáció státusza PASS.
- [ ] Elkészült a `results/wazuh_real/metrics_summary.csv`.
- [ ] Elkészült a `results/ae_lab/metrics_summary.csv`.
- [ ] Elkészült a `results/hybrid_real/metrics_summary.csv`.
- [ ] Elkészült a `results/real_comparison/metrics_comparison.md`.
- [ ] Elkészültek a `results/real_comparison/*.png` ábrák.
- [ ] Elkészült a `reports/real_measurement/real_lab_results_report.md`.
- [ ] Elkészült a `reports/real_measurement/thesis_real_lab_section.md`.
- [ ] Elkészült a `reports/real_measurement/measurement_manifest.md`.
- [ ] Szükség esetén elkészült az anonimizált riport: `reports/real_measurement_redacted/`.

## Dolgozatba emelés

- [ ] A bemeneti fájlok valódi lab mérésből származnak.
- [ ] Nincs kézzel beírt vagy kitalált metrika.
- [ ] A Wazuh-only eredmény natív Wazuh exportból készült.
- [ ] Az AE-only eredmény ugyanazon `event_id` készleten készült.
- [ ] A hibrid eredmény ugyanazon ground truth eseményeken értelmezhető.
- [ ] A korlátok rögzítve vannak: lab mérés, nem hosszú idejű éles SOC-validáció.
