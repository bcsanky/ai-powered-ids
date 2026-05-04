# Valódi lab mérési checklist

Ez a checklist a natív Wazuh-only, AE-Minimal lab és hibrid Wazuh+AE mérés előkészítéséhez használható.

## Környezet

- [ ] Lefutott a `make live-smoke`.
- [ ] A `reports/live_smoke/live_smoke_readiness.md` státusza READY_FOR_LAB_EXECUTION vagy READY_WITH_WARNINGS.
- [ ] A Docker, ML service, OpenSearch és modellállomány ellenőrzése rendben van, vagy a WARN státusz oka dokumentált.
- [ ] Elkészült a `reports/live_smoke/operator_smoke_brief.md`.
- [ ] A Wazuh manager és dashboard elérhető.
- [ ] A target gép Wazuh agentje látszik a manageren.
- [ ] A target gép naplói megjelennek a Wazuh felületen.
- [ ] A lab gépek órája szinkronban van.
- [ ] A támadásszimuláció kizárólag izolált saját lab hálózaton fut.

## Ground truth

- [ ] Lefutott a `make lab-session-prep`.
- [ ] Elkészült a `reports/lab_session/session_plan.md`.
- [ ] Elkészült a `reports/lab_session/operator_command_log_template.md`, és a mérés közben kitöltésre került.
- [ ] A szcenáriók start/end markerei a `scenario_marker_helper` használatával rögzültek.
- [ ] A `reports/lab_session/run_commands.md` parancslista alapján történt a pipeline előkészítése.
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
- [ ] `make real-measurement-preflight` lefutott a tényleges lab mérés előtt.
- [ ] Zeek input esetén `make lab-build-features-zeek` lefutott.
- [ ] Flow CSV input esetén `make lab-build-features-flow-csv` lefutott.
- [ ] Elkészült a `data/lab/lab_features.csv`.
- [ ] Lefutott a `make lab-validate-real-inputs`.
- [ ] Elkészült a `reports/lab_input_validation/input_validation_report.md`.
- [ ] Elkészült a `reports/lab_input_validation/input_validation_summary.csv`.
- [ ] Lefutott a `make lab-session-after-capture`, és a post-session input check PASS státuszt adott.

## Teljes mérési lánc

- [ ] Lefutott a `make final-real-measurement-package`.
- [ ] Lefutott a `make final-real-hybrid`.
- [ ] Lefutott a `make real-measurement-validate-bundle`.
- [ ] A bundle validáció státusza PASS.
- [ ] Elkészült a `reports/real_measurement/measurement_provenance.json`.
- [ ] Lefutott a `make check-no-demo-real-results`.
- [ ] Lefutott a `make real-measurement-postrun-qa`.
- [ ] A post-run QA státusza READY vagy READY_WITH_LIMITATIONS.
- [ ] Elkészült a `results/wazuh_real/metrics_summary.csv`.
- [ ] Elkészült a `results/ae_lab/metrics_summary.csv`.
- [ ] Elkészült a `results/hybrid_real/metrics_summary.csv`.
- [ ] Elkészült a `results/real_comparison/metrics_comparison.md`.
- [ ] Elkészültek a `results/real_comparison/*.png` ábrák.
- [ ] Elkészült a `reports/real_measurement/real_lab_results_report.md`.
- [ ] Elkészült a `reports/real_measurement/thesis_real_lab_section.md`.
- [ ] Elkészült a `reports/real_measurement/measurement_manifest.md`.
- [ ] Elkészültek a `reports/real_measurement_qa/thesis_table_*.md` táblázatok.
- [ ] Elkészült a `reports/real_measurement_qa/defense_notes_real_measurement.md`.
- [ ] Szükség esetén elkészült az anonimizált riport: `reports/real_measurement_redacted/`.

## Live integration és dashboard-ready kimenet

- [ ] Lefutott a `make final-live-integration`.
- [ ] Elkészült a `reports/live_integration/enriched_alerts.csv`.
- [ ] Elkészült a `reports/live_integration/enriched_alerts.jsonl`.
- [ ] Elkészült a `reports/live_integration/unmatched_alerts.csv`.
- [ ] Az unmatched alert arány elfogadható, vagy a korlátok között külön szerepel.
- [ ] Elkészült a `reports/live_integration/dashboard_payload.json`.
- [ ] Elkészült a `reports/live_integration/dashboard_summary.md`.
- [ ] A `reports/live_integration/live_integration_validation.md` READY vagy READY_WITH_LIMITATIONS státuszt ad.
- [ ] Elkészült a `reports/live_integration/thesis_live_integration_section.md`.
- [ ] OpenSearchbe írás megtörtént, ha dashboard demó szükséges.
- [ ] Az integrációs kimenet nem lett benchmarkként értelmezve.
- [ ] Lefutott a `make lab-session-after-results`.
- [ ] Elkészült a `reports/lab_session/session_summary.md`.

## Dolgozatba emelés

- [ ] A bemeneti fájlok valódi lab mérésből származnak.
- [ ] Nincs kézzel beírt vagy kitalált metrika.
- [ ] A Wazuh-only eredmény natív Wazuh exportból készült.
- [ ] Az AE-only eredmény ugyanazon `event_id` készleten készült.
- [ ] A hibrid eredmény ugyanazon ground truth eseményeken értelmezhető.
- [ ] A korlátok rögzítve vannak: lab mérés, nem hosszú idejű éles SOC-validáció.
- [ ] A 6. fejezetbe történő beemelés a `thesis_readiness.md` alapján megtörtént.
- [ ] A `repo-hygiene-audit` lefutott, és a cleanup terv át lett nézve.
- [ ] A thesis_readiness READY státusza csak érvényes provenance mellett lett elfogadva.

## Dolgozati fejezetintegráció

- [ ] Lefutott a `make final-thesis-integration`.
- [ ] Elkészült a `reports/thesis_integration/chapter5_implementation_generated.md`.
- [ ] Elkészült a `reports/thesis_integration/chapter6_results_generated.md`.
- [ ] Elkészült a `reports/thesis_integration/chapter7_osszegzes_generated.md`.
- [ ] Elkészült a `reports/thesis_integration/chapter8_summary_generated.md`.
- [ ] Elkészült a `reports/thesis_integration/abstract_hu_generated.md`.
- [ ] Elkészült a `reports/thesis_integration/abstract_en_generated.md`.
- [ ] Elkészült a `reports/thesis_integration/figures_plan.md`.
- [ ] Elkészült a `reports/thesis_integration/tables_plan.md`.
- [ ] Elkészült a `reports/thesis_integration/appendix_plan.md`.
- [ ] Elkészült a `reports/thesis_integration/defense_questions_generated.md`.
- [ ] A Wordbe beemelés megtörtént.
- [ ] A metrikák, ábraszámok, táblázatszámok és hivatkozások kézi ellenőrzése megtörtént.

## Final acceptance

- [ ] Lefutott a `make final-acceptance`.
- [ ] A `reports/final_acceptance/release_candidate_readiness.md` státusza READY_FOR_REAL_LAB_RUN vagy READY_WITH_WARNINGS.
- [ ] Elkészült a `reports/final_acceptance/real_lab_execution_brief.md`.
- [ ] A failure mode check sikeres.
- [ ] Nincs tiltott dokumentációs állítás.
- [ ] A tracked futási output guard sikeres.
- [ ] A final acceptance után továbbra is lefutott a `make repo-hygiene-check`.
- [ ] A final acceptance után továbbra is lefutott a `make final-validate`.

## Final submission check

- [ ] Lefutott a `make final-submission-check`.
- [ ] A requirement coverage PASS vagy csak runtime outputokra vonatkozó WARN státuszú.
- [ ] A no-overclaiming check PASS.
- [ ] Elkészült a `reports/final_submission_check/submission_artifact_plan.md`.
- [ ] Elkészült a `reports/final_submission_check/biraloi_risk_questions.md`.
- [ ] A `reports/final_submission_check/final_submission_readiness.md` státusza a mérési állapotnak megfelelő.
