# Real-lab mérési riport

## Adateredet és provenance

- measurement_source: `real_lab`
- ground_truth_path: `data/lab/lab_ground_truth.csv`
- ground_truth_sha256: `e5e42bc6a9c94645713ad05e135a884e09a0447f1a2452faef816d31b8d85f92`
- lab_features_path: `data/lab/lab_features.csv`
- lab_features_sha256: `112b9d672b586c117aa3199127c3501ea182bece23a61fd20e5a22c776b2565d`
- wazuh_alerts_path: `data/wazuh/alerts.jsonl`
- wazuh_alerts_sha256: `8fe6e050919349323c5eceb31dcb7cece2b0dbf430e60bb80e94cfa81226e921`

## Mérési cél
A mérés célja annak ellenőrzése, hogy ugyanazon címkézett lab eseményeken hogyan viszonyul egymáshoz a Wazuh-only szabályalapú baseline, az AE-Minimal offline lab pontozás és a hibrid Wazuh+AE döntés.

## Inputok
A mérés a `lab_ground_truth.csv`, a `lab_features.csv` és a Wazuh alert export alapján készült. A metrikák kizárólag ezekből és a feldolgozási lánc kimeneteiből származnak.

## Konfigurációk értelmezése
- Wazuh-only: natív Wazuh alert exportból korrelált szabályalapú jelzés.
- AE-Minimal lab: a végleges AE-Minimal modell offline pontozása a lab feature-ökön.
- Hybrid OR, weighted és priority: a Wazuh és AE jelzések kontrollált kombinációi.

## Fő eredménytábla

| configuration | precision | recall | f1 | false_positive_rate | false_negative_rate | alert_count | mean_ttd | n_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wazuh-only | 0 | 0 | 0 | 0 | 1 | 0 |  | 100 |
| AE-Minimal lab | 0.4 | 1 | 0.571429 | 1 | 0 | 100 |  | 100 |
| Hybrid OR | 0.4 | 1 | 0.571429 | 1 | 0 | 100 |  | 100 |
| Hybrid weighted | 0 | 0 | 0 | 0.0166667 | 1 | 1 |  | 100 |
| Hybrid priority | 0.4 | 1 | 0.571429 | 1 | 0 | 100 |  | 100 |

## Mérnöki értékelés
Az összefoglaló értékelés szerint a vizsgált lab mérésben javulás figyelhető meg: a legjobb hibrid konfiguráció (Hybrid OR) F1 értéke magasabb, mint a Wazuh-only baseline F1 értéke.

- Recall: a Hybrid OR értéke magasabb, mint a Wazuh-only érték.
- FPR: a Hybrid OR értéke nem alacsonyabb, mint a Wazuh-only érték.
- Riasztásszám: a Hybrid OR értéke nem alacsonyabb, mint a Wazuh-only érték.

## Korlátok
- A mérés lab környezetben készült, nem hosszú idejű éles SOC-validáció.
- Az AE-only ág offline scoring, ezért natív detektálási idő csak a Wazuh-alapú riasztásoknál értelmezhető.
- A hibrid döntés minősége az event_id alapú illesztés és az input feature mapping pontosságától függ.
- A Wazuh export teljessége és az időszinkron közvetlenül befolyásolja az eredményt.

## Input validáció

# Lab input validációs jelentés

| Ellenőrzés | Státusz | Részlet |
|---|---|---|
| ground_truth_schema | ok | 100 esemény |
| lab_features_schema | ok | 100 feature sor |
| event_id_consistency | ok | 100 egyező event_id |
| label_coverage | ok | {"benign": 60, "attack": 40} |
| scenario_coverage | ok | 7 scenario |
| wazuh_alert_input | ok | 210 alert sor, formátum: json_or_jsonl |

## Wazuh export összefoglaló

# Wazuh alert export összefoglaló

- Alert count: 210
- Első timestamp: 2026-05-11T13:03:21Z
- Utolsó timestamp: 2026-05-11T16:29:53Z

## Top rule_id-k
- `5402`: 69
- `5501`: 69
- `5502`: 69
- `5503`: 2
- `5404`: 1

## Top rule level értékek
- `3`: 207
- `5`: 2
- `10`: 1

## Top agentek
- `target-ubuntu`: 210

## Top source_ip értékek
Nincs elérhető adat.

## Megjegyzés
Az összefoglaló a megadott Wazuh alert exportból készült, nem hoz létre új mérési eseményt.

