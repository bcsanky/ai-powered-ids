# Esettanulmány: port_scan

## Szcenárió leírása

Több célportot érintő, rövid időtartamú és magasabb csomagrátájú események.

## Bemeneti események jellemzői

| event_id | timestamp | description | rule_flag | rule_level |
| --- | --- | --- | --- | --- |
| lab-005 | 2026-05-08T09:01:00Z | Rövid kapcsolat FTP portra port scan jelleggel | True | 5 |
| lab-006 | 2026-05-08T09:01:03Z | Rövid kapcsolat Telnet portra port scan jelleggel | True | 5 |
| lab-007 | 2026-05-08T09:01:06Z | Rövid kapcsolat SMTP portra port scan jelleggel | True | 5 |
| lab-008 | 2026-05-08T09:01:09Z | Rövid kapcsolat magas portra port scan jelleggel | True | 5 |
| lab-018 | 2026-05-08T09:04:04Z | Rövid kapcsolat adminisztrációs portra | True | 5 |

## Scoring eredmények

| event_id | anomaly_score | threshold_name | ml_alert | rule_flag | risk_level | reason |
| --- | --- | --- | --- | --- | --- | --- |
| lab-005 | 0.4483 | f1_optimum | True | True | critical | Szabályalapú jelzés és ML riasztás egyszerre jelentkezett. |
| lab-006 | 0.4482 | f1_optimum | True | True | critical | Szabályalapú jelzés és ML riasztás egyszerre jelentkezett. |
| lab-007 | 0.4483 | f1_optimum | True | True | critical | Szabályalapú jelzés és ML riasztás egyszerre jelentkezett. |
| lab-008 | 0.1691 | f1_optimum | True | True | critical | Szabályalapú jelzés és ML riasztás egyszerre jelentkezett. |
| lab-018 | 0.2095 | f1_optimum | True | True | critical | Szabályalapú jelzés és ML riasztás egyszerre jelentkezett. |

## Kockázati szintek

| risk_level | event_count |
| --- | --- |
| normal | 0 |
| medium | 0 |
| high | 0 |
| critical | 5 |

## False positive / false negative jellegű megfigyelések

A port_scan szcenárióban 0 esemény kapott normál prioritást. Ezek replay elváráshoz viszonyítva false negative jellegű megfigyelésként lennének értelmezhetők. A jelen kimenet nem formális ground truth mérés.

## Legfontosabb események

| event_id | description | anomaly_score | risk_level |
| --- | --- | --- | --- |
| lab-007 | Rövid kapcsolat SMTP portra port scan jelleggel | 0.4483 | critical |
| lab-005 | Rövid kapcsolat FTP portra port scan jelleggel | 0.4483 | critical |
| lab-006 | Rövid kapcsolat Telnet portra port scan jelleggel | 0.4482 | critical |
| lab-018 | Rövid kapcsolat adminisztrációs portra | 0.2095 | critical |
| lab-008 | Rövid kapcsolat magas portra port scan jelleggel | 0.1691 | critical |

## Elvárt viselkedés és megfigyelések

| event_id | expected_behavior | risk_level | ml_alert | rule_flag |
| --- | --- | --- | --- | --- |
| lab-005 | Legalább közepes kockázati prioritás várható. | critical | True | True |
| lab-006 | Legalább közepes kockázati prioritás várható. | critical | True | True |
| lab-007 | Legalább közepes kockázati prioritás várható. | critical | True | True |
| lab-008 | Magasabb kockázati prioritás várható. | critical | True | True |
| lab-018 | Magasabb kockázati prioritás várható. | critical | True | True |

## Dolgozatba emelhető rövid összefoglaló

A szcenárió azt szemlélteti, hogy a prototípus a bemeneti flow jellemzők és a szabályalapú jelzések alapján eseményszintű prioritást rendel a kontrollált mintákhoz. A megfigyelések demonstrációs jellegűek, és nem helyettesítik a nagy elemszámú mérési benchmarkot.
