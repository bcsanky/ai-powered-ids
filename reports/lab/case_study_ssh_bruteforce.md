# Esettanulmány: ssh_bruteforce

## Szcenárió leírása

Ismétlődő SSH kapcsolati kísérleteket leíró események.

## Bemeneti események jellemzői

| event_id | timestamp | description | rule_flag | rule_level |
| --- | --- | --- | --- | --- |
| lab-009 | 2026-05-08T09:02:00Z | Ismétlődő SSH kapcsolati kísérlet | True | 10 |
| lab-010 | 2026-05-08T09:02:04Z | Második SSH brute force jellegű kísérlet | True | 10 |
| lab-011 | 2026-05-08T09:02:08Z | Harmadik SSH brute force jellegű kísérlet | True | 10 |
| lab-012 | 2026-05-08T09:02:12Z | Negyedik SSH brute force jellegű kísérlet | True | 10 |
| lab-019 | 2026-05-08T09:04:08Z | Ötödik SSH brute force jellegű kísérlet | True | 10 |

## Scoring eredmények

| event_id | anomaly_score | threshold_name | ml_alert | rule_flag | risk_level | reason |
| --- | --- | --- | --- | --- | --- | --- |
| lab-009 | 0.4504 | f1_optimum | True | True | critical | Szabályalapú jelzés és ML riasztás egyszerre jelentkezett. |
| lab-010 | 0.4506 | f1_optimum | True | True | critical | Szabályalapú jelzés és ML riasztás egyszerre jelentkezett. |
| lab-011 | 0.4510 | f1_optimum | True | True | critical | Szabályalapú jelzés és ML riasztás egyszerre jelentkezett. |
| lab-012 | 0.4514 | f1_optimum | True | True | critical | Szabályalapú jelzés és ML riasztás egyszerre jelentkezett. |
| lab-019 | 0.4518 | f1_optimum | True | True | critical | Szabályalapú jelzés és ML riasztás egyszerre jelentkezett. |

## Kockázati szintek

| risk_level | event_count |
| --- | --- |
| normal | 0 |
| medium | 0 |
| high | 0 |
| critical | 5 |

## False positive / false negative jellegű megfigyelések

A ssh_bruteforce szcenárióban 0 esemény kapott normál prioritást. Ezek replay elváráshoz viszonyítva false negative jellegű megfigyelésként lennének értelmezhetők. A jelen kimenet nem formális ground truth mérés.

## Legfontosabb események

| event_id | description | anomaly_score | risk_level |
| --- | --- | --- | --- |
| lab-019 | Ötödik SSH brute force jellegű kísérlet | 0.4518 | critical |
| lab-012 | Negyedik SSH brute force jellegű kísérlet | 0.4514 | critical |
| lab-011 | Harmadik SSH brute force jellegű kísérlet | 0.4510 | critical |
| lab-010 | Második SSH brute force jellegű kísérlet | 0.4506 | critical |
| lab-009 | Ismétlődő SSH kapcsolati kísérlet | 0.4504 | critical |

## Elvárt viselkedés és megfigyelések

| event_id | expected_behavior | risk_level | ml_alert | rule_flag |
| --- | --- | --- | --- | --- |
| lab-009 | Magas vagy kritikus kockázati prioritás várható. | critical | True | True |
| lab-010 | Magas vagy kritikus kockázati prioritás várható. | critical | True | True |
| lab-011 | Magas vagy kritikus kockázati prioritás várható. | critical | True | True |
| lab-012 | Magas vagy kritikus kockázati prioritás várható. | critical | True | True |
| lab-019 | Magas vagy kritikus kockázati prioritás várható. | critical | True | True |

## Dolgozatba emelhető rövid összefoglaló

A szcenárió azt szemlélteti, hogy a prototípus a bemeneti flow jellemzők és a szabályalapú jelzések alapján eseményszintű prioritást rendel a kontrollált mintákhoz. A megfigyelések demonstrációs jellegűek, és nem helyettesítik a nagy elemszámú mérési benchmarkot.
