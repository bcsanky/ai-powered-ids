# Esettanulmány: benign_activity

## Szcenárió leírása

Normál jellegű, mérsékelt forgalmi intenzitású események.

## Bemeneti események jellemzői

| event_id | timestamp | description | rule_flag | rule_level |
| --- | --- | --- | --- | --- |
| lab-001 | 2026-05-08T09:00:00Z | Normál HTTP böngészési jellegű flow | False | 0 |
| lab-002 | 2026-05-08T09:00:05Z | Normál HTTPS szolgáltatáselérés | False | 0 |
| lab-003 | 2026-05-08T09:00:10Z | DNS lekérdezés mérsékelt forgalommal | False | 0 |
| lab-004 | 2026-05-08T09:00:15Z | Rövid belső alkalmazásforgalom | False | 0 |
| lab-017 | 2026-05-08T09:04:00Z | Hosszabb, kiegyensúlyozott HTTPS flow | False | 0 |

## Scoring eredmények

| event_id | anomaly_score | threshold_name | ml_alert | rule_flag | risk_level | reason |
| --- | --- | --- | --- | --- | --- | --- |
| lab-001 | 0.4494 | f1_optimum | True | False | high | Az anomáliapontszám jelentősen meghaladja a kiválasztott küszöböt. |
| lab-002 | 0.4451 | f1_optimum | True | False | high | Az anomáliapontszám jelentősen meghaladja a kiválasztott küszöböt. |
| lab-003 | 0.4481 | f1_optimum | True | False | high | Az anomáliapontszám jelentősen meghaladja a kiválasztott küszöböt. |
| lab-004 | 0.2220 | f1_optimum | True | False | high | Az anomáliapontszám jelentősen meghaladja a kiválasztott küszöböt. |
| lab-017 | 0.4475 | f1_optimum | True | False | high | Az anomáliapontszám jelentősen meghaladja a kiválasztott küszöböt. |

## Kockázati szintek

| risk_level | event_count |
| --- | --- |
| normal | 0 |
| medium | 0 |
| high | 5 |
| critical | 0 |

## False positive / false negative jellegű megfigyelések

A benign_activity szcenárióban 5 esemény kapott közepes vagy magasabb prioritást. Ez replay elváráshoz viszonyított false positive jellegű megfigyelésként kezelhető, nem formális mérési címkeként.

## Legfontosabb események

| event_id | description | anomaly_score | risk_level |
| --- | --- | --- | --- |
| lab-001 | Normál HTTP böngészési jellegű flow | 0.4494 | high |
| lab-003 | DNS lekérdezés mérsékelt forgalommal | 0.4481 | high |
| lab-017 | Hosszabb, kiegyensúlyozott HTTPS flow | 0.4475 | high |
| lab-002 | Normál HTTPS szolgáltatáselérés | 0.4451 | high |
| lab-004 | Rövid belső alkalmazásforgalom | 0.2220 | high |

## Elvárt viselkedés és megfigyelések

| event_id | expected_behavior | risk_level | ml_alert | rule_flag |
| --- | --- | --- | --- | --- |
| lab-001 | Alacsonyabb kockázati prioritás várható. | high | True | False |
| lab-002 | Alacsonyabb kockázati prioritás várható. | high | True | False |
| lab-003 | Alacsonyabb kockázati prioritás várható. | high | True | False |
| lab-004 | Alacsonyabb kockázati prioritás várható. | high | True | False |
| lab-017 | Alacsonyabb kockázati prioritás várható. | high | True | False |

## Dolgozatba emelhető rövid összefoglaló

A szcenárió azt szemlélteti, hogy a prototípus a bemeneti flow jellemzők és a szabályalapú jelzések alapján eseményszintű prioritást rendel a kontrollált mintákhoz. A megfigyelések demonstrációs jellegűek, és nem helyettesítik a nagy elemszámú mérési benchmarkot.
