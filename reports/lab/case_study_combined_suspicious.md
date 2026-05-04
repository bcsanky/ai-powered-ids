# Esettanulmány: combined_suspicious

## Szcenárió leírása

Szabályalapú jelzéssel és magasabb forgalmi intenzitással jellemzett események.

## Bemeneti események jellemzői

| event_id | timestamp | description | rule_flag | rule_level |
| --- | --- | --- | --- | --- |
| lab-013 | 2026-05-08T09:03:00Z | Magas csomagráta és szabályjelzés RDP jellegű portra | True | 10 |
| lab-014 | 2026-05-08T09:03:04Z | Magas port és intenzív forgalmi arány | True | 10 |
| lab-015 | 2026-05-08T09:03:08Z | Rövid UDP flow magas csomagrátával | True | 5 |
| lab-016 | 2026-05-08T09:03:12Z | Webes porton szokatlanul nagy forgalmi intenzitás | True | 5 |
| lab-020 | 2026-05-08T09:04:12Z | Magas port, szabályjelzés és intenzív forgalom | True | 10 |

## Scoring eredmények

| event_id | anomaly_score | threshold_name | ml_alert | rule_flag | risk_level | reason |
| --- | --- | --- | --- | --- | --- | --- |
| lab-013 | 0.4032 | f1_optimum | True | True | critical | Szabályalapú jelzés és ML riasztás egyszerre jelentkezett. |
| lab-014 | 0.1652 | f1_optimum | True | True | critical | Szabályalapú jelzés és ML riasztás egyszerre jelentkezett. |
| lab-015 | 0.4493 | f1_optimum | True | True | critical | Szabályalapú jelzés és ML riasztás egyszerre jelentkezett. |
| lab-016 | 0.4444 | f1_optimum | True | True | critical | Szabályalapú jelzés és ML riasztás egyszerre jelentkezett. |
| lab-020 | 0.1703 | f1_optimum | True | True | critical | Szabályalapú jelzés és ML riasztás egyszerre jelentkezett. |

## Kockázati szintek

| risk_level | event_count |
| --- | --- |
| normal | 0 |
| medium | 0 |
| high | 0 |
| critical | 5 |

## False positive / false negative jellegű megfigyelések

A combined_suspicious szcenárióban 0 esemény kapott normál prioritást. Ezek replay elváráshoz viszonyítva false negative jellegű megfigyelésként lennének értelmezhetők. A jelen kimenet nem formális ground truth mérés.

## Legfontosabb események

| event_id | description | anomaly_score | risk_level |
| --- | --- | --- | --- |
| lab-015 | Rövid UDP flow magas csomagrátával | 0.4493 | critical |
| lab-016 | Webes porton szokatlanul nagy forgalmi intenzitás | 0.4444 | critical |
| lab-013 | Magas csomagráta és szabályjelzés RDP jellegű portra | 0.4032 | critical |
| lab-020 | Magas port, szabályjelzés és intenzív forgalom | 0.1703 | critical |
| lab-014 | Magas port és intenzív forgalmi arány | 0.1652 | critical |

## Elvárt viselkedés és megfigyelések

| event_id | expected_behavior | risk_level | ml_alert | rule_flag |
| --- | --- | --- | --- | --- |
| lab-013 | Kritikus vagy magas kockázati prioritás várható. | critical | True | True |
| lab-014 | Kritikus vagy magas kockázati prioritás várható. | critical | True | True |
| lab-015 | Legalább közepes kockázati prioritás várható. | critical | True | True |
| lab-016 | Legalább közepes kockázati prioritás várható. | critical | True | True |
| lab-020 | Kritikus vagy magas kockázati prioritás várható. | critical | True | True |

## Dolgozatba emelhető rövid összefoglaló

A szcenárió azt szemlélteti, hogy a prototípus a bemeneti flow jellemzők és a szabályalapú jelzések alapján eseményszintű prioritást rendel a kontrollált mintákhoz. A megfigyelések demonstrációs jellegűek, és nem helyettesítik a nagy elemszámú mérési benchmarkot.
