# Szakértői biztonsági összefoglaló

Futtatás dátuma: 2026-05-04 15:26:22

## Validált konfigurációk

AE-Minimal, AE-Context, Statisztikai baseline, Szabályalapú proxy baseline, Offline hibrid

## Fő összehasonlító metrikák

| configuration | status | precision | recall | f1 | false_positive_rate | alert_count |
| --- | --- | --- | --- | --- | --- | --- |
| AE-Minimal | ok | 0.6203 | 1.0000 | 0.7657 | 1.0000 | 448628.0000 |
| AE-Context | ok | 0.6203 | 1.0000 | 0.7657 | 1.0000 | 448626.0000 |
| Statisztikai baseline | ok | 0.5147 | 0.0320 | 0.0603 | 0.0493 | 17305.0000 |
| Szabályalapú proxy baseline | ok | 0.5253 | 0.0334 | 0.0628 | 0.0493 | 17690.0000 |
| Offline hibrid | ok | 0.6203 | 1.0000 | 0.7657 | 1.0000 | 448628.0000 |
| Natív Wazuh baseline | missing |  |  |  |  |  |

## AE-Minimal és AE-Context összevetése

Az AE-Context F1 értéke az AE-Minimal eredményéhez képest gyakorlatilag azonos.

## Baseline és hibrid státusz

- Statisztikai baseline: ok
- Szabályalapú proxy baseline: ok
- Offline hibrid: ok
- Natív Wazuh baseline: missing

## Legmagasabb kockázatú pontozott események

| event_id | risk_level | anomaly_score | threshold_name | threshold_value | ml_alert | rule_flag | rule_level | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| demo-005 | critical | 0.4536 | f1_optimum | 0.0002 | True | True | 10 | Szabályalapú jelzés és ML riasztás egyszerre jelentkezett. |
| demo-004 | critical | 0.4483 | f1_optimum | 0.0002 | True | True | 5 | Szabályalapú jelzés és ML riasztás egyszerre jelentkezett. |
| demo-006 | critical | 0.4030 | f1_optimum | 0.0002 | True | True | 10 | Szabályalapú jelzés és ML riasztás egyszerre jelentkezett. |
| demo-001 | high | 0.4490 | f1_optimum | 0.0002 | True | False | 0 | Az anomáliapontszám jelentősen meghaladja a kiválasztott küszöböt. |
| demo-002 | high | 0.4453 | f1_optimum | 0.0002 | True | False | 0 | Az anomáliapontszám jelentősen meghaladja a kiválasztott küszöböt. |
| demo-003 | high | 0.1691 | f1_optimum | 0.0002 | True | False | 0 | Az anomáliapontszám jelentősen meghaladja a kiválasztott küszöböt. |

## Korlátok

- A mérés CIC-IDS2017 flow-alapú adatokon történt.
- A Wazuh logorientált adatmodellje és a CIC flow jellemzői között szerkezeti eltérés van.
- A rule_proxy kontrollált flow-alapú szabályproxy, nem natív Wazuh teljesítménymérés.
- A hibrid eredmény offline, azonos teszthalmaz-sorrenden alapuló kiértékelés.
- A riport szakdolgozati demonstrációs összefoglaló, nem éles SOC incidensjelentés.
