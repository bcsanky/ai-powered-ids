# Lab/replay esettanulmány összefoglaló

Összes esemény száma: 20

## Események szcenáriónként

| scenario | event_count | normal_count | medium_count | high_count | critical_count | avg_anomaly_score | max_anomaly_score | ml_alert_count | rule_alert_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| benign_activity | 5 | 0 | 0 | 5 | 0 | 0.4024 | 0.4494 | 5 | 0 |
| combined_suspicious | 5 | 0 | 0 | 0 | 5 | 0.3265 | 0.4493 | 5 | 5 |
| port_scan | 5 | 0 | 0 | 0 | 5 | 0.3447 | 0.4483 | 5 | 5 |
| ssh_bruteforce | 5 | 0 | 0 | 0 | 5 | 0.4511 | 0.4518 | 5 | 5 |

## Kockázati szintek eloszlása

| risk_level | event_count |
| --- | --- |
| normal | 0 |
| medium | 0 |
| high | 5 |
| critical | 15 |

## Legmagasabb anomáliapontszámú események

| event_id | scenario | risk_level | anomaly_score | reason |
| --- | --- | --- | --- | --- |
| lab-019 | ssh_bruteforce | critical | 0.4518 | Szabályalapú jelzés és ML riasztás egyszerre jelentkezett. |
| lab-012 | ssh_bruteforce | critical | 0.4514 | Szabályalapú jelzés és ML riasztás egyszerre jelentkezett. |
| lab-011 | ssh_bruteforce | critical | 0.4510 | Szabályalapú jelzés és ML riasztás egyszerre jelentkezett. |
| lab-010 | ssh_bruteforce | critical | 0.4506 | Szabályalapú jelzés és ML riasztás egyszerre jelentkezett. |
| lab-009 | ssh_bruteforce | critical | 0.4504 | Szabályalapú jelzés és ML riasztás egyszerre jelentkezett. |

Közepes, magas vagy kritikus kockázati szintet kapott események száma: 20

## Szcenárió-kockázat mátrix

| scenario | normal | medium | high | critical |
| --- | --- | --- | --- | --- |
| benign_activity | 0 | 0 | 5 | 0 |
| combined_suspicious | 0 | 0 | 0 | 5 |
| port_scan | 0 | 0 | 0 | 5 |
| ssh_bruteforce | 0 | 0 | 0 | 5 |

## Szakmai értelmezés

A replay-alapú demonstráció célja annak bemutatása, hogy a pontozási lánc eseményszinten képes kockázati prioritást rendelni normál és gyanúsabb mintázatokhoz. Az eredmények a validált AE-Minimal modell és az egyszerű szabályjelzések kombinált értelmezését szemléltetik.

## Korlát

Ez kontrollált replay-alapú demonstráció, nem éles SOC mérés és nem natív Wazuh teljesítménymérés. A kis elemszámú eseménysor esettanulmányos bemutatásra alkalmas, általános teljesítménykövetkeztetésre nem.
