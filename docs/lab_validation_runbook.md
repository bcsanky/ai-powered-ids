# Lab/replay validációs futtatási leírás

## Cél

A lab/replay validáció célja a prototípus eseményszintű scoring és riportkészítési folyamatának bemutatása kontrollált eseménysoron. A bemenet normál, port scan, SSH brute force jellegű és kombináltan gyanús mintákat tartalmaz. Ez demonstrációs validáció, nem natív Wazuh teljesítménymérés és nem éles SOC-validáció.

## Bemeneti események szerkezete

A bemenetek helye:

```text
examples/lab/lab_events.jsonl
examples/lab/lab_events.csv
```

Kötelező mezők:

- `event_id`
- `timestamp`
- `scenario`
- `description`
- `destination_port`
- `flow_duration`
- `total_fwd_packets`
- `total_backward_packets`
- `flow_bytes_per_sec`
- `flow_packets_per_sec`
- `protocol`
- `rule_flag`
- `rule_level`
- `expected_behavior`

Szcenáriók:

- `benign_activity`
- `port_scan`
- `ssh_bruteforce`
- `combined_suspicious`

## Batch scoring futtatása

```bash
make score-lab-events
```

Kimenetek:

```text
reports/lab/lab_scored_events.jsonl
reports/lab/lab_scored_events.csv
```

A pontozott események megőrzik a bemeneti `timestamp`, `scenario`, `description` és `expected_behavior` mezőket, így az esettanulmányok visszaköthetők az eredeti kontrollált eseménysorhoz.

## Case study riportok előállítása

```bash
make generate-case-studies
```

Fő kimenetek:

```text
reports/lab/case_study_summary.md
reports/lab/case_study_benign_activity.md
reports/lab/case_study_port_scan.md
reports/lab/case_study_ssh_bruteforce.md
reports/lab/case_study_combined_suspicious.md
reports/lab/scenario_summary.csv
reports/lab/scenario_risk_matrix.csv
reports/lab/lab_timeline.png
reports/lab/risk_level_distribution.png
```

## Ábrák gyűjtése

```bash
make collect-thesis-figures
```

Kimeneti könyvtár:

```text
reports/final/thesis_figures/
```

Az `figure_manifest.csv` rögzíti, hogy mely ábrák kerültek át, melyik forrásfájlból származnak, és mely dolgozati fejezethez kapcsolhatók.

## Eredmények felhasználása a dolgozatban

- Az 5. fejezetben bemutatható a lab/replay eseménysor, a batch scoring és a case study riportkészítés.
- Az 5. fejezet ábráihoz használható a `lab_timeline.png` és a `risk_level_distribution.png`.
- A 6. fejezetben továbbra is a `results/final/` alatti validált mérési eredmények szolgálnak benchmark jellegű összehasonlításként.

## Helyes értelmezés

A lab/replay validáció kis elemszámú, kontrollált eseménysorra épül. Az események nem valós incidensbizonyítékok, nem natív Wazuh exportból származnak, és nem igazolják önmagukban az éles üzemi IDS teljesítményt. A cél az implementált scoring és riportkészítési feldolgozási lánc működésének szemléltetése.
