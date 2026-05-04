# Real-lab after action review template

Ez kitölthető sablon a mérés utáni operátori összefoglalóhoz. Nem mérési eredmény és nem provenance. Nyilvános vagy idegen IP-t tilos célozni; rövid szabály: tilos idegen IP.

Kötelező hivatkozások: provenance, Wazuh, `lab_ground_truth.csv`, `lab_features.csv`, `alerts.jsonl`, `final-real-measurement-package-with-provenance`, `final-measurement-quality`, tilos idegen IP.

## Alapadatok

- Session ID:
- Dátum:
- Operátor:
- Használt commit hash:
- Branch:

## Lab topológia

- Attacker VM:
- Target VM Wazuh agenttel:
- Wazuh manager / OpenSearch:
- Zeek/flow collector:
- Időszinkron állapota:

## Futtatott szcenáriók

| Event ID | Scenario | Label | Attack type | Sikeres? | Megjegyzés |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

## Eseményösszesítés

- Sikeres események száma:
- Sikertelen vagy megszakított események:
- Benign események:
- Attack események:
- Scenario-k száma:

## Export státusz

- `data/lab/lab_ground_truth.csv` elkészült:
- `data/wazuh/alerts.jsonl` elkészült:
- `data/lab/lab_features.csv` elkészült:
- Wazuh export időablaka:
- Feature build forrása: Zeek / flow CSV

## Pipeline státusz

- `final-real-measurement-package-with-provenance`:
- `final-live-integration`:
- `final-measurement-quality`:
- `final-thesis-integration`:
- `final-submission-check`:
- `final-submission-bundle`:

## Quality gate és kutatási állítás

- Measurement quality státusz:
- Claim strength státusz:
- Submission readiness státusz:
- Használható-e a 6. fejezetben:

## Problémák

- Wazuh export:
- Feature alignment:
- AE scoring:
- Provenance:
- Live integration:
- Egyéb:

## Következő teendők

- [ ] Hiányzó scenario pótlása.
- [ ] Hibás export javítása.
- [ ] Új quality gate futtatás.
- [ ] Dolgozati szöveg kézi ellenőrzése.
- [ ] Melléklet anonimizálása.

