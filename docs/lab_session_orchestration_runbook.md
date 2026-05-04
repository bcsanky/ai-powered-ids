# Lab session orchestration runbook

Ez a runbook a tényleges real-lab mérés operátori végrehajtását támogatja. A session réteg nem generál mérési inputot, nem indít támadó parancsot, nem hoz létre Wazuh alertet és nem számol metrikát. Feladata az előfeltételek ellenőrzése, a mérési terv és parancsnapló előkészítése, valamint a mérés utáni inputok meglétének ellenőrzése.

## Mit csinál?

- ellenőrzi a projekt, sablonok, dokumentációk és Python modulok elérhetőségét;
- létrehoz egy session tervet;
- létrehoz egy kitölthető operátori parancsnapló sablont;
- wrapperként segíti a scenario marker start/end/export használatát;
- ellenőrzi, hogy a ground truth, Wazuh export és Zeek/flow input ténylegesen létrejött-e;
- parancslistát készít a mérési pipeline kézi futtatásához;
- összefoglalja, hogy a session után mely inputok és outputok állnak rendelkezésre.

## Mit nem csinál?

- nem állít elő `lab_ground_truth.csv` fájlt tényleges marker használat nélkül;
- nem állít elő `lab_features.csv` fájlt valós Zeek vagy flow input nélkül;
- nem állít elő `data/wazuh/alerts.jsonl` fájlt;
- nem futtat port scan, brute force vagy más támadó parancsot;
- nem hoz létre metrikát vagy Wazuh+AE eredményt.

## Mérés előtti előkészítés

```bash
make lab-session-prep LAB_SESSION_ID=real-lab-YYYYMMDD ATTACKER_IP=<ip> TARGET_IP=<ip> WAZUH_MANAGER_IP=<ip>
```

A cél létrehozza vagy frissíti:

- `reports/lab_session/session_doctor_report.md`
- `reports/lab_session/session_plan.yaml`
- `reports/lab_session/session_plan.md`
- `reports/lab_session/operator_command_log_template.md`
- `reports/lab_session/run_commands.md`

Ezek futási segédletek, nem mérési eredmények.

## Marker használat mérés közben

Szcenárió indítása:

```bash
python -m ml.src.lab_session.scenario_marker_helper start \
  --scenario port_scan \
  --event-id lab-003 \
  --source-ip 192.168.56.20 \
  --target-ip 192.168.56.10
```

Szcenárió lezárása:

```bash
python -m ml.src.lab_session.scenario_marker_helper end --event-id lab-003
```

Ground truth export:

```bash
python -m ml.src.lab_session.scenario_marker_helper export \
  --output data/lab/lab_ground_truth.csv
```

A ground truth csak akkor tekinthető valid mérési inputnak, ha a markerek tényleges lab futás közben lettek rögzítve.

## Mérés utáni input ellenőrzés

Miután elkészült a Wazuh alert export és a Zeek/flow input:

```bash
make lab-session-after-capture
```

Ez ellenőrzi:

- `data/lab/lab_ground_truth.csv`;
- `data/wazuh/alerts.jsonl`;
- `data/lab/zeek/conn.log` vagy `data/lab/flows.csv`.

## Valódi pipeline futtatása

Zeek vagy flow input előállítása után:

```bash
make final-real-measurement-package-with-provenance
make final-live-integration
```

Ezek már a tényleges real-lab mérési és integrációs lánc részei. Demo vagy sample bemenetet a guardok nem fogadnak el real-lab inputként.

## Session összefoglaló

```bash
make lab-session-after-results
```

Kimenetek:

- `reports/lab_session/session_summary.md`
- `reports/lab_session/session_summary.json`

Az összefoglaló csak meglévő inputokat és kimeneteket listáz. Nem talál ki metrikát.

## Beadási melléklet

A beadási mellékletbe a provenance és manifest alapján kerülhetnek állományok. Jellemzően:

- validált metrika CSV-k;
- dolgozatba beemelt ábrák;
- `measurement_provenance.json`;
- `measurement_manifest.*`;
- anonimizált riportok;
- szükség esetén raw inputok, ha adatvédelmi szempontból megengedett.

## Git hygiene

A `reports/lab_session/` futási output, ezért nem kerül Gitbe. A session terv és parancsnapló a mérési végrehajtást dokumentálja, de nem kutatási eredmény. A valódi eredményt kizárólag a provenance-ben rögzített inputokból számolt kimenetek adják.

