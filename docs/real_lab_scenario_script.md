# Real-lab scenario script

Ez nem futtatható shell script, hanem kézzel követhető mérési forgatókönyv. Minden művelet kizárólag saját, izolált lab környezetben végezhető. Nyilvános vagy idegen IP-t tilos célozni; rövid szabály: tilos idegen IP.

Kötelező hivatkozások: provenance, Wazuh, `lab_ground_truth.csv`, `lab_features.csv`, `alerts.jsonl`, `final-real-measurement-package-with-provenance`, `final-measurement-quality`, tilos idegen IP.

Helyettesítendő változók:

- `<ATTACKER_IP>`
- `<TARGET_IP>`
- `<SESSION_ID>`

## A) benign_ssh_login

- Cél: normál SSH belépés.
- Label: benign.
- Attack type: none.

Marker start:

```bash
python -m ml.src.lab_session.scenario_marker_helper start --scenario benign_ssh_login --event-id lab-001 --source-ip <ATTACKER_IP> --target-ip <TARGET_IP>
```

Kézi művelet: az operátor normál SSH belépést végez a saját target VM-re, majd rövid, ártalmatlan parancsokat futtat, például `whoami` és `hostname`.

Marker end:

```bash
python -m ml.src.lab_session.scenario_marker_helper end --event-id lab-001
```

Várható Wazuh viselkedés: normál login események, súlyos riasztás nélkül vagy alacsony szintű autentikációs log.

Várható feature-hatás: alacsony csomagszám, normál SSH célport, rövid flow-időtartam.

## B) benign_package_update

- Cél: normál adminisztratív forgalom.
- Label: benign.
- Attack type: none.

Marker start:

```bash
python -m ml.src.lab_session.scenario_marker_helper start --scenario benign_package_update --event-id lab-002 --source-ip <TARGET_IP> --target-ip <TARGET_IP>
```

Kézi művelet: a target VM-en normál csomaglista-frissítés, például `sudo apt update`, kizárólag saját lab gépen.

Marker end:

```bash
python -m ml.src.lab_session.scenario_marker_helper end --event-id lab-002
```

Várható Wazuh viselkedés: rendszer- vagy csomagkezelési logok, nem támadási célú riasztások.

Várható feature-hatás: több külső kapcsolat, HTTP/HTTPS jellegű forgalom, benign címke.

## C) port_scan

- Cél: port scan jellegű aktivitás saját lab célgépen.
- Label: attack.
- Attack type: port_scan.

Safety warning: csak `<TARGET_IP>` saját izolált lab célgép lehet; tilos idegen IP.

Marker start:

```bash
python -m ml.src.lab_session.scenario_marker_helper start --scenario port_scan --event-id lab-003 --source-ip <ATTACKER_IP> --target-ip <TARGET_IP>
```

Kézi parancspélda saját targetre:

```bash
nmap -sS -T2 -p 22,80,443,8080 <TARGET_IP>
```

Marker end:

```bash
python -m ml.src.lab_session.scenario_marker_helper end --event-id lab-003
```

Várható Wazuh/Zeek hatás: több rövid kapcsolat, több célport, esetleges scan jellegű Wazuh szabály, Zeek conn logban emelkedett célport-változatosság.

## D) ssh_failed_logins

- Cél: néhány sikertelen SSH login.
- Label: attack.
- Attack type: failed_login.

Safety warning: csak saját izolált lab target IP-re és saját teszt felhasználóra vonatkozhat; idegen IP tilos.

Marker start:

```bash
python -m ml.src.lab_session.scenario_marker_helper start --scenario ssh_failed_logins --event-id lab-004 --source-ip <ATTACKER_IP> --target-ip <TARGET_IP>
```

Kézi lépések: az operátor 3-5 alkalommal hibás jelszóval próbál belépni egy saját teszt userre a target VM-en.

Marker end:

```bash
python -m ml.src.lab_session.scenario_marker_helper end --event-id lab-004
```

Várható Wazuh alert: sikertelen autentikációs események, esetleg SSH authentication failure szabály.

Várható feature-hatás: ismétlődő SSH flow-k, rövid kapcsolatminták.

## E) ssh_bruteforce

- Cél: kontrollált brute force jellegű teszt saját tesztuseren.
- Label: attack.
- Attack type: brute_force.

Safety warning: csak saját izolált lab target IP-re és saját teszt felhasználóra vonatkozhat; idegen IP tilos, nyilvános IP tilos, és a próbálkozásszám legyen alacsony. A repository nem automatizál brute force műveletet.

Marker start:

```bash
python -m ml.src.lab_session.scenario_marker_helper start --scenario ssh_bruteforce --event-id lab-005 --source-ip <ATTACKER_IP> --target-ip <TARGET_IP>
```

Kézi lépések: az operátor saját teszt userre, korlátozott számban ismétel sikertelen SSH belépéseket. Ne használj idegen felhasználót vagy nem saját rendszert.

Marker end:

```bash
python -m ml.src.lab_session.scenario_marker_helper end --event-id lab-005
```

Várható Wazuh alert: ismételt sikertelen SSH autentikáció, magasabb súlyosságú riasztás lehet.

Várható feature-hatás: több hasonló SSH flow, emelkedett csomag- vagy kapcsolatgyakoriság.

## F) file_integrity_change

- Cél: fájlváltozás detektálása saját targeten.
- Label: attack.
- Attack type: file_integrity_change.

Safety warning: csak saját izolált lab targeten és erre kijelölt tesztfájlon végezhető; idegen IP tilos.

Marker start:

```bash
python -m ml.src.lab_session.scenario_marker_helper start --scenario file_integrity_change --event-id lab-006 --source-ip <TARGET_IP> --target-ip <TARGET_IP>
```

Kézi lépések: a Wazuh FIM által figyelt tesztkönyvtárban hozz létre vagy módosíts egy tesztfájlt, majd várd meg a Wazuh jelzést.

Marker end:

```bash
python -m ml.src.lab_session.scenario_marker_helper end --event-id lab-006
```

Várható Wazuh FIM jelzés: fájl létrehozás/módosítás/törlés jellegű esemény.

Várható feature-hatás: hálózati flow oldalon kisebb hatás, Wazuh alert oldalon erősebb jel.

## G) privilege_change

- Cél: jogosultságváltozás saját targeten.
- Label: attack.
- Attack type: privilege_change.

Safety warning: csak saját izolált lab targeten és előre létrehozott teszt useren végezhető; idegen IP tilos.

Marker start:

```bash
python -m ml.src.lab_session.scenario_marker_helper start --scenario privilege_change --event-id lab-007 --source-ip <TARGET_IP> --target-ip <TARGET_IP>
```

Kézi lépések: az operátor saját teszt user jogosultságát módosítja, majd visszaállítja az eredeti állapotot. A műveletet naplózni kell az after action review sablonban.

Marker end:

```bash
python -m ml.src.lab_session.scenario_marker_helper end --event-id lab-007
```

Várható Wazuh jelzés: user/group vagy jogosultságváltozás jellegű riasztás.

Várható feature-hatás: Wazuh oldali esemény erős, flow oldali hatás korlátozott lehet.

## Export

A szcenáriók után:

```bash
python -m ml.src.lab_session.scenario_marker_helper export --output data/lab/lab_ground_truth.csv
```

Ezután következik a Wazuh `alerts.jsonl` export, a `lab_features.csv` előállítása Zeek/flow inputból, majd a `final-real-measurement-package-with-provenance` és `final-measurement-quality`.
