# Lab támadásszimulációs runbook

Ez a runbook a valós lab-alapú Wazuh-only, AE-only és hibrid Wazuh+AE mérés bemeneteinek előállítását írja le. A leírt műveletek kizárólag saját, izolált lab környezetben hajthatók végre. A cél nem éles környezet támadása, hanem reprodukálható, címkézett mérési események előállítása a diplomamunka validációjához.

## Lab környezet előfeltételei

Ajánlott szerepek:

| Szerep | Példa | Feladat |
|---|---|---|
| Attacker gép | Kali vagy Linux kliens | Kontrollált tesztparancsok indítása saját lab hálózaton. |
| Target gép | Linux VM Wazuh agenttel | A vizsgált események fogadása és naplózása. |
| Wazuh manager/dashboard | Wazuh stack | Alert gyűjtés, szabályalapú riasztás és export. |
| Flow szenzor | Zeek vagy flow export | `conn.log` vagy flow CSV előállítása az AE feature buildhez. |

Kritikus előfeltételek:

- minden gép órája legyen szinkronban;
- a Wazuh agent látszódjon a manageren;
- a lab hálózat izolált legyen, és ne célozzon nyilvános IP-t;
- brute force jellegű teszt csak saját teszt userre és saját teszt VM-re történjen;
- minden esemény előtt és után rögzíteni kell a ground truth időablakot.

## Ground truth rögzítése

Esemény indítása:

```bash
python -m ml.src.lab_capture.event_marker start \
  --event-id lab-001 \
  --scenario port_scan \
  --label attack \
  --attack-type port_scan \
  --source-ip 192.168.56.20 \
  --target-ip 192.168.56.10
```

Esemény lezárása:

```bash
python -m ml.src.lab_capture.event_marker end --event-id lab-001
```

Ground truth export:

```bash
python -m ml.src.lab_capture.event_marker export \
  --output data/lab/lab_ground_truth.csv
```

Az exportált CSV-t a Wazuh baseline validátor is ellenőrzi, ezért lezáratlan esemény vagy hibás címke esetén a feldolgozás megáll.

## Wazuh alert export

A Wazuh Dashboard vagy indexer felületén a lab futás időintervallumára kell szűrni az alert indexeket, majd JSON vagy JSONL formátumban exportálni az eredményt:

```text
data/wazuh/alerts.jsonl
```

Az exportnál különösen fontosak az időbélyeg, a rule azonosító, a rule level, a rule description, az agent neve, a forrás IP és a cél IP mezők. A parser több gyakori Wazuh mezőelnevezést kezel, de a pontos mezők meglétét a mérés előtt ellenőrizni kell.

## Zeek conn.log vagy flow CSV előállítása

Zeek használata esetén a lab forgalom alapján `conn.log` állományt kell előállítani:

```text
data/lab/zeek/conn.log
```

Ha Zeek nem áll rendelkezésre, használható általános flow CSV is:

```text
data/lab/flows.csv
```

A flow CSV oszlopnevei a Makefile célokhoz tartozó script paramétereivel módosíthatók.

## Szcenáriók

### benign_ssh_login

- Cél: normál, engedélyezett SSH belépés rögzítése.
- Label: `benign`
- Attack type: üres
- Javasolt parancs saját labban:

```bash
ssh testuser@192.168.56.10
```

- Ground truth: start a belépés előtt, end a kilépés után.
- Várható Wazuh alert: legfeljebb alacsony vagy információs szintű bejelentkezési esemény.
- Várható feature-hatás: célport 22, mérsékelt csomagszám és időtartam.

### benign_package_update

- Cél: normál adminisztratív forgalom és naplóesemény rögzítése.
- Label: `benign`
- Attack type: üres
- Javasolt parancs saját lab targeten:

```bash
sudo apt update
```

- Ground truth: start a frissítés előtt, end a parancs lefutása után.
- Várható Wazuh alert: normál rendszeresemények, esetleg package manager naplók.
- Várható feature-hatás: HTTP/HTTPS vagy DNS forgalom, változó flow méret.

### port_scan

- Cél: kontrollált port scan detektálhatóságának vizsgálata.
- Label: `attack`
- Attack type: `port_scan`
- Javasolt parancs saját izolált labban:

```bash
nmap -sS -p 1-1000 192.168.56.10
```

- Ground truth: start közvetlenül a scan előtt, end a scan befejezése után.
- Várható Wazuh alert: port scan vagy hálózati gyanús aktivitás, ha a szabályok és logforrások támogatják.
- Várható feature-hatás: több célport, rövid flow-k, magasabb csomagráta.

### ssh_failed_logins

- Cél: néhány sikertelen SSH belépési kísérlet rögzítése.
- Label: `attack`
- Attack type: `ssh_failed_logins`
- Javasolt parancs saját teszt userrel:

```bash
ssh wronguser@192.168.56.10
```

- Ground truth: start az első próbálkozás előtt, end az utolsó után.
- Várható Wazuh alert: sikertelen SSH belépés.
- Várható feature-hatás: célport 22, ismételt rövid kapcsolatok.

### ssh_bruteforce

- Cél: korlátozott, labon belüli brute force jellegű mintázat előállítása.
- Label: `attack`
- Attack type: `ssh_bruteforce`
- Javasolt parancs saját teszt userre, alacsony próbálkozásszámmal:

```bash
for i in 1 2 3 4 5; do ssh -o BatchMode=yes testuser@192.168.56.10 true; done
```

- Ground truth: start a ciklus előtt, end a ciklus után.
- Várható Wazuh alert: több sikertelen SSH autentikáció vagy brute force jellegű szabály.
- Várható feature-hatás: célport 22, ismételt kapcsolatok, magasabb alert szint.

### file_integrity_change

- Cél: Wazuh file integrity monitoring esemény vizsgálata.
- Label: `attack`
- Attack type: `file_integrity_change`
- Javasolt parancs saját lab targeten, előre kijelölt tesztfájlon:

```bash
echo "lab-test" | sudo tee -a /tmp/wazuh_fim_test.txt
```

- Ground truth: start a módosítás előtt, end az alert várható beérkezése után.
- Várható Wazuh alert: fájlmódosítási esemény, ha a fájl útvonala monitorozott.
- Várható feature-hatás: hálózati flow hatás csekély lehet, ezért ez a Wazuh-only ág számára lehet informatívabb.

### privilege_change

- Cél: jogosultságváltozáshoz kapcsolódó Wazuh esemény vizsgálata.
- Label: `attack`
- Attack type: `privilege_change`
- Javasolt parancs saját lab userrel és visszaállítási tervvel:

```bash
sudo usermod -aG sudo testuser
```

- Ground truth: start a módosítás előtt, end a naplóesemény beérkezése után.
- Várható Wazuh alert: user vagy group változás, ha a szabályok és logok támogatják.
- Várható feature-hatás: hálózati flow hatás nem feltétlenül jelentős; Wazuh log oldali jelzés lehet domináns.

## Teljes mérési sorrend

1. Sablonok létrehozása:

```bash
make lab-templates
```

2. Események rögzítése `event_marker.py` start/end parancsokkal.
3. Ground truth export:

```bash
python -m ml.src.lab_capture.event_marker export --output data/lab/lab_ground_truth.csv
```

4. Wazuh alert export mentése `data/wazuh/alerts.jsonl` útvonalra.
5. Zeek conn.log vagy flow CSV előállítása.
6. Feature build és teljes hibrid mérés:

```bash
make final-real-hybrid-zeek
```

vagy:

```bash
make final-real-hybrid-flow-csv
```

7. Eredmények ellenőrzése:

```text
results/real_comparison/metrics_comparison.md
results/real_comparison/*.png
```
