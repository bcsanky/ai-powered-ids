# real-lab-001 missing-flow rerun runbook

Ez a runbook a `real-lab-001` mérés 22 hiányzó Zeek-flow eseményének célzott újrafuttatásához készült. Nem készít fake pcapet, fake `conn.log`-ot, fake alertet, fake feature-t, fake metrikát vagy fake provenance-t.

## Miért kell rerun

A végleges mérésben 100 ground truth esemény szerepelt, de a Zeek feature build 22 event ID-hez nem talált illeszkedő flow-t. Ezeket az eseményeket külön rerun sessionben kell újramarkerelni, új hálózati kapcsolattal, hogy a Zeek `conn.log` tényleges flow-t tartalmazzon az adott event window-ban.

Érintett event ID-k: `ABRUTE-006`, `APORT-007`, `APORT-008`, `APRIV-006`, `APRIV-008`, `BPKG-014`, `BPKG-016`, `BPKG-021`, `BPKG-025`, `BPKG-026`, `BPKG-027`, `BPKG-028`, `BPKG-029`, `BPKG-030`, `BSSH-015`, `BSSH-024`, `BSSH-025`, `BSSH-026`, `BSSH-027`, `BSSH-028`, `BSSH-029`, `BSSH-030`.

## Biztonsági keret

- Engedélyezett cél: `192.168.56.101`.
- Attacker/Kali: `192.168.56.102`.
- Wazuh Manager: `192.168.56.1`.
- Tilos: public IP, céges hálózat, router, más VM, internetes host.
- Minden rerun lab művelethez új SSH/network kapcsolat kell; már nyitott SSH sessiont ne használj.

## 1. Tcpdump indítása Kalin

Kalin az `eth1` host-only interfészen indítsd a capture-t a rerun első marker előtt:

```bash
mkdir -p ~/real-lab-001-rerun-missing-flow
sudo tcpdump -i eth1 -s 0 -nn \
  -w ~/real-lab-001-rerun-missing-flow/rerun_missing_flow.pcap \
  'host 192.168.56.101 and host 192.168.56.102'
```

Ha az `eth1` nem a `192.168.56.102` címet hordozza, ne indítsd el a rerunt; előbb ellenőrizd az interfészt:

```bash
ip -br addr
```

## 2. Rerun marker parancsok futtatása

WSL-ben, a repository gyökérkönyvtárából használd a külön state fájlt tartalmazó sablont:

```text
templates/lab/rerun_missing_flow_event_marker_commands.md
```

A state fájl:

```text
data/lab/rerun_missing_flow_events.json
```

Minden markerablakban nyiss új kapcsolatot a target felé, majd zárd le az eseményt. Az események ne fedjenek át, és két esemény között hagyj legalább 60-90 másodpercet.

## 3. Tcpdump leállítása

Az utolsó rerun marker lezárása után várj röviden, majd a Kali tcpdump terminálban:

```text
Ctrl+C
```

Jegyezd fel UTC-ben a rerun capture start/end időpontját. A Wazuh export új teljes időablakának a régi mérés és a rerun időtartamát is fednie kell.

## 4. Rerun conn.log generálása

Kalin vagy WSL-ben, ahol Zeek elérhető:

```bash
mkdir -p ~/real-lab-001-rerun-missing-flow/zeek
cd ~/real-lab-001-rerun-missing-flow/zeek
zeek -r ~/real-lab-001-rerun-missing-flow/rerun_missing_flow.pcap
```

Másold a valós rerun `conn.log`-ot a repo külön rerun bemenetére:

```bash
mkdir -p data/lab/rerun_missing_flow
cp ~/real-lab-001-rerun-missing-flow/zeek/conn.log data/lab/rerun_missing_flow/conn.log
```

## 5. Rerun ground truth export

WSL-ben, a repository gyökérkönyvtárából:

```bash
python3 -m ml.src.lab_capture.event_marker \
  --state-file data/lab/rerun_missing_flow_events.json \
  export --output data/lab/rerun_missing_flow/lab_ground_truth.csv
```

Ez az export csak a 22 rerun eseményt tartalmazza.

## 6. Ground truth merge

Előbb készíts lokális backupot az eredeti 100 soros ground truthról:

```bash
cp data/lab/lab_ground_truth.csv data/lab/lab_ground_truth.before_missing_flow_rerun.csv
```

Ezután a 22 rerun sort cseréld be az eredeti sorrend megtartásával:

```bash
python3 scripts/merge_missing_flow_rerun_ground_truth.py \
  --base data/lab/lab_ground_truth.csv \
  --rerun data/lab/rerun_missing_flow/lab_ground_truth.csv \
  --output data/lab/lab_ground_truth.csv
```

A script hibázik, ha a rerun ismeretlen event ID-t tartalmaz, ha duplikált event ID van, vagy ha a 100 soros eloszlás sérül.

## 7. Zeek conn.log összefűzés

Előbb mentsd el az eredeti Zeek logot külön backupként:

```bash
cp data/lab/zeek/conn.log data/lab/zeek/conn.original_real_lab_001.log
```

Majd fűzd össze az eredeti és rerun Zeek logot:

```bash
bash scripts/merge_zeek_conn_logs.sh
```

Alapértelmezett bemenetek:

- original backup: `data/lab/zeek/conn.original_real_lab_001.log`
- rerun conn.log: `data/lab/rerun_missing_flow/conn.log`
- output: `data/lab/zeek/conn.log`

## 8. Wazuh alert export új teljes időablakra

Az új Wazuh export időtartománya fedje le az eredeti 100 eseményes mérést és a rerun időszakot is:

```bash
make wazuh-export-opensearch \
  WAZUH_EXPORT_START=<eredeti_meres_kezdete_UTC> \
  WAZUH_EXPORT_END=<rerun_vege_UTC> \
  OPENSEARCH_PASSWORD=<helyi_titok>
```

Jelszó nem kerülhet fájlba vagy Gitbe.

## 9. Zeek feature build és validáció

```bash
make lab-build-features-zeek
make lab-validate-real-inputs
make final-real-measurement-package-zeek
make real-measurement-provenance
```

Ha továbbra is van missing flow, ne pótold kézzel a `conn.log` sort. Ellenőrizd a capture időablakot, az event marker időket, és hogy minden rerun eseményben új hálózati kapcsolat nyílt-e.
