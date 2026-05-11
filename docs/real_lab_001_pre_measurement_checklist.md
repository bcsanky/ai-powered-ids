# real-lab-001 meres elotti checklist

Ez a lista a 100 esemenyes valos labor session inditasa elotti allapotot ellenorzi. Nem tartalmaz jelszot, es nem hoz letre meresi eredmenyt.

## Repo es sablonok

- [ ] A branch `thesis/final`.
- [ ] Lefutott: `git pull`.
- [ ] Lefutott: `make lab-templates`.
- [ ] Lefutott:

```bash
make lab-session-prep LAB_SESSION_ID=real-lab-001 ATTACKER_IP=192.168.56.102 TARGET_IP=192.168.56.101 WAZUH_MANAGER_IP=192.168.56.1
```

- [ ] Lefutott: `make real-measurement-preflight`.

## Labor allapot

- [ ] Target VM: `192.168.56.101`.
- [ ] Attacker VM: `192.168.56.102`.
- [ ] Wazuh Manager: `192.168.56.1`.
- [ ] WSL IP: `172.20.31.173`.
- [ ] Wazuh agent status: Active.
- [ ] Kali snapshot elerheto: ready.
- [ ] Target snapshot elerheto: ready.
- [ ] Wazuh login ellenorizve: `admin`; jelszo nincs fajlba irva.
- [ ] Az orak UTC-ben ellenorizve WSL-en, Kalin, Targeten es Wazuh Manageren.

## Capture es export

- [ ] Zeek vagy Flow capture elinditva az elso event marker elott.
- [ ] Kali interfesz azonositva, amelyen a `192.168.56.102` cim van.
- [ ] `tcpdump` telepitve vagy elerheto Kalin.
- [ ] Zeek telepitve vagy elerheto az offline pcap feldolgozashoz.
- [ ] A tcpdump filter szigoruan: `host 192.168.56.101 and host 192.168.56.102`.
- [ ] Wazuh export kezdo es zaro idopontja UTC-ben feljegyezve.
- [ ] A Wazuh export idotartomanya lefedi a teljes merest.
- [ ] A capture csak az utolso event marker utan all le.
- [ ] A valos pcapbol generalva a `conn.log` atmasolva ide: `data/lab/zeek/conn.log`.

## Pilot es tiszta inditas

- [ ] Pilot esemenyek lefutottak kulon, nem a `real-lab-001` vegleges session reszekent.
- [ ] A pilot kulon state fajlt hasznal: `data/lab/pilot_session_events.json`.
- [ ] A pilot utan a cel es a tamado VM session allapota visszaallitva vagy kitakaritva.
- [ ] A vegleges `real-lab-001` session elott nincs nyitott event marker.
- [ ] A vegleges meres elott a `data/lab/session_events.json` el lett tavolitva, ha korabbi state-bol maradt.
- [ ] A vegleges `real-lab-001` session elott nincs felhasznalhato fake alert, fake feature, fake metric vagy fake provenance.
- [ ] A vegrehajtando cel minden attack esemenynel kizarolag `192.168.56.101`.
