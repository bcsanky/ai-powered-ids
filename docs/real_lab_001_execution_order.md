# real-lab-001 Zeek-alapu vegrehajtasi sorrend

Ez a sorrend a 100 esemenyes session kezi futtatasahoz keszult. A cel kizarolag `192.168.56.101`; public IP, ceges halozat, router, mas VM es internetes host tilos.

## 1. Repo elokeszites WSL-ben

```bash
git checkout thesis/final
git pull
make lab-templates
make lab-session-prep LAB_SESSION_ID=real-lab-001 ATTACKER_IP=192.168.56.102 TARGET_IP=192.168.56.101 WAZUH_MANAGER_IP=192.168.56.1
make real-measurement-preflight
```

## 2. Idoellenorzes

Ellenorizd UTC-ben az orat WSL-en, Kalin, Targeten es a Wazuh Manageren. Az event marker, Wazuh export es Zeek korrelacio csak akkor ertelmezheto, ha az idok kozel azonosak.

```bash
date -u
```

## 3. Zeek pcap capture inditasa Kalin

Azonositsd a Kali `192.168.56.102` cimet hordozo interfeszet:

```bash
ip -br addr
export KALI_IFACE=<interface_with_192.168.56.102>
```

Inditsd el a capture-t az elso pilot vagy vegleges marker elott:

```bash
mkdir -p ~/real-lab-001-capture
sudo tcpdump -i "$KALI_IFACE" -s 0 -nn \
  -w ~/real-lab-001-capture/real-lab-001.pcap \
  'host 192.168.56.101 and host 192.168.56.102'
```

## 4. Pilot kulon state fajllal

Futtasd a harom pilot markert kulon state fajlba:

```text
data/lab/pilot_session_events.json
```

Hasznald a sablont:

```text
templates/lab/pilot_event_marker_commands_real_lab_001.md
```

A pilot eredmenye nem resze a vegleges `real-lab-001` 100 esemenyes ground truthnak.

## 5. Vegleges session tisztitasa

A 100 esemenyes session elott ne maradjon nyitott vagy pilotbol szarmazo vegleges state:

```bash
rm -f data/lab/session_events.json
```

Ellenorizd, hogy a pilot state tovabbra is kulon fajlban maradt, es nem keveredik a vegleges exporttal.

## 6. 100 event_marker start/end futtatas

Kovesd a 100 esemenyes marker sablont:

```text
templates/lab/event_marker_commands_100.md
```

Minden esemeny legyen idoben kulonallo. Ket esemeny kozott ajanlott 60-90 masodperc szunet. Attack jellegu muvelet csak `192.168.56.101` cel ellen tortenhet.

## 7. Capture leallitasa

Az utolso `event_marker end` utan allitsd le a tcpdumpot a Kali capture terminalban:

```text
Ctrl+C
```

Jegyezd fel UTC-ben a capture stop idopontot es a Wazuh export zaro idopontjat.

## 8. Zeek conn.log generalas es masolas

```bash
mkdir -p ~/real-lab-001-zeek
cd ~/real-lab-001-zeek
zeek -r ~/real-lab-001-capture/real-lab-001.pcap
```

Masold a tenyleges `conn.log` fajlt a repo vart bemenetere:

```bash
mkdir -p data/lab/zeek
cp ~/real-lab-001-zeek/conn.log data/lab/zeek/conn.log
```

## 9. Ground truth es Wazuh export

Exportald a vegleges event marker state-et:

```bash
python -m ml.src.lab_capture.event_marker export --output data/lab/lab_ground_truth.csv
```

Exportald a Wazuh alertokat a teljes UTC meresi idotartomanyra. A jelszo csak shell sessionben legyen jelen, fajlba ne keruljon.

```bash
make wazuh-export-opensearch \
  WAZUH_EXPORT_START=<YYYY-MM-DDTHH:MM:SSZ> \
  WAZUH_EXPORT_END=<YYYY-MM-DDTHH:MM:SSZ> \
  OPENSEARCH_PASSWORD=<local_secret>
```

## 10. Zeek-alapu meresi pipeline

```bash
make final-real-measurement-package-zeek
make real-measurement-provenance
make final-real-measurement-thesis-ready
make final-measurement-quality MEASUREMENT_QUALITY_THRESHOLDS=docs/measurement_quality_thresholds_real_lab_100.yaml
```

Ha barmelyik parancs hibat jelez, ne potold kezzel az inputot, es ne keszits kezzel gyartott vagy nem tenyleges lab futasbol szarmazo eredmenyt. A hibat a valos lab bemeneteken kell javitani vagy korlatkent dokumentalni.
