# real-lab-001 pilot event marker sablonok

Ez a pilot csak a marker, Wazuh es Zeek capture lanc gyors ellenorzesehez valo. Kulon state fajlt hasznal, es nem resze a vegleges 100 esemenyes ground truthnak. A sablon nem tartalmaz offenziv parancsot.

```bash
# PILOT-BSSH-001 benign SSH login
python -m ml.src.lab_capture.event_marker --state-file data/lab/pilot_session_events.json start --event-id PILOT-BSSH-001 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a kontrollalt pilot benign SSH muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker --state-file data/lab/pilot_session_events.json end --event-id PILOT-BSSH-001

# PILOT-APORT-001 port_scan
python -m ml.src.lab_capture.event_marker --state-file data/lab/pilot_session_events.json start --event-id PILOT-APORT-001 --scenario port_scan --label attack --attack-type port_scan --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a kontrollalt pilot lab muveletet csak a 192.168.56.101 cel ellen; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker --state-file data/lab/pilot_session_events.json end --event-id PILOT-APORT-001

# PILOT-AFIM-001 file_integrity_change
python -m ml.src.lab_capture.event_marker --state-file data/lab/pilot_session_events.json start --event-id PILOT-AFIM-001 --scenario file_integrity_change --label attack --attack-type file_integrity_change --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a kontrollalt pilot lab muveletet csak a 192.168.56.101 cel ellen; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker --state-file data/lab/pilot_session_events.json end --event-id PILOT-AFIM-001
```
