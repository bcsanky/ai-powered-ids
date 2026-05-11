# real-lab-001 missing-flow rerun marker sablon

Ez a sablon a 22 olyan `real-lab-001` esemény célzott újrafuttatásához készült, amelyekhez a Zeek feature build nem talált illeszkedő flow-t. A rerun külön marker state fájlt használ, és nem írja felül a végleges 100 eseményes state-et.

Biztonsági scope:

- Engedélyezett cél: `192.168.56.101`.
- Attacker/source: `192.168.56.102`.
- Tiltott célok: public IP-k, céges hálózatok, router, más VM-ek és internetes hostok.
- Minden tényleges lab művelet kizárólag a `192.168.56.101` célgép ellen történhet.

Fontos capture szabály: minden eseményablakon belül új SSH/network kapcsolatot kell nyitni. Ne használj már nyitott SSH sessiont, mert abból nem feltétlenül keletkezik új Zeek flow az adott event window-ban.

A Kali VM-en futtatandó lab műveletek konkrét parancsként szerepelnek. Minden parancs új hálózati kapcsolatot nyit a `192.168.56.101` célgép felé, és nem tartalmaz jelszót vagy titkot.

```bash
# ABRUTE-006 ssh_bruteforce
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json start --event-id ABRUTE-006 --scenario ssh_bruteforce --label attack --attack-type "ssh_bruteforce" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
for p in real-lab-001-rerun-brute-{01..08}; do sshpass -p "$p" ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 -o NumberOfPasswordPrompts=1 real_lab_invalid@192.168.56.101 'true' || true; sleep 1; done
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json end --event-id ABRUTE-006

# APORT-007 port_scan
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json start --event-id APORT-007 --scenario port_scan --label attack --attack-type "port_scan" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
nmap -Pn -n --max-retries 1 --host-timeout 30s -p 22,80,443,1514,1515 192.168.56.101
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json end --event-id APORT-007

# APORT-008 port_scan
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json start --event-id APORT-008 --scenario port_scan --label attack --attack-type "port_scan" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
nmap -Pn -n --max-retries 1 --host-timeout 30s -p 22,80,443,1514,1515 192.168.56.101
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json end --event-id APORT-008

# APRIV-006 privilege_change
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json start --event-id APRIV-006 --scenario privilege_change --label attack --attack-type "privilege_change" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 admin@192.168.56.101 'sudo touch /etc/real_lab_001_privilege_marker && printf "APRIV-006 rerun privilege_change %s\n" "$(date -u +%FT%TZ)" | sudo tee -a /etc/real_lab_001_privilege_marker >/dev/null && sudo chmod 600 /etc/real_lab_001_privilege_marker && sudo chmod 640 /etc/real_lab_001_privilege_marker'
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json end --event-id APRIV-006

# APRIV-008 privilege_change
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json start --event-id APRIV-008 --scenario privilege_change --label attack --attack-type "privilege_change" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
ssh -tt -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 admin@192.168.56.101 'sudo touch /etc/real_lab_001_privilege_marker && printf "APRIV-008 rerun privilege_change %s\n" "$(date -u +%FT%TZ)" | sudo tee -a /etc/real_lab_001_privilege_marker >/dev/null && sudo chmod 600 /etc/real_lab_001_privilege_marker && sudo chmod 640 /etc/real_lab_001_privilege_marker'
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json end --event-id APRIV-008

# BPKG-014 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json start --event-id BPKG-014 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
ssh -tt -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 admin@192.168.56.101 'if command -v apt-get >/dev/null 2>&1; then sudo apt-get -s upgrade >/tmp/BPKG-014_rerun_package_update_simulation.log; elif command -v dnf >/dev/null 2>&1; then sudo dnf -q check-update --cacheonly >/tmp/BPKG-014_rerun_package_update_cache_check.log || true; else printf "BPKG-014 rerun no_supported_package_manager %s\n" "$(date -u +%FT%TZ)" >/tmp/BPKG-014_rerun_package_update_note.log; fi'
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json end --event-id BPKG-014

# BPKG-016 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json start --event-id BPKG-016 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
ssh -tt -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 admin@192.168.56.101 'if command -v apt-get >/dev/null 2>&1; then sudo apt-get -s upgrade >/tmp/BPKG-016_rerun_package_update_simulation.log; elif command -v dnf >/dev/null 2>&1; then sudo dnf -q check-update --cacheonly >/tmp/BPKG-016_rerun_package_update_cache_check.log || true; else printf "BPKG-016 rerun no_supported_package_manager %s\n" "$(date -u +%FT%TZ)" >/tmp/BPKG-016_rerun_package_update_note.log; fi'
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json end --event-id BPKG-016

# BPKG-021 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json start --event-id BPKG-021 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
ssh -tt -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 admin@192.168.56.101 'if command -v apt-get >/dev/null 2>&1; then sudo apt-get -s upgrade >/tmp/BPKG-021_rerun_package_update_simulation.log; elif command -v dnf >/dev/null 2>&1; then sudo dnf -q check-update --cacheonly >/tmp/BPKG-021_rerun_package_update_cache_check.log || true; else printf "BPKG-021 rerun no_supported_package_manager %s\n" "$(date -u +%FT%TZ)" >/tmp/BPKG-021_rerun_package_update_note.log; fi'
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json end --event-id BPKG-021

# BPKG-025 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json start --event-id BPKG-025 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
ssh -tt -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 admin@192.168.56.101 'if command -v apt-get >/dev/null 2>&1; then sudo apt-get -s upgrade >/tmp/BPKG-025_rerun_package_update_simulation.log; elif command -v dnf >/dev/null 2>&1; then sudo dnf -q check-update --cacheonly >/tmp/BPKG-025_rerun_package_update_cache_check.log || true; else printf "BPKG-025 rerun no_supported_package_manager %s\n" "$(date -u +%FT%TZ)" >/tmp/BPKG-025_rerun_package_update_note.log; fi'
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json end --event-id BPKG-025

# BPKG-026 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json start --event-id BPKG-026 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
ssh -tt -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 admin@192.168.56.101 'if command -v apt-get >/dev/null 2>&1; then sudo apt-get -s upgrade >/tmp/BPKG-026_rerun_package_update_simulation.log; elif command -v dnf >/dev/null 2>&1; then sudo dnf -q check-update --cacheonly >/tmp/BPKG-026_rerun_package_update_cache_check.log || true; else printf "BPKG-026 rerun no_supported_package_manager %s\n" "$(date -u +%FT%TZ)" >/tmp/BPKG-026_rerun_package_update_note.log; fi'
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json end --event-id BPKG-026

# BPKG-027 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json start --event-id BPKG-027 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
ssh -tt -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 admin@192.168.56.101 'if command -v apt-get >/dev/null 2>&1; then sudo apt-get -s upgrade >/tmp/BPKG-027_rerun_package_update_simulation.log; elif command -v dnf >/dev/null 2>&1; then sudo dnf -q check-update --cacheonly >/tmp/BPKG-027_rerun_package_update_cache_check.log || true; else printf "BPKG-027 rerun no_supported_package_manager %s\n" "$(date -u +%FT%TZ)" >/tmp/BPKG-027_rerun_package_update_note.log; fi'
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json end --event-id BPKG-027

# BPKG-028 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json start --event-id BPKG-028 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
ssh -tt -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 admin@192.168.56.101 'if command -v apt-get >/dev/null 2>&1; then sudo apt-get -s upgrade >/tmp/BPKG-028_rerun_package_update_simulation.log; elif command -v dnf >/dev/null 2>&1; then sudo dnf -q check-update --cacheonly >/tmp/BPKG-028_rerun_package_update_cache_check.log || true; else printf "BPKG-028 rerun no_supported_package_manager %s\n" "$(date -u +%FT%TZ)" >/tmp/BPKG-028_rerun_package_update_note.log; fi'
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json end --event-id BPKG-028

# BPKG-029 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json start --event-id BPKG-029 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
ssh -tt -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 admin@192.168.56.101 'if command -v apt-get >/dev/null 2>&1; then sudo apt-get -s upgrade >/tmp/BPKG-029_rerun_package_update_simulation.log; elif command -v dnf >/dev/null 2>&1; then sudo dnf -q check-update --cacheonly >/tmp/BPKG-029_rerun_package_update_cache_check.log || true; else printf "BPKG-029 rerun no_supported_package_manager %s\n" "$(date -u +%FT%TZ)" >/tmp/BPKG-029_rerun_package_update_note.log; fi'
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json end --event-id BPKG-029

# BPKG-030 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json start --event-id BPKG-030 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
ssh -tt -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 admin@192.168.56.101 'if command -v apt-get >/dev/null 2>&1; then sudo apt-get -s upgrade >/tmp/BPKG-030_rerun_package_update_simulation.log; elif command -v dnf >/dev/null 2>&1; then sudo dnf -q check-update --cacheonly >/tmp/BPKG-030_rerun_package_update_cache_check.log || true; else printf "BPKG-030 rerun no_supported_package_manager %s\n" "$(date -u +%FT%TZ)" >/tmp/BPKG-030_rerun_package_update_note.log; fi'
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json end --event-id BPKG-030

# BSSH-015 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json start --event-id BSSH-015 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 admin@192.168.56.101 'printf "BSSH-015 rerun benign_ssh_login %s\n" "$(date -u +%FT%TZ)" >> /tmp/real_lab_001_rerun_benign_ssh_login.log'
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json end --event-id BSSH-015

# BSSH-024 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json start --event-id BSSH-024 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 admin@192.168.56.101 'printf "BSSH-024 rerun benign_ssh_login %s\n" "$(date -u +%FT%TZ)" >> /tmp/real_lab_001_rerun_benign_ssh_login.log'
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json end --event-id BSSH-024

# BSSH-025 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json start --event-id BSSH-025 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 admin@192.168.56.101 'printf "BSSH-025 rerun benign_ssh_login %s\n" "$(date -u +%FT%TZ)" >> /tmp/real_lab_001_rerun_benign_ssh_login.log'
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json end --event-id BSSH-025

# BSSH-026 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json start --event-id BSSH-026 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 admin@192.168.56.101 'printf "BSSH-026 rerun benign_ssh_login %s\n" "$(date -u +%FT%TZ)" >> /tmp/real_lab_001_rerun_benign_ssh_login.log'
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json end --event-id BSSH-026

# BSSH-027 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json start --event-id BSSH-027 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 admin@192.168.56.101 'printf "BSSH-027 rerun benign_ssh_login %s\n" "$(date -u +%FT%TZ)" >> /tmp/real_lab_001_rerun_benign_ssh_login.log'
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json end --event-id BSSH-027

# BSSH-028 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json start --event-id BSSH-028 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 admin@192.168.56.101 'printf "BSSH-028 rerun benign_ssh_login %s\n" "$(date -u +%FT%TZ)" >> /tmp/real_lab_001_rerun_benign_ssh_login.log'
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json end --event-id BSSH-028

# BSSH-029 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json start --event-id BSSH-029 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 admin@192.168.56.101 'printf "BSSH-029 rerun benign_ssh_login %s\n" "$(date -u +%FT%TZ)" >> /tmp/real_lab_001_rerun_benign_ssh_login.log'
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json end --event-id BSSH-029

# BSSH-030 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json start --event-id BSSH-030 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 admin@192.168.56.101 'printf "BSSH-030 rerun benign_ssh_login %s\n" "$(date -u +%FT%TZ)" >> /tmp/real_lab_001_rerun_benign_ssh_login.log'
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/rerun_missing_flow_events.json end --event-id BSSH-030
```
