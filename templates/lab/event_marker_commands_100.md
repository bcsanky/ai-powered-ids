# real-lab-001 végleges 100 eseményes event marker parancssablon

Ez a fájl a `real-lab-001` végleges, 100 eseményes mérésének kézi markereléséhez készült. Nem tartalmaz tényleges támadó parancsot, nem generál mérési eredményt, és nem helyettesíti a futás közbeni operátori ellenőrzést.

Biztonsági scope:

- Engedélyezett cél: `192.168.56.101`.
- Attacker/source: `192.168.56.102`.
- Tiltott célok: public IP-k, céges hálózatok, router, más VM-ek és internetes hostok.
- Minden lab művelet kizárólag a `192.168.56.101` célgépen hajtható végre.

Futási előfeltételek:

- A Zeek/tcpdump capture már fusson az első event előtt.
- A Wazuh export időtartománya fedje le a teljes mérést.
- Az események nem fedhetik át egymást időben.
- Két esemény között hagyj 60-90 másodperc szünetet.
- A végleges marker state fájl: `data/lab/session_events.json`.

```bash
# 001. esemény: BSSH-001 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-001 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-001
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 002. esemény: BSSH-002 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-002 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-002
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 003. esemény: BPKG-001 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-001 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-001
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 004. esemény: APORT-001 port_scan
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id APORT-001 --scenario port_scan --label attack --attack-type "port_scan" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id APORT-001
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 005. esemény: AFAIL-001 ssh_failed_logins
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id AFAIL-001 --scenario ssh_failed_logins --label attack --attack-type "ssh_failed_logins" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id AFAIL-001
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 006. esemény: BPKG-002 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-002 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-002
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 007. esemény: BPKG-003 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-003 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-003
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 008. esemény: BSSH-003 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-003 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-003
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 009. esemény: ABRUTE-001 ssh_bruteforce
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id ABRUTE-001 --scenario ssh_bruteforce --label attack --attack-type "ssh_bruteforce" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id ABRUTE-001
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 010. esemény: AFIM-001 file_integrity_change
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id AFIM-001 --scenario file_integrity_change --label attack --attack-type "file_integrity_change" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id AFIM-001
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 011. esemény: BSSH-004 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-004 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-004
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 012. esemény: BSSH-005 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-005 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-005
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 013. esemény: BPKG-004 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-004 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-004
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 014. esemény: APRIV-001 privilege_change
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id APRIV-001 --scenario privilege_change --label attack --attack-type "privilege_change" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id APRIV-001
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 015. esemény: APORT-002 port_scan
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id APORT-002 --scenario port_scan --label attack --attack-type "port_scan" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id APORT-002
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 016. esemény: BPKG-005 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-005 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-005
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 017. esemény: BPKG-006 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-006 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-006
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 018. esemény: BSSH-006 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-006 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-006
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 019. esemény: AFAIL-002 ssh_failed_logins
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id AFAIL-002 --scenario ssh_failed_logins --label attack --attack-type "ssh_failed_logins" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id AFAIL-002
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 020. esemény: ABRUTE-002 ssh_bruteforce
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id ABRUTE-002 --scenario ssh_bruteforce --label attack --attack-type "ssh_bruteforce" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id ABRUTE-002
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 021. esemény: BSSH-007 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-007 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-007
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 022. esemény: BSSH-008 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-008 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-008
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 023. esemény: BPKG-007 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-007 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-007
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 024. esemény: AFIM-002 file_integrity_change
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id AFIM-002 --scenario file_integrity_change --label attack --attack-type "file_integrity_change" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id AFIM-002
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 025. esemény: APRIV-002 privilege_change
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id APRIV-002 --scenario privilege_change --label attack --attack-type "privilege_change" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id APRIV-002
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 026. esemény: BPKG-008 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-008 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-008
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 027. esemény: BPKG-009 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-009 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-009
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 028. esemény: BSSH-009 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-009 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-009
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 029. esemény: APORT-003 port_scan
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id APORT-003 --scenario port_scan --label attack --attack-type "port_scan" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id APORT-003
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 030. esemény: AFAIL-003 ssh_failed_logins
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id AFAIL-003 --scenario ssh_failed_logins --label attack --attack-type "ssh_failed_logins" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id AFAIL-003
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 031. esemény: BSSH-010 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-010 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-010
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 032. esemény: BSSH-011 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-011 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-011
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 033. esemény: BPKG-010 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-010 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-010
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 034. esemény: ABRUTE-003 ssh_bruteforce
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id ABRUTE-003 --scenario ssh_bruteforce --label attack --attack-type "ssh_bruteforce" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id ABRUTE-003
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 035. esemény: AFIM-003 file_integrity_change
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id AFIM-003 --scenario file_integrity_change --label attack --attack-type "file_integrity_change" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id AFIM-003
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 036. esemény: BPKG-011 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-011 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-011
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 037. esemény: BPKG-012 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-012 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-012
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 038. esemény: BSSH-012 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-012 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-012
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 039. esemény: APRIV-003 privilege_change
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id APRIV-003 --scenario privilege_change --label attack --attack-type "privilege_change" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id APRIV-003
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 040. esemény: APORT-004 port_scan
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id APORT-004 --scenario port_scan --label attack --attack-type "port_scan" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id APORT-004
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 041. esemény: BSSH-013 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-013 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-013
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 042. esemény: BSSH-014 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-014 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-014
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 043. esemény: BPKG-013 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-013 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-013
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 044. esemény: AFAIL-004 ssh_failed_logins
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id AFAIL-004 --scenario ssh_failed_logins --label attack --attack-type "ssh_failed_logins" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id AFAIL-004
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 045. esemény: ABRUTE-004 ssh_bruteforce
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id ABRUTE-004 --scenario ssh_bruteforce --label attack --attack-type "ssh_bruteforce" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id ABRUTE-004
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 046. esemény: BPKG-014 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-014 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-014
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 047. esemény: BPKG-015 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-015 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-015
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 048. esemény: BSSH-015 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-015 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-015
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 049. esemény: AFIM-004 file_integrity_change
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id AFIM-004 --scenario file_integrity_change --label attack --attack-type "file_integrity_change" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id AFIM-004
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 050. esemény: APRIV-004 privilege_change
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id APRIV-004 --scenario privilege_change --label attack --attack-type "privilege_change" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id APRIV-004
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 051. esemény: BSSH-016 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-016 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-016
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 052. esemény: BSSH-017 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-017 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-017
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 053. esemény: BPKG-016 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-016 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-016
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 054. esemény: APORT-005 port_scan
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id APORT-005 --scenario port_scan --label attack --attack-type "port_scan" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id APORT-005
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 055. esemény: AFAIL-005 ssh_failed_logins
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id AFAIL-005 --scenario ssh_failed_logins --label attack --attack-type "ssh_failed_logins" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id AFAIL-005
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 056. esemény: BPKG-017 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-017 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-017
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 057. esemény: BPKG-018 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-018 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-018
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 058. esemény: BSSH-018 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-018 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-018
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 059. esemény: ABRUTE-005 ssh_bruteforce
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id ABRUTE-005 --scenario ssh_bruteforce --label attack --attack-type "ssh_bruteforce" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id ABRUTE-005
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 060. esemény: AFIM-005 file_integrity_change
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id AFIM-005 --scenario file_integrity_change --label attack --attack-type "file_integrity_change" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id AFIM-005
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 061. esemény: BSSH-019 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-019 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-019
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 062. esemény: BSSH-020 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-020 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-020
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 063. esemény: BPKG-019 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-019 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-019
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 064. esemény: APRIV-005 privilege_change
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id APRIV-005 --scenario privilege_change --label attack --attack-type "privilege_change" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id APRIV-005
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 065. esemény: APORT-006 port_scan
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id APORT-006 --scenario port_scan --label attack --attack-type "port_scan" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id APORT-006
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 066. esemény: BPKG-020 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-020 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-020
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 067. esemény: BPKG-021 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-021 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-021
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 068. esemény: BSSH-021 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-021 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-021
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 069. esemény: AFAIL-006 ssh_failed_logins
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id AFAIL-006 --scenario ssh_failed_logins --label attack --attack-type "ssh_failed_logins" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id AFAIL-006
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 070. esemény: ABRUTE-006 ssh_bruteforce
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id ABRUTE-006 --scenario ssh_bruteforce --label attack --attack-type "ssh_bruteforce" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id ABRUTE-006
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 071. esemény: BSSH-022 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-022 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-022
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 072. esemény: BSSH-023 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-023 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-023
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 073. esemény: BPKG-022 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-022 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-022
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 074. esemény: AFIM-006 file_integrity_change
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id AFIM-006 --scenario file_integrity_change --label attack --attack-type "file_integrity_change" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id AFIM-006
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 075. esemény: APRIV-006 privilege_change
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id APRIV-006 --scenario privilege_change --label attack --attack-type "privilege_change" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id APRIV-006
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 076. esemény: BPKG-023 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-023 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-023
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 077. esemény: BPKG-024 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-024 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-024
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 078. esemény: BSSH-024 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-024 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-024
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 079. esemény: APORT-007 port_scan
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id APORT-007 --scenario port_scan --label attack --attack-type "port_scan" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id APORT-007
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 080. esemény: AFAIL-007 ssh_failed_logins
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id AFAIL-007 --scenario ssh_failed_logins --label attack --attack-type "ssh_failed_logins" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id AFAIL-007
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 081. esemény: BSSH-025 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-025 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-025
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 082. esemény: BSSH-026 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-026 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-026
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 083. esemény: BPKG-025 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-025 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-025
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 084. esemény: ABRUTE-007 ssh_bruteforce
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id ABRUTE-007 --scenario ssh_bruteforce --label attack --attack-type "ssh_bruteforce" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id ABRUTE-007
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 085. esemény: AFIM-007 file_integrity_change
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id AFIM-007 --scenario file_integrity_change --label attack --attack-type "file_integrity_change" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id AFIM-007
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 086. esemény: BPKG-026 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-026 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-026
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 087. esemény: BPKG-027 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-027 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-027
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 088. esemény: BSSH-027 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-027 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-027
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 089. esemény: APRIV-007 privilege_change
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id APRIV-007 --scenario privilege_change --label attack --attack-type "privilege_change" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id APRIV-007
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 090. esemény: APORT-008 port_scan
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id APORT-008 --scenario port_scan --label attack --attack-type "port_scan" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id APORT-008
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 091. esemény: BSSH-028 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-028 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-028
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 092. esemény: BSSH-029 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-029 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-029
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 093. esemény: BPKG-028 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-028 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-028
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 094. esemény: AFAIL-008 ssh_failed_logins
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id AFAIL-008 --scenario ssh_failed_logins --label attack --attack-type "ssh_failed_logins" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id AFAIL-008
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 095. esemény: ABRUTE-008 ssh_bruteforce
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id ABRUTE-008 --scenario ssh_bruteforce --label attack --attack-type "ssh_bruteforce" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id ABRUTE-008
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 096. esemény: BPKG-029 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-029 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-029
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 097. esemény: BPKG-030 benign_package_update
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BPKG-030 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BPKG-030
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 098. esemény: BSSH-030 benign_ssh_login
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id BSSH-030 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id BSSH-030
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 099. esemény: AFIM-008 file_integrity_change
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id AFIM-008 --scenario file_integrity_change --label attack --attack-type "file_integrity_change" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id AFIM-008
Futtatás helye: WSL, a repository gyökérkönyvtára.
sleep 60

# 100. esemény: APRIV-008 privilege_change
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json start --event-id APRIV-008 --scenario privilege_change --label attack --attack-type "privilege_change" --source-ip 192.168.56.102 --target-ip 192.168.56.101
Futtatás helye: Kali VM (192.168.56.102), kizárólag a Target VM (192.168.56.101) ellen.
# Itt hajtsd végre a tervezett lab műveletet kizárólag a 192.168.56.101 célgépen.
Futtatás helye: WSL, a repository gyökérkönyvtára.
python3 -m ml.src.lab_capture.event_marker --state-file data/lab/session_events.json end --event-id APRIV-008

```
