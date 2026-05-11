# real-lab-001 event marker parancssablonok

Az alabbi start/end parancsok a 100 esemenyes meres markerelesehez keszultek. A kommentelt helyen az operator csak az adott, kontrollalt lab muveletet hajtja vegre; a sablon nem tartalmaz offenziv parancsot.

```bash
# BSSH-001 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-001 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-001

# BSSH-002 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-002 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-002

# BPKG-001 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-001 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-001

# APORT-001 port_scan
python -m ml.src.lab_capture.event_marker start --event-id APORT-001 --scenario port_scan --label attack --attack-type port_scan --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id APORT-001

# AFAIL-001 ssh_failed_logins
python -m ml.src.lab_capture.event_marker start --event-id AFAIL-001 --scenario ssh_failed_logins --label attack --attack-type ssh_failed_logins --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id AFAIL-001

# BPKG-002 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-002 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-002

# BPKG-003 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-003 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-003

# BSSH-003 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-003 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-003

# ABRUTE-001 ssh_bruteforce
python -m ml.src.lab_capture.event_marker start --event-id ABRUTE-001 --scenario ssh_bruteforce --label attack --attack-type ssh_bruteforce --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id ABRUTE-001

# AFIM-001 file_integrity_change
python -m ml.src.lab_capture.event_marker start --event-id AFIM-001 --scenario file_integrity_change --label attack --attack-type file_integrity_change --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id AFIM-001

# BSSH-004 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-004 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-004

# BSSH-005 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-005 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-005

# BPKG-004 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-004 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-004

# APRIV-001 privilege_change
python -m ml.src.lab_capture.event_marker start --event-id APRIV-001 --scenario privilege_change --label attack --attack-type privilege_change --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id APRIV-001

# APORT-002 port_scan
python -m ml.src.lab_capture.event_marker start --event-id APORT-002 --scenario port_scan --label attack --attack-type port_scan --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id APORT-002

# BPKG-005 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-005 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-005

# BPKG-006 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-006 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-006

# BSSH-006 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-006 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-006

# AFAIL-002 ssh_failed_logins
python -m ml.src.lab_capture.event_marker start --event-id AFAIL-002 --scenario ssh_failed_logins --label attack --attack-type ssh_failed_logins --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id AFAIL-002

# ABRUTE-002 ssh_bruteforce
python -m ml.src.lab_capture.event_marker start --event-id ABRUTE-002 --scenario ssh_bruteforce --label attack --attack-type ssh_bruteforce --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id ABRUTE-002

# BSSH-007 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-007 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-007

# BSSH-008 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-008 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-008

# BPKG-007 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-007 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-007

# AFIM-002 file_integrity_change
python -m ml.src.lab_capture.event_marker start --event-id AFIM-002 --scenario file_integrity_change --label attack --attack-type file_integrity_change --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id AFIM-002

# APRIV-002 privilege_change
python -m ml.src.lab_capture.event_marker start --event-id APRIV-002 --scenario privilege_change --label attack --attack-type privilege_change --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id APRIV-002

# BPKG-008 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-008 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-008

# BPKG-009 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-009 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-009

# BSSH-009 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-009 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-009

# APORT-003 port_scan
python -m ml.src.lab_capture.event_marker start --event-id APORT-003 --scenario port_scan --label attack --attack-type port_scan --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id APORT-003

# AFAIL-003 ssh_failed_logins
python -m ml.src.lab_capture.event_marker start --event-id AFAIL-003 --scenario ssh_failed_logins --label attack --attack-type ssh_failed_logins --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id AFAIL-003

# BSSH-010 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-010 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-010

# BSSH-011 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-011 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-011

# BPKG-010 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-010 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-010

# ABRUTE-003 ssh_bruteforce
python -m ml.src.lab_capture.event_marker start --event-id ABRUTE-003 --scenario ssh_bruteforce --label attack --attack-type ssh_bruteforce --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id ABRUTE-003

# AFIM-003 file_integrity_change
python -m ml.src.lab_capture.event_marker start --event-id AFIM-003 --scenario file_integrity_change --label attack --attack-type file_integrity_change --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id AFIM-003

# BPKG-011 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-011 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-011

# BPKG-012 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-012 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-012

# BSSH-012 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-012 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-012

# APRIV-003 privilege_change
python -m ml.src.lab_capture.event_marker start --event-id APRIV-003 --scenario privilege_change --label attack --attack-type privilege_change --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id APRIV-003

# APORT-004 port_scan
python -m ml.src.lab_capture.event_marker start --event-id APORT-004 --scenario port_scan --label attack --attack-type port_scan --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id APORT-004

# BSSH-013 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-013 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-013

# BSSH-014 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-014 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-014

# BPKG-013 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-013 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-013

# AFAIL-004 ssh_failed_logins
python -m ml.src.lab_capture.event_marker start --event-id AFAIL-004 --scenario ssh_failed_logins --label attack --attack-type ssh_failed_logins --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id AFAIL-004

# ABRUTE-004 ssh_bruteforce
python -m ml.src.lab_capture.event_marker start --event-id ABRUTE-004 --scenario ssh_bruteforce --label attack --attack-type ssh_bruteforce --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id ABRUTE-004

# BPKG-014 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-014 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-014

# BPKG-015 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-015 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-015

# BSSH-015 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-015 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-015

# AFIM-004 file_integrity_change
python -m ml.src.lab_capture.event_marker start --event-id AFIM-004 --scenario file_integrity_change --label attack --attack-type file_integrity_change --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id AFIM-004

# APRIV-004 privilege_change
python -m ml.src.lab_capture.event_marker start --event-id APRIV-004 --scenario privilege_change --label attack --attack-type privilege_change --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id APRIV-004

# BSSH-016 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-016 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-016

# BSSH-017 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-017 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-017

# BPKG-016 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-016 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-016

# APORT-005 port_scan
python -m ml.src.lab_capture.event_marker start --event-id APORT-005 --scenario port_scan --label attack --attack-type port_scan --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id APORT-005

# AFAIL-005 ssh_failed_logins
python -m ml.src.lab_capture.event_marker start --event-id AFAIL-005 --scenario ssh_failed_logins --label attack --attack-type ssh_failed_logins --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id AFAIL-005

# BPKG-017 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-017 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-017

# BPKG-018 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-018 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-018

# BSSH-018 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-018 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-018

# ABRUTE-005 ssh_bruteforce
python -m ml.src.lab_capture.event_marker start --event-id ABRUTE-005 --scenario ssh_bruteforce --label attack --attack-type ssh_bruteforce --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id ABRUTE-005

# AFIM-005 file_integrity_change
python -m ml.src.lab_capture.event_marker start --event-id AFIM-005 --scenario file_integrity_change --label attack --attack-type file_integrity_change --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id AFIM-005

# BSSH-019 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-019 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-019

# BSSH-020 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-020 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-020

# BPKG-019 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-019 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-019

# APRIV-005 privilege_change
python -m ml.src.lab_capture.event_marker start --event-id APRIV-005 --scenario privilege_change --label attack --attack-type privilege_change --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id APRIV-005

# APORT-006 port_scan
python -m ml.src.lab_capture.event_marker start --event-id APORT-006 --scenario port_scan --label attack --attack-type port_scan --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id APORT-006

# BPKG-020 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-020 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-020

# BPKG-021 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-021 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-021

# BSSH-021 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-021 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-021

# AFAIL-006 ssh_failed_logins
python -m ml.src.lab_capture.event_marker start --event-id AFAIL-006 --scenario ssh_failed_logins --label attack --attack-type ssh_failed_logins --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id AFAIL-006

# ABRUTE-006 ssh_bruteforce
python -m ml.src.lab_capture.event_marker start --event-id ABRUTE-006 --scenario ssh_bruteforce --label attack --attack-type ssh_bruteforce --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id ABRUTE-006

# BSSH-022 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-022 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-022

# BSSH-023 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-023 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-023

# BPKG-022 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-022 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-022

# AFIM-006 file_integrity_change
python -m ml.src.lab_capture.event_marker start --event-id AFIM-006 --scenario file_integrity_change --label attack --attack-type file_integrity_change --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id AFIM-006

# APRIV-006 privilege_change
python -m ml.src.lab_capture.event_marker start --event-id APRIV-006 --scenario privilege_change --label attack --attack-type privilege_change --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id APRIV-006

# BPKG-023 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-023 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-023

# BPKG-024 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-024 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-024

# BSSH-024 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-024 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-024

# APORT-007 port_scan
python -m ml.src.lab_capture.event_marker start --event-id APORT-007 --scenario port_scan --label attack --attack-type port_scan --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id APORT-007

# AFAIL-007 ssh_failed_logins
python -m ml.src.lab_capture.event_marker start --event-id AFAIL-007 --scenario ssh_failed_logins --label attack --attack-type ssh_failed_logins --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id AFAIL-007

# BSSH-025 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-025 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-025

# BSSH-026 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-026 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-026

# BPKG-025 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-025 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-025

# ABRUTE-007 ssh_bruteforce
python -m ml.src.lab_capture.event_marker start --event-id ABRUTE-007 --scenario ssh_bruteforce --label attack --attack-type ssh_bruteforce --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id ABRUTE-007

# AFIM-007 file_integrity_change
python -m ml.src.lab_capture.event_marker start --event-id AFIM-007 --scenario file_integrity_change --label attack --attack-type file_integrity_change --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id AFIM-007

# BPKG-026 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-026 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-026

# BPKG-027 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-027 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-027

# BSSH-027 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-027 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-027

# APRIV-007 privilege_change
python -m ml.src.lab_capture.event_marker start --event-id APRIV-007 --scenario privilege_change --label attack --attack-type privilege_change --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id APRIV-007

# APORT-008 port_scan
python -m ml.src.lab_capture.event_marker start --event-id APORT-008 --scenario port_scan --label attack --attack-type port_scan --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id APORT-008

# BSSH-028 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-028 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-028

# BSSH-029 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-029 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-029

# BPKG-028 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-028 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-028

# AFAIL-008 ssh_failed_logins
python -m ml.src.lab_capture.event_marker start --event-id AFAIL-008 --scenario ssh_failed_logins --label attack --attack-type ssh_failed_logins --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id AFAIL-008

# ABRUTE-008 ssh_bruteforce
python -m ml.src.lab_capture.event_marker start --event-id ABRUTE-008 --scenario ssh_bruteforce --label attack --attack-type ssh_bruteforce --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id ABRUTE-008

# BPKG-029 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-029 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-029

# BPKG-030 benign_package_update
python -m ml.src.lab_capture.event_marker start --event-id BPKG-030 --scenario benign_package_update --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BPKG-030

# BSSH-030 benign_ssh_login
python -m ml.src.lab_capture.event_marker start --event-id BSSH-030 --scenario benign_ssh_login --label benign --attack-type "" --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id BSSH-030

# AFIM-008 file_integrity_change
python -m ml.src.lab_capture.event_marker start --event-id AFIM-008 --scenario file_integrity_change --label attack --attack-type file_integrity_change --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id AFIM-008

# APRIV-008 privilege_change
python -m ml.src.lab_capture.event_marker start --event-id APRIV-008 --scenario privilege_change --label attack --attack-type privilege_change --source-ip 192.168.56.102 --target-ip 192.168.56.101
# Operator: hajtsa vegre a tervezett labor muveletet; parancsot ne rogzitsetek ebbe a sablonba.
python -m ml.src.lab_capture.event_marker end --event-id APRIV-008
```
