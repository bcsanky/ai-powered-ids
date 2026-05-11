# real-lab-001 100 esemenyes meresi terv

Ez a terv a `real-lab-001` valos labor session 100 darab, kezzel vegrehajtott es event markerrel rogzitett esemenyet irja le. A dokumentum elokeszitesre szolgal: nem tartalmaz valodi eredmenyt, nem general alertet, feature-t vagy metrikat.

## Biztonsagi keret

- Engedelyezett cel: `192.168.56.101`.
- Engedelyezett forras a laborban: `192.168.56.102`.
- Minden attack jellegu aktivitas csak `192.168.56.101` ellen tortenhet.
- Nyilvanos IP, ceges halozat, router, masik VM vagy internetes host nem szerepelhet celkent.
- A benign package update esemeny target-oldali benign muvelet; ha tavoli inditas kell, benign SSH session hasznalhato a `192.168.56.102 -> 192.168.56.101` iranyban, hogy az esemeny a host-only lab forgalommal igazithato legyen.

## Esemennyeloszlas

| Scenario | Label | Attack type | Darab |
| --- | --- | --- | ---: |
| `benign_ssh_login` | `benign` | ures | 30 |
| `benign_package_update` | `benign` | ures | 30 |
| `port_scan` | `attack` | `port_scan` | 8 |
| `ssh_failed_logins` | `attack` | `ssh_failed_logins` | 8 |
| `ssh_bruteforce` | `attack` | `ssh_bruteforce` | 8 |
| `file_integrity_change` | `attack` | `file_integrity_change` | 8 |
| `privilege_change` | `attack` | `privilege_change` | 8 |

Osszesen: 60 benign es 40 attack esemeny, 7 scenario, 5 attack scenario.

## Utemezesi szabalyok

- A 100 esemenyt 20 blokkban kell vegrehajtani, blokkonkent 5 esemeny.
- Egy esemeny sem fedhet at idoben masik esemenyt.
- Ket esemeny kozott ajanlott legalabb 60-90 masodperc szunet.
- A Zeek vagy Flow capture a legelso marker elott induljon, es csak az utolso marker utan alljon le.
- A Wazuh export idointervalluma fedje le a teljes merest: capture inditas elotti rovid pufferrel kezdodjon, es az utolso event utan rovid pufferrel zarodjon.
- A start/end markerek idejet az operatori logban is rogziteni kell.

## 20 blokkos vegrehajtasi terv

| Blokk | E1 | E2 | E3 | E4 | E5 |
| --- | --- | --- | --- | --- | --- |
| 01 | `BSSH-001` benign_ssh_login | `BSSH-002` benign_ssh_login | `BPKG-001` benign_package_update | `APORT-001` port_scan | `AFAIL-001` ssh_failed_logins |
| 02 | `BPKG-002` benign_package_update | `BPKG-003` benign_package_update | `BSSH-003` benign_ssh_login | `ABRUTE-001` ssh_bruteforce | `AFIM-001` file_integrity_change |
| 03 | `BSSH-004` benign_ssh_login | `BSSH-005` benign_ssh_login | `BPKG-004` benign_package_update | `APRIV-001` privilege_change | `APORT-002` port_scan |
| 04 | `BPKG-005` benign_package_update | `BPKG-006` benign_package_update | `BSSH-006` benign_ssh_login | `AFAIL-002` ssh_failed_logins | `ABRUTE-002` ssh_bruteforce |
| 05 | `BSSH-007` benign_ssh_login | `BSSH-008` benign_ssh_login | `BPKG-007` benign_package_update | `AFIM-002` file_integrity_change | `APRIV-002` privilege_change |
| 06 | `BPKG-008` benign_package_update | `BPKG-009` benign_package_update | `BSSH-009` benign_ssh_login | `APORT-003` port_scan | `AFAIL-003` ssh_failed_logins |
| 07 | `BSSH-010` benign_ssh_login | `BSSH-011` benign_ssh_login | `BPKG-010` benign_package_update | `ABRUTE-003` ssh_bruteforce | `AFIM-003` file_integrity_change |
| 08 | `BPKG-011` benign_package_update | `BPKG-012` benign_package_update | `BSSH-012` benign_ssh_login | `APRIV-003` privilege_change | `APORT-004` port_scan |
| 09 | `BSSH-013` benign_ssh_login | `BSSH-014` benign_ssh_login | `BPKG-013` benign_package_update | `AFAIL-004` ssh_failed_logins | `ABRUTE-004` ssh_bruteforce |
| 10 | `BPKG-014` benign_package_update | `BPKG-015` benign_package_update | `BSSH-015` benign_ssh_login | `AFIM-004` file_integrity_change | `APRIV-004` privilege_change |
| 11 | `BSSH-016` benign_ssh_login | `BSSH-017` benign_ssh_login | `BPKG-016` benign_package_update | `APORT-005` port_scan | `AFAIL-005` ssh_failed_logins |
| 12 | `BPKG-017` benign_package_update | `BPKG-018` benign_package_update | `BSSH-018` benign_ssh_login | `ABRUTE-005` ssh_bruteforce | `AFIM-005` file_integrity_change |
| 13 | `BSSH-019` benign_ssh_login | `BSSH-020` benign_ssh_login | `BPKG-019` benign_package_update | `APRIV-005` privilege_change | `APORT-006` port_scan |
| 14 | `BPKG-020` benign_package_update | `BPKG-021` benign_package_update | `BSSH-021` benign_ssh_login | `AFAIL-006` ssh_failed_logins | `ABRUTE-006` ssh_bruteforce |
| 15 | `BSSH-022` benign_ssh_login | `BSSH-023` benign_ssh_login | `BPKG-022` benign_package_update | `AFIM-006` file_integrity_change | `APRIV-006` privilege_change |
| 16 | `BPKG-023` benign_package_update | `BPKG-024` benign_package_update | `BSSH-024` benign_ssh_login | `APORT-007` port_scan | `AFAIL-007` ssh_failed_logins |
| 17 | `BSSH-025` benign_ssh_login | `BSSH-026` benign_ssh_login | `BPKG-025` benign_package_update | `ABRUTE-007` ssh_bruteforce | `AFIM-007` file_integrity_change |
| 18 | `BPKG-026` benign_package_update | `BPKG-027` benign_package_update | `BSSH-027` benign_ssh_login | `APRIV-007` privilege_change | `APORT-008` port_scan |
| 19 | `BSSH-028` benign_ssh_login | `BSSH-029` benign_ssh_login | `BPKG-028` benign_package_update | `AFAIL-008` ssh_failed_logins | `ABRUTE-008` ssh_bruteforce |
| 20 | `BPKG-029` benign_package_update | `BPKG-030` benign_package_update | `BSSH-030` benign_ssh_login | `AFIM-008` file_integrity_change | `APRIV-008` privilege_change |

Az attack scenariok ot blokkos ciklusban forognak: port scan + failed login, brute force + FIM, privilege change + port scan, failed login + brute force, FIM + privilege change. A ciklus negyszer ismetlodik, igy minden attack scenario pontosan 8 alkalommal szerepel.
