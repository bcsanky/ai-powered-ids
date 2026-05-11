# real-lab-001 operatori parancsnaplo sablon

Ez a 100 soros sablon a kezi meres alatti adminisztrativ naplozashoz keszult. A `Executed action` mezobe csak rovid, utolagosan ertelmezheto muveletleiras keruljon; valodi tamado parancsot ne irj a sablonba.

| Event ID | Planned scenario | Start time | End time | Executed action | Expected Wazuh effect | Expected Zeek/Flow effect | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `BSSH-001` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `BSSH-002` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `BPKG-001` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `APORT-001` | `port_scan` |  |  | `<kitoltendo, parancs nelkul>` | Recon vagy scan jellegu Wazuh jelzes lehet. | Tobb rovid kapcsolat a cel IP fele. |  |
| `AFAIL-001` | `ssh_failed_logins` |  |  | `<kitoltendo, parancs nelkul>` | Sikertelen SSH login jelzes lehet. | SSH probalkozasok a cel IP fele. |  |
| `BPKG-002` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `BPKG-003` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `BSSH-003` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `ABRUTE-001` | `ssh_bruteforce` |  |  | `<kitoltendo, parancs nelkul>` | Tobb sikertelen SSH probalkozas vagy brute-force jelzes lehet. | Ismetelt SSH kapcsolatok a cel IP fele. |  |
| `AFIM-001` | `file_integrity_change` |  |  | `<kitoltendo, parancs nelkul>` | FIM valtozas jelzes varhato. | SSH menedzsment forgalom; host alerttel korrelalando. |  |
| `BSSH-004` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `BSSH-005` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `BPKG-004` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `APRIV-001` | `privilege_change` |  |  | `<kitoltendo, parancs nelkul>` | Jogosultsagvaltozas vagy sudo audit jelzes varhato. | SSH menedzsment forgalom; host alerttel korrelalando. |  |
| `APORT-002` | `port_scan` |  |  | `<kitoltendo, parancs nelkul>` | Recon vagy scan jellegu Wazuh jelzes lehet. | Tobb rovid kapcsolat a cel IP fele. |  |
| `BPKG-005` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `BPKG-006` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `BSSH-006` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `AFAIL-002` | `ssh_failed_logins` |  |  | `<kitoltendo, parancs nelkul>` | Sikertelen SSH login jelzes lehet. | SSH probalkozasok a cel IP fele. |  |
| `ABRUTE-002` | `ssh_bruteforce` |  |  | `<kitoltendo, parancs nelkul>` | Tobb sikertelen SSH probalkozas vagy brute-force jelzes lehet. | Ismetelt SSH kapcsolatok a cel IP fele. |  |
| `BSSH-007` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `BSSH-008` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `BPKG-007` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `AFIM-002` | `file_integrity_change` |  |  | `<kitoltendo, parancs nelkul>` | FIM valtozas jelzes varhato. | SSH menedzsment forgalom; host alerttel korrelalando. |  |
| `APRIV-002` | `privilege_change` |  |  | `<kitoltendo, parancs nelkul>` | Jogosultsagvaltozas vagy sudo audit jelzes varhato. | SSH menedzsment forgalom; host alerttel korrelalando. |  |
| `BPKG-008` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `BPKG-009` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `BSSH-009` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `APORT-003` | `port_scan` |  |  | `<kitoltendo, parancs nelkul>` | Recon vagy scan jellegu Wazuh jelzes lehet. | Tobb rovid kapcsolat a cel IP fele. |  |
| `AFAIL-003` | `ssh_failed_logins` |  |  | `<kitoltendo, parancs nelkul>` | Sikertelen SSH login jelzes lehet. | SSH probalkozasok a cel IP fele. |  |
| `BSSH-010` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `BSSH-011` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `BPKG-010` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `ABRUTE-003` | `ssh_bruteforce` |  |  | `<kitoltendo, parancs nelkul>` | Tobb sikertelen SSH probalkozas vagy brute-force jelzes lehet. | Ismetelt SSH kapcsolatok a cel IP fele. |  |
| `AFIM-003` | `file_integrity_change` |  |  | `<kitoltendo, parancs nelkul>` | FIM valtozas jelzes varhato. | SSH menedzsment forgalom; host alerttel korrelalando. |  |
| `BPKG-011` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `BPKG-012` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `BSSH-012` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `APRIV-003` | `privilege_change` |  |  | `<kitoltendo, parancs nelkul>` | Jogosultsagvaltozas vagy sudo audit jelzes varhato. | SSH menedzsment forgalom; host alerttel korrelalando. |  |
| `APORT-004` | `port_scan` |  |  | `<kitoltendo, parancs nelkul>` | Recon vagy scan jellegu Wazuh jelzes lehet. | Tobb rovid kapcsolat a cel IP fele. |  |
| `BSSH-013` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `BSSH-014` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `BPKG-013` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `AFAIL-004` | `ssh_failed_logins` |  |  | `<kitoltendo, parancs nelkul>` | Sikertelen SSH login jelzes lehet. | SSH probalkozasok a cel IP fele. |  |
| `ABRUTE-004` | `ssh_bruteforce` |  |  | `<kitoltendo, parancs nelkul>` | Tobb sikertelen SSH probalkozas vagy brute-force jelzes lehet. | Ismetelt SSH kapcsolatok a cel IP fele. |  |
| `BPKG-014` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `BPKG-015` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `BSSH-015` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `AFIM-004` | `file_integrity_change` |  |  | `<kitoltendo, parancs nelkul>` | FIM valtozas jelzes varhato. | SSH menedzsment forgalom; host alerttel korrelalando. |  |
| `APRIV-004` | `privilege_change` |  |  | `<kitoltendo, parancs nelkul>` | Jogosultsagvaltozas vagy sudo audit jelzes varhato. | SSH menedzsment forgalom; host alerttel korrelalando. |  |
| `BSSH-016` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `BSSH-017` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `BPKG-016` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `APORT-005` | `port_scan` |  |  | `<kitoltendo, parancs nelkul>` | Recon vagy scan jellegu Wazuh jelzes lehet. | Tobb rovid kapcsolat a cel IP fele. |  |
| `AFAIL-005` | `ssh_failed_logins` |  |  | `<kitoltendo, parancs nelkul>` | Sikertelen SSH login jelzes lehet. | SSH probalkozasok a cel IP fele. |  |
| `BPKG-017` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `BPKG-018` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `BSSH-018` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `ABRUTE-005` | `ssh_bruteforce` |  |  | `<kitoltendo, parancs nelkul>` | Tobb sikertelen SSH probalkozas vagy brute-force jelzes lehet. | Ismetelt SSH kapcsolatok a cel IP fele. |  |
| `AFIM-005` | `file_integrity_change` |  |  | `<kitoltendo, parancs nelkul>` | FIM valtozas jelzes varhato. | SSH menedzsment forgalom; host alerttel korrelalando. |  |
| `BSSH-019` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `BSSH-020` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `BPKG-019` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `APRIV-005` | `privilege_change` |  |  | `<kitoltendo, parancs nelkul>` | Jogosultsagvaltozas vagy sudo audit jelzes varhato. | SSH menedzsment forgalom; host alerttel korrelalando. |  |
| `APORT-006` | `port_scan` |  |  | `<kitoltendo, parancs nelkul>` | Recon vagy scan jellegu Wazuh jelzes lehet. | Tobb rovid kapcsolat a cel IP fele. |  |
| `BPKG-020` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `BPKG-021` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `BSSH-021` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `AFAIL-006` | `ssh_failed_logins` |  |  | `<kitoltendo, parancs nelkul>` | Sikertelen SSH login jelzes lehet. | SSH probalkozasok a cel IP fele. |  |
| `ABRUTE-006` | `ssh_bruteforce` |  |  | `<kitoltendo, parancs nelkul>` | Tobb sikertelen SSH probalkozas vagy brute-force jelzes lehet. | Ismetelt SSH kapcsolatok a cel IP fele. |  |
| `BSSH-022` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `BSSH-023` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `BPKG-022` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `AFIM-006` | `file_integrity_change` |  |  | `<kitoltendo, parancs nelkul>` | FIM valtozas jelzes varhato. | SSH menedzsment forgalom; host alerttel korrelalando. |  |
| `APRIV-006` | `privilege_change` |  |  | `<kitoltendo, parancs nelkul>` | Jogosultsagvaltozas vagy sudo audit jelzes varhato. | SSH menedzsment forgalom; host alerttel korrelalando. |  |
| `BPKG-023` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `BPKG-024` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `BSSH-024` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `APORT-007` | `port_scan` |  |  | `<kitoltendo, parancs nelkul>` | Recon vagy scan jellegu Wazuh jelzes lehet. | Tobb rovid kapcsolat a cel IP fele. |  |
| `AFAIL-007` | `ssh_failed_logins` |  |  | `<kitoltendo, parancs nelkul>` | Sikertelen SSH login jelzes lehet. | SSH probalkozasok a cel IP fele. |  |
| `BSSH-025` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `BSSH-026` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `BPKG-025` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `ABRUTE-007` | `ssh_bruteforce` |  |  | `<kitoltendo, parancs nelkul>` | Tobb sikertelen SSH probalkozas vagy brute-force jelzes lehet. | Ismetelt SSH kapcsolatok a cel IP fele. |  |
| `AFIM-007` | `file_integrity_change` |  |  | `<kitoltendo, parancs nelkul>` | FIM valtozas jelzes varhato. | SSH menedzsment forgalom; host alerttel korrelalando. |  |
| `BPKG-026` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `BPKG-027` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `BSSH-027` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `APRIV-007` | `privilege_change` |  |  | `<kitoltendo, parancs nelkul>` | Jogosultsagvaltozas vagy sudo audit jelzes varhato. | SSH menedzsment forgalom; host alerttel korrelalando. |  |
| `APORT-008` | `port_scan` |  |  | `<kitoltendo, parancs nelkul>` | Recon vagy scan jellegu Wazuh jelzes lehet. | Tobb rovid kapcsolat a cel IP fele. |  |
| `BSSH-028` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `BSSH-029` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `BPKG-028` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `AFAIL-008` | `ssh_failed_logins` |  |  | `<kitoltendo, parancs nelkul>` | Sikertelen SSH login jelzes lehet. | SSH probalkozasok a cel IP fele. |  |
| `ABRUTE-008` | `ssh_bruteforce` |  |  | `<kitoltendo, parancs nelkul>` | Tobb sikertelen SSH probalkozas vagy brute-force jelzes lehet. | Ismetelt SSH kapcsolatok a cel IP fele. |  |
| `BPKG-029` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `BPKG-030` | `benign_package_update` |  |  | `<kitoltendo, parancs nelkul>` | Csomagkezelo vagy audit naplo; benign kontextus. | Host-only lab forgalom; benign SSH inditas igazithato. |  |
| `BSSH-030` | `benign_ssh_login` |  |  | `<kitoltendo, parancs nelkul>` | Sikeres SSH audit bejegyzes; benign kontextus. | SSH kapcsolat `192.168.56.102 -> 192.168.56.101`. |  |
| `AFIM-008` | `file_integrity_change` |  |  | `<kitoltendo, parancs nelkul>` | FIM valtozas jelzes varhato. | SSH menedzsment forgalom; host alerttel korrelalando. |  |
| `APRIV-008` | `privilege_change` |  |  | `<kitoltendo, parancs nelkul>` | Jogosultsagvaltozas vagy sudo audit jelzes varhato. | SSH menedzsment forgalom; host alerttel korrelalando. |  |
