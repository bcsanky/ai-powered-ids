# real-lab-001 scope es safety

Ez a dokumentum a `real-lab-001` meresi session biztonsagi hatarait rogziti. A cel a 100 esemenyes valos labor meres elokeszitese, nem meresi eredmenyek eloallitasa.

## Session metadata

| Mezo | Ertek |
| --- | --- |
| Branch | `thesis/final` |
| Session ID | `real-lab-001` |
| Target VM | `192.168.56.101` |
| Attacker VM | `192.168.56.102` |
| Wazuh Manager | `192.168.56.1` |
| WSL IP | `172.20.31.173` |
| Wazuh login | `admin`; a jelszo lokalisan ismert, fajlba nem kerulhet |
| Target Wazuh agent status | Active |
| Kali snapshot | ready |
| Target snapshot | ready |

## Engedelyezett cel

- Az egyetlen engedelyezett cel IP: `192.168.56.101`.
- Minden attack jellegu esemenyt kizarolag erre az IP-re szabad iranyitani.
- A benign SSH es benign package update esemenyek is a `192.168.56.102 -> 192.168.56.101` labor iranyhoz igazodnak.

## Tiltott celok

Tilos celkent megadni vagy erinteni:

- nyilvanos IP-cimeket;
- ceges vagy egyetemi halozati cimeket;
- routert vagy gateway-t;
- barmely masik VM-et;
- internetes hostokat;
- barmely olyan gepet, amely nem `192.168.56.101`.

## Muveleti szabalyok

- Minden tamado aktivitasnak a `192.168.56.101` celgepre kell korlatozodnia.
- Nem hasznalhato destruktiv, tartosan modositott vagy kontrollalatlan tamado parancs.
- A dokumentumok nem tartalmazhatnak Wazuh jelszot vagy mas titkot.
- Fake alert, fake feature, fake metric, fake provenance vagy mas fiktiv meresi eredmeny nem keszulhet.
- A session csak akkor indithato, ha a snapshotok elerhetok, a Wazuh agent Active, es a meresi idointervallum elore feljegyezheto.
