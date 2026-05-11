# real-lab-001 Zeek capture runbook

Ez a runbook a `real-lab-001` session Zeek-alapu network inputjanak elokesziteset irja le. A meres pcap capture-rel indul Kalin, majd offline Zeek feldolgozassal keszul a `conn.log`. A dokumentum nem hoz letre kezzel gyartott pcap, conn.log, feature vagy provenance allomanyt.

## Biztonsagi keret

- Engedelyezett cel: `192.168.56.101`.
- Engedelyezett labor forras: `192.168.56.102`.
- A capture filter csak a `192.168.56.101` es `192.168.56.102` kozotti forgalmat engedje at.
- Nyilvanos IP, ceges halozat, router, mas VM vagy internetes host nem lehet cel.
- A Zeek/Flow capture az elso event marker elott induljon, es csak az utolso event marker utan alljon le.

## Miert Kalin fusson a capture

- Kali az attacker VM, ezert a `192.168.56.102 -> 192.168.56.101` host-only forgalom a sajat labor interfeszen biztosan latszik.
- WSL-ben a VirtualBox host-only forgalom lathatosaga platform- es routingfuggo lehet, ezert konnyebb reszleges vagy rossz interfeszes capture-t kesziteni.
- Kali oldalon egyszerubb ellenorizni, hogy a capture pontosan a meresi forras es cel kozotti csomagokat tartalmazza.
- A pcap offline Zeek feldolgozasa reprodukalhato: a nyers capture megmarad, a `conn.log` kesobb ujrageneralhato.

## Kali interfesz azonositas

Kalin keresd meg azt az interfeszt, amelyen a `192.168.56.102` cim van:

```bash
ip -br addr
```

Reszletes ellenorzeshez:

```bash
ip addr show
```

Az interfesz nevet rogzitsd az operatori jegyzetben, majd a capture shellben allitsd be:

```bash
export KALI_IFACE=<interface_with_192.168.56.102>
```

Csak akkor induljon a meres, ha a kivalasztott interfeszen tenylegesen a `192.168.56.102` cim szerepel.

## Pcap capture inditasa Kalin

Inditas az elso event marker elott:

```bash
mkdir -p ~/real-lab-001-capture
sudo tcpdump -i "$KALI_IFACE" -s 0 -nn \
  -w ~/real-lab-001-capture/real-lab-001.pcap \
  'host 192.168.56.101 and host 192.168.56.102'
```

A filter szandekosan szuk: csak a target VM es a Kali VM kozotti host-only forgalmat rogzit. Ha a tcpdump nem indul, vagy a filter nem ervenyes, a 100 esemenyes sessiont nem szabad elkezdeni.

## Capture leallitasa

Az utolso event marker lezaratasa utan varj roviden, hogy a kesoi csomagok is bekeruljenek, majd a tcpdump terminalban:

```text
Ctrl+C
```

Jegyezd fel az UTC start es stop idopontot. A pcap fajl csak akkor hasznalhato, ha az elso event elotti es az utolso event utani idoszakot is lefedi.

## Zeek conn.log generalasa pcapbol

Offline feldolgozas Kalin vagy WSL-ben is vegezheto, ha a Zeek elerheto. Javasolt kulon munkakonyvtarban futtatni, mert Zeek tobb logot is letrehoz.

```bash
mkdir -p ~/real-lab-001-zeek
cd ~/real-lab-001-zeek
zeek -r ~/real-lab-001-capture/real-lab-001.pcap
```

Ellenorizd, hogy letrejott:

```text
~/real-lab-001-zeek/conn.log
```

Ha nincs `conn.log`, a meresi pipeline nem indithato tovabb Zeek inputtal.

## conn.log masolasa a repoba

A Zeek kapcsolatlog vegleges repo-beli bemeneti helye:

```text
data/lab/zeek/conn.log
```

Masolas pelda:

```bash
mkdir -p data/lab/zeek
cp ~/real-lab-001-zeek/conn.log data/lab/zeek/conn.log
```

Ez valos meresi bemenet, ezert csak a tenyleges `real-lab-001` pcapbol generalva kerulhet ide. Kezzel gyartott vagy nem tenyleges pcapbol generalt `conn.log` nem hasznalhato.
