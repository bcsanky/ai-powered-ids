# Wazuh bind mount troubleshooting

Ez a dokumentum a Wazuh Docker stack indításakor előforduló bind mount típushibák diagnosztikáját írja le. Nem hoz létre mérési adatot, nem készít Wazuh alertet, és nem módosítja automatikusan a host fájlokat.

## Hiba jelentése

A következő hiba tipikusan azt jelenti, hogy a Docker egy konténeren belüli fájl célpontra próbál mountolni egy host oldali könyvtárat:

```text
not a directory: Are you trying to mount a directory onto a file (or vice-versa)?
```

A konkrét esetben a compose mount:

```text
./wazuh/config/wazuh_indexer/wazuh.indexer.yml:/usr/share/wazuh-indexer/config/opensearch.yml
```

A konténer oldali cél fájl, ezért a host oldali `infra/wazuh/config/wazuh_indexer/wazuh.indexer.yml` útvonalnak is fájlnak kell lennie. Ha ez az útvonal hiányzik, egyes Docker használati minták mellett véletlenül könyvtárként jöhet létre, ami később megakasztja a konténer indulását.

## Ellenőrzés

Automatikus diagnosztika:

```bash
make infra-diagnostics
```

Kézi ellenőrzés:

```bash
ls -ld infra/wazuh/config/wazuh_indexer/wazuh.indexer.yml
file infra/wazuh/config/wazuh_indexer/wazuh.indexer.yml
git ls-files infra/wazuh/config/wazuh_indexer/wazuh.indexer.yml
```

Ha a `ls -ld` sor elején `d` látható, akkor könyvtárról van szó. Ebben az esetben a Docker fájl targetre könyvtár source-t próbál mountolni.

## Biztonságos javítási irány

1. Állítsd le a Wazuh stack indítási próbálkozását.
2. Ellenőrizd, hogy a problémás path könyvtár vagy fájl.
3. Ha könyvtárként jött létre, ne töröld automatikusan. Ellenőrizd kézzel, hogy nincs-e benne megőrzendő tartalom.
4. Verziókezelt konfigurációs fájlnál a biztonságos helyreállítás lehet:

```bash
git restore -- infra/wazuh/config/wazuh_indexer/wazuh.indexer.yml
```

5. Ha tanúsítvány hiányzik, ne készíts nem valós lab tanúsítványt méréshez. Használd a dokumentált Wazuh lab cert generálási folyamatot vagy a Wazuh setup hivatalos lépéseit.

## Mikor git restore

`git restore -- <path>` csak akkor javasolt, ha a hiányzó vagy hibás fájl verziókezelt konfiguráció:

```bash
git ls-files <path>
```

Ha a parancs visszaadja az útvonalat, a fájl Gitből helyreállítható. Tanúsítványokra ez általában nem jó megoldás, mert azok lokális lab setuphoz tartozó érzékeny vagy generált állományok lehetnek.

## Mit nem szabad

- Nem szabad kézzel pótlólagos Wazuh alertet készíteni.
- Nem szabad mesterséges mérési adatot létrehozni.
- Nem szabad kézzel írt metrikát vagy metrics CSV-t létrehozni.
- Nem szabad nem dokumentált tanúsítványt mérési bizonyítékként használni.
- Nem szabad jelszót, kulcsot vagy tokent riportba írni.

## Kapcsolódó riportok

Az infra diagnosztika kimenetei:

- `reports/infra_diagnostics/wazuh_bind_mount_check.md`
- `reports/infra_diagnostics/docker_compose_mount_policy.md`

Ezek futási diagnosztikai riportok, Gitbe nem kerülnek.

