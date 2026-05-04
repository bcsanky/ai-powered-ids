# Valódi lab mérési export és csomagolási runbook

Ez a runbook a natív Wazuh alert export, a valós lab-alapú Wazuh+AE kiértékelés és a beadási melléklethez használható mérési csomag előállításának lépéseit foglalja össze. A folyamat kizárólag tényleges lab bemenetekből dolgozik; hiányzó input esetén a parancsok hibával állnak le, és nem készítenek hamis mérési eredményt.

## Előfeltételek

- Rendelkezésre áll a `data/lab/lab_ground_truth.csv` fájl a lab eseményablakokkal.
- Rendelkezésre áll a `data/lab/lab_features.csv` fájl az AE-Minimal kompatibilis feature-ökkel.
- A Wazuh alert export előállítható OpenSearchből vagy manuális JSON exportból.
- A lab gépek időszinkronja ellenőrzött, mert a Wazuh korreláció időablakokra épül.
- A Wazuh export, a ground truth és a feature fájl ugyanahhoz a mérési időablakhoz tartozik.

## Wazuh alert export OpenSearchből

OpenSearch alapú Wazuh indexből a következő paranccsal készíthető időablakra szűrt JSONL export:

```bash
make wazuh-export-opensearch \
  WAZUH_EXPORT_START=2026-05-04T10:00:00Z \
  WAZUH_EXPORT_END=2026-05-04T11:00:00Z \
  OPENSEARCH_PASSWORD=<jelszó>
```

A célfájl alapértelmezés szerint:

```text
data/wazuh/alerts.jsonl
```

A parancs metaadatot is készít:

```text
data/wazuh/alerts_export_metadata.json
```

A metaadat nem tartalmaz jelszót. Ha a TLS-ellenőrzés ki van kapcsolva, azt a metaadat labor környezeti megjegyzésként rögzíti.

## Wazuh alert export manuális JSON fájlból

Ha a Wazuh vagy OpenSearch felületről már rendelkezésre áll manuális export, abból normalizált JSONL készíthető:

```bash
make wazuh-export-from-file \
  WAZUH_RAW_EXPORT=raw/wazuh_export.json \
  WAZUH_EXPORT_START=2026-05-04T10:00:00Z \
  WAZUH_EXPORT_END=2026-05-04T11:00:00Z
```

A script JSON listát, OpenSearch `hits.hits` formátumot, JSONL bemenetet és egyetlen JSON objektumot is kezel. Az export időszűrést végez `@timestamp` vagy `timestamp` mező alapján.

## Alert összefoglaló készítése

A Wazuh export rövid ellenőrző összefoglalója:

```bash
make wazuh-export-summary
```

Kimenetek:

- `reports/wazuh_export/wazuh_export_summary.md`
- `reports/wazuh_export/wazuh_export_rule_summary.csv`
- `reports/wazuh_export/wazuh_export_timeline.csv`
- `reports/wazuh_export/run_metadata.json`

Az összefoglaló alert darabszámot, időintervallumot, gyakori rule azonosítókat, rule level értékeket, agenteket és forrás IP-ket tartalmaz, ha ezek a mezők rendelkezésre állnak.

## Teljes real-lab mérési csomag

A teljes mérési csomag előállítása:

```bash
make final-real-measurement-package
```

A cél sorrendben futtatja:

- lab input validáció;
- Wazuh-only, AE-Minimal lab és hibrid kiértékelés;
- Wazuh export összefoglaló;
- mérési csomag validáció;
- dolgozatba beemelhető real-lab riport;
- mérési manifest.

Ha Zeek `conn.log` vagy általános flow CSV alapján kell előállítani a `lab_features.csv` fájlt, használható:

```bash
make final-real-measurement-package-zeek
```

vagy:

```bash
make final-real-measurement-package-flow-csv
```

## Mérési manifest

A mérési manifest külön is elkészíthető:

```bash
make real-measurement-manifest
```

Kimenetek:

- `reports/real_measurement/measurement_manifest.csv`
- `reports/real_measurement/measurement_manifest.md`
- `reports/real_measurement/measurement_manifest.json`

A manifest relatív fájlutat, fájlméretet, SHA256 hash-t, kategóriát és beadási mellékletre vonatkozó javaslatot tartalmaz. A `data/wazuh/alerts.jsonl` alapértelmezés szerint nem kerül automatikusan beadási mellékletbe, mert érzékeny logadatot tartalmazhat.

## Beadási mellékletbe javasolt állományok

Általában mellékelhetők:

- metrikatáblák;
- összehasonlító Markdown táblázatok;
- ábrák;
- validációs riportok;
- mérési manifest;
- anonimizált real-lab riportok.

Külön mérlegelést igényelnek:

- nyers Wazuh alert exportok;
- pcap vagy pcapng állományok;
- hostneveket, IP-címeket vagy környezeti azonosítókat tartalmazó fájlok;
- nagy méretű köztes állományok.

## Anonimizálás

IP-címek és hostnevek determinisztikus anonimizálásához:

```bash
make real-measurement-redact
```

Kimenet:

```text
reports/real_measurement_redacted/
```

Az anonimizálási mapping érzékeny állomány, ezért külön kell kezelni, és nem célszerű verziókezelésben tartani.

## Dolgozatba emelés

A következő fájl közvetlenül használható a 6. fejezet real-lab eredményeinek megfogalmazásához:

```text
reports/real_measurement/thesis_real_lab_section.md
```

A szakasz csak a ténylegesen előállított metrikák alapján értelmezhető. Ha valamely bemenet vagy eredmény hiányzik, a real-lab Wazuh-only vs AE-only vs hibrid összehasonlítás nem tekinthető lezártnak.
