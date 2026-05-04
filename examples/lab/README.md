# Lab/replay bemeneti események

Ez a könyvtár kontrollált replay-alapú demonstrációs eseménysort tartalmaz a prototípus eseményszintű pontozásának bemutatásához. Az `lab_events.jsonl` és `lab_events.csv` fájlok demonstrációs bemenetek: nem valós Wazuh lab mérésből származnak, nem használhatók real-lab benchmarkhoz, és nem tekinthetők natív Wazuh teljesítménymérésnek.

## Szcenáriók

- `benign_activity`: normál jellegű, mérsékelt forgalmi intenzitású flow események.
- `port_scan`: több célportot érintő, rövid időtartamú és magasabb csomagrátájú események.
- `ssh_bruteforce`: ismétlődő SSH kapcsolati kísérleteket leíró események.
- `combined_suspicious`: ML és szabályalapú jelzés szempontjából is gyanúsabb események.

## Fájlok

- `lab_events.jsonl`: JSONL bemenet a batch scoringhoz.
- `lab_events.csv`: ugyanaz a kontrollált eseménysor CSV formátumban.

## Futtatás

```bash
make score-lab-events
make generate-case-studies
```

Az elvárt kimenetek a `reports/lab/` könyvtárban jönnek létre. A kimenetek a diplomamunka esettanulmányos bemutatását támogatják, de nem tekinthetők éles SOC-validációnak vagy natív Wazuh teljesítménymérésnek.

## Real-lab méréshez szükséges bemenetek

Valódi lab-alapú Wazuh+AE összehasonlításhoz nem ezeket a demo fájlokat kell használni, hanem tényleges lab futásból származó bemeneteket:

- `data/lab/lab_ground_truth.csv`
- `data/lab/lab_features.csv`
- `data/wazuh/alerts.jsonl`

A real-lab mérési csomagnak `reports/real_measurement/measurement_provenance.json` fájllal kell rendelkeznie. A provenance rögzíti a bemeneti állományok útvonalát és SHA256 azonosítóját, de nem helyettesíti a szakmai validációt.
