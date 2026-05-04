# Lab/replay bemeneti események

Ez a könyvtár kontrollált replay-alapú demonstrációs eseménysort tartalmaz a prototípus eseményszintű pontozásának bemutatásához. Az események nem valós incidensbizonyítékok, és nem natív Wazuh exportból származnak.

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
