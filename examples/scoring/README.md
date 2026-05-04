# Batch scoring bemeneti példák

Ez a könyvtár demonstrációs célú mintabemeneteket tartalmaz a batch scoring parancshoz. A példák nem valós incidensbizonyítékok, hanem a pontozási feldolgozási lánc bemutatását szolgálják.

## Fájlok

- `sample_events.jsonl`: JSONL formátumú mintaesemények.
- `sample_events.csv`: ugyanaz a mintakészlet CSV formátumban.

## Futtatás

```bash
make score-sample-events
```

Az elvárt kimenet:

```text
reports/scored_events.jsonl
```

A kimeneti állomány eseményenként tartalmazza az anomáliapontszámot, a kiválasztott küszöböt, az ML riasztást, a szabályalapú jelzést és a kockázati szintet.
