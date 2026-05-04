# Batch scoring bemeneti példák

Ez a könyvtár demonstrációs célú mintabemeneteket tartalmaz a batch scoring parancshoz. A `sample_events.jsonl` és `sample_events.csv` fájlok csak API/scoring demonstrációra szolgálnak. Nem mérési eredmények, nem használhatók dolgozati benchmarkhoz, és nem használhatók Wazuh+AE real-lab összehasonlításhoz.

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

## Korlát

Az itt található minták nem helyettesítik a tényleges lab ground truth, lab feature és Wazuh alert export bemeneteket. Real-lab méréshez verified real_lab provenance és valós mérési inputok szükségesek.
