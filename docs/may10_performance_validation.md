# Május 10-i teljesítménymérés és hibaanalízis validáció

## Dátum és branch

- Dátum: 2026-05-10
- Branch: `thesis/final`

## Cél

A validáció célja a rendszermérnöki minőség erősítése volt a meglévő batch scoring teljesítménymérés kiegészítésével. A mérés CPU-idő és memóriahasználati oszlopokkal bővült, miközben megmaradtak a korábbi latency és throughput kimenetek. A kiegészítés nem indított új modell-tanítást és nem változtatott a detektálási logikán.

Az eredeti event/perc terhelési célok helyett a prototípus jelen állapotában batch scoring/inference teljesítménymérés készült. A mérés 100, 500, 1000, 5000 és 10000 esemény feldolgozását vizsgálta több batch size mellett. Ez a mérés a scoring komponens feldolgozási költségét mutatja, nem a teljes SIEM/Wazuh end-to-end terhelhetőségét.

## Futtatott parancsok

```bash
make final-validate
make benchmark-scoring
make plot-performance
make generate-performance-report
make <kompatibilis performance export cél>
make final-day10-performance
```

## Mérési beállítások

Mért eseményszámok:

- 100
- 500
- 1000
- 5000
- 10000

A várható benchmark sorok száma:

```text
5 event-count × 4 batch-size × 3 repeat = 60 sor
```

Az ellenőrzött futásban a `benchmark_results.csv` és a `performance_metrics.csv` egyaránt 60 sort tartalmazott.

Batch size értékek:

- 1
- 10
- 50
- 100

## Mért mutatók

- throughput
- átlagos késleltetés
- p50 késleltetés
- p95 késleltetés
- p99 késleltetés
- processz CPU-idő
- CPU-idő eseményenként
- memória RSS a futás előtt és után, ha elérhető
- memória RSS delta, ha elérhető
- csúcsmemória, ha elérhető
- hibás események száma

A CPU-idő mérése standard könyvtári `time.process_time()` alapján történik. A memória RSS érték psutil jelenléte esetén érhető el, a csúcsmemória pedig Unix/Linux környezetben `resource.getrusage()` alapján rögzíthető. Ha egy memóriaérték az adott platformon nem mérhető megbízhatóan, az oszlop üresen maradhat.

Az ellenőrzött futásban a CPU-idő és a csúcsmemória oszlopok kitöltődtek. A memória RSS előtte/utána és delta oszlopai a használt környezetben üresek maradtak, mert a közvetlen RSS méréshez szükséges opcionális mérési forrás nem volt elérhető.

A CPU mérés processz CPU-időként értelmezendő, nem teljes rendszer CPU százalékként. A RAM mérés RSS, delta és csúcsmemória jellegű, attól függően, hogy az adott platformon melyik adat érhető el.

## Létrejött kimenetek

- `reports/performance/benchmark_results.csv`
- `results/performance/performance_metrics.csv`
- `reports/performance/latency_by_batch_size.png`
- `reports/performance/throughput_by_batch_size.png`
- `reports/performance/scoring_time_distribution.png`
- `reports/performance/resource_usage_by_batch_size.png`
- `figures/final/latency_by_load.png`
- `figures/final/throughput.png`
- `reports/performance/performance_report.md`
- `reports/performance/performance_report.html`

## Hibaanalízis dokumentum

A hibaanalízis és korlátok külön dokumentumban szerepelnek:

- `docs/thesis_limitations_and_error_analysis.md`

## Dolgozatba beemelhető részek

- 6.5 Teljesítményértékelés
- 6.6 Hibaanalízis
- 6.7 Korlátok

A végleges számozást a Word-dokumentum tartalomjegyzéke szerint kell igazítani.

## Korlátok

- Lokális/labor mérés.
- Batch scoring mérés.
- Nem event/perc alapú replay mérés.
- Nem éles üzemi teljesítménygarancia.
- Nem natív Wazuh indexelési teljesítmény.
- Nem teljes SIEM feldolgozási lánc mérése.
- Hardverfüggő eredmény.
