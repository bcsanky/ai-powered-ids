# Teljesítménymérési futtatási leírás

## Cél

A teljesítménymérés célja a meglévő AE-Minimal batch scoring feldolgozási lánc lokális/labor vizsgálata. A mérés a feldolgozási időt, a késleltetést, az áteresztőképességet és a hibás események számát rögzíti. Nem éles üzemi benchmark, nem natív Wazuh indexelési teljesítménymérés, és nem teljes SIEM end-to-end mérés.

## Előfeltételek

- Elérhető AE-Minimal modellkimenet.
- Elérhető preprocess állomány.
- Elérhető lab/replay bemeneti eseményfájl:

```text
examples/lab/lab_events.jsonl
```

## Benchmark futtatása

```bash
make benchmark-scoring
```

A target a következő beállításokat használja:

- eseményszámok: `100`, `500`, `1000`, `5000`
- batch size értékek: `1`, `10`, `50`, `100`
- ismétlések száma: `3`

Az eseményszám növelése determinisztikus ismétléssel történik. Ez kizárólag a feldolgozási kapacitás mérését szolgálja, nem új detektálási adatkészlet.

## Ábrák előállítása

```bash
make plot-performance
```

Kimenetek:

```text
reports/performance/latency_by_batch_size.png
reports/performance/throughput_by_batch_size.png
reports/performance/scoring_time_distribution.png
```

## Riport generálása

```bash
make generate-performance-report
```

Kimenetek:

```text
reports/performance/performance_report.md
reports/performance/performance_report.html
```

## Eredmények helye

```text
reports/performance/benchmark_results.csv
reports/performance/benchmark_summary.md
reports/performance/system_info.json
reports/performance/performance_report.md
reports/performance/performance_report.html
```

## Metrikák értelmezése

- `events_per_second`: áteresztőképesség esemény/másodpercben.
- `avg_latency_ms`: átlagos eseményszintű késleltetés.
- `p50_latency_ms`, `p95_latency_ms`, `p99_latency_ms`: percentilis késleltetések.
- `failed_events`: hibával zárult pontozási kísérletek száma.
- `model_load_time_s`: egyszer mért modellbetöltési idő.

## Dolgozati felhasználás

A teljesítménymérési ábrák és riport a diplomamunka 5. fejezetében a prototípus működési jellemzőihez, a 6. fejezetben pedig a lokális teljesítménymérési alfejezethez használhatók.

## Korlátok

- Lokális/labor mérés.
- Hardver- és környezetfüggő eredmény.
- Nem hosszú idejű éles üzemi terhelés.
- Nem natív Wazuh indexelési teljesítmény.
- Batch scoring mérés, nem teljes SIEM end-to-end benchmark.
