# Május 9-i teljesítménymérés és stabilitási validáció

## Dátum és branch

- Dátum: 2026-05-09
- Branch: `thesis/final`

## Napi cél

A napi cél a meglévő batch scoring feldolgozási lánc lokális/labor teljesítménymérési rétegének elkészítése volt. A mérés sebességet, késleltetést, áteresztőképességet és hibás eseményszámot rögzít. Nem új detektálási modell, nem éles SOC-terhelés és nem natív Wazuh teljesítménymérés.

## Előzetes ellenőrzés

- A branch ellenőrzése sikeres volt: `thesis/final`.
- A május 8-i dokumentációk és lab/replay bemenetek rendelkezésre álltak.
- A pontozott lab kimenetek rendelkezésre álltak.

## Futtatott parancsok

```bash
make final-validate
make benchmark-scoring
make plot-performance
make generate-performance-report
make collect-thesis-figures
make final-day9
```

## Benchmark input

```text
examples/lab/lab_events.jsonl
```

Az eseményszám növelése determinisztikus ismétléssel történt, kizárólag teljesítménymérési célra.

## Benchmark output

```text
reports/performance/benchmark_results.csv
reports/performance/benchmark_summary.md
reports/performance/system_info.json
```

## Létrejött ábrák

```text
reports/performance/latency_by_batch_size.png
reports/performance/throughput_by_batch_size.png
reports/performance/scoring_time_distribution.png
```

## Riport

```text
reports/performance/performance_report.md
reports/performance/performance_report.html
```

## Fő mért mutatók

- `events_per_second`
- `avg_latency_ms`
- `p50_latency_ms`
- `p95_latency_ms`
- `p99_latency_ms`
- `failed_events`

## Futtatási eredmény

- `make final-validate`: sikeres, 39 teszt lefutott, 1 nem blokkoló sklearn konvergencia figyelmeztetéssel.
- `make benchmark-scoring`: sikeres, 48 mérési sor készült.
- `make plot-performance`: sikeres, 3 performance ábra készült.
- `make generate-performance-report`: sikeres.
- `make collect-thesis-figures`: sikeres, 10 ábra átmásolva, hiányzó ábra nem volt.
- `make final-day9`: sikeres.

## Rövid eredményösszefoglaló

- Eseményszámok: `100`, `500`, `1000`, `5000`.
- Batch size értékek: `1`, `10`, `50`, `100`.
- Ismétlések száma: `3`.
- Hibás események száma: `0`.
- A riport aggregált eredménye szerint a legjobb áteresztőképesség 382.34 esemény/másodperc volt.
- A legalacsonyabb p95 késleltetés 3.2442 ms volt.

## Korlátok

- Lokális/labor mérés.
- Batch scoring mérés.
- Nem éles üzemi benchmark.
- Nem natív Wazuh indexelési teljesítmény.
- Hardverfüggő eredmény.

## Következő napra átadott feladatok

- A diplomamunka 5. Implementáció fejezetének írása.
- A diplomamunka 6. Eredmények és értékelés fejezetének írása.
- Hibaanalízis és korlátok kidolgozása.
