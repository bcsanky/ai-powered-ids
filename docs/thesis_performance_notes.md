# Teljesítménymérési jegyzetek a diplomamunkához

## Javasolt alfejezet

A teljesítménymérési eredmények a 6. fejezetben külön alfejezetként szerepelhetnek, például:

```text
6.x A batch scoring feldolgozási lánc lokális teljesítménymérése
```

Az alfejezetben jelezni kell, hogy a mérés laboratóriumi környezetben készült, és nem általánosítható éles üzemi SOC-terhelésre.

## Javasolt táblázatok

- Benchmark konfigurációk:
  - bemeneti eseményfájl;
  - eseményszámok;
  - batch size értékek;
  - ismétlésszám;
  - mérési környezet fő jellemzői.
- Fő throughput és latency eredmények:
  - legjobb áteresztőképesség;
  - legalacsonyabb p95 késleltetés;
  - hibás események száma;
  - batch size és eseményszám szerinti bontás.

## Javasolt ábrák

- `reports/performance/latency_by_batch_size.png`
- `reports/performance/throughput_by_batch_size.png`
- `reports/performance/scoring_time_distribution.png`

A dolgozatba rendezett másolatok:

- `reports/final/thesis_figures/performance_latency_by_batch_size.png`
- `reports/final/thesis_figures/performance_throughput_by_batch_size.png`
- `reports/final/thesis_figures/performance_scoring_time_distribution.png`

## Értelmezési szabályok

- Az eredmény lokális/labor teljesítménymérés.
- Az események ismétlése a feldolgozási kapacitás mérésére szolgál.
- Az eredmény hardver- és környezetfüggő.
- Nem natív Wazuh indexelési teljesítmény.
- Nem teljes SIEM end-to-end benchmark.
- Nem szabad éles üzemi teljesítménygaranciaként megfogalmazni.

## Dolgozatba emelhető rövid szöveg

A prototípus batch scoring komponensén lokális/labor teljesítménymérés készült több eseményszám és batch size beállítás mellett. A mérés az esemény/másodperc alapú áteresztőképességet, az átlagos és percentilis késleltetést, valamint a hibás események számát rögzíti. Az eredmények a vizsgált környezetre és bemeneti eseménysorra vonatkoznak.
