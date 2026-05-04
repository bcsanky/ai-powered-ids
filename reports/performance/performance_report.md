# Batch scoring teljesítményriport

## Mérés célja

A mérés célja a meglévő AE-Minimal batch scoring feldolgozási lánc lokális/labor sebességének, késleltetésének és áteresztőképességének dokumentált vizsgálata.

## Bemenet és modell

- Bemeneti eseménykészlet: `examples/lab/lab_events.jsonl`
- Modell: AE-Minimal végleges modellkimenet.
- Preprocess fájl: `data/processed/final/ae_minimal/preprocess.pkl`
- Eseményszámok: 100, 500, 1000, 5000
- Batch size értékek: 1, 10, 50, 100

## Rendszerinformáció

- timestamp: `2026-05-04T17:40:38`
- python_version: `3.8.10 (default, Mar 18 2025, 20:04:55) 
[GCC 9.4.0]`
- platform: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.29`
- processor: `x86_64`
- cpu_count: `12`
- input_file: `examples/lab/lab_events.jsonl`
- preprocess_file: `data/processed/final/ae_minimal/preprocess.pkl`

## Fő eredmények

- Legjobb áteresztőképesség: 382.34 esemény/másodperc, batch size 1, eseményszám 500.
- Legalacsonyabb p95 késleltetés: 3.2442 ms, batch size 1, eseményszám 500.
- Hibás események összesen: 0.

## Összesített táblázat

| total_events | batch_size | events_per_second | avg_latency_ms | p50_latency_ms | p95_latency_ms | p99_latency_ms | failed_events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100.0000 | 1.0000 | 365.3581 | 2.7750 | 2.6256 | 3.5844 | 3.9788 | 0.0000 |
| 100.0000 | 10.0000 | 364.5700 | 2.7474 | 2.6288 | 3.4352 | 4.3815 | 0.0000 |
| 100.0000 | 50.0000 | 362.3441 | 2.7689 | 2.5400 | 3.9187 | 4.5351 | 0.0000 |
| 100.0000 | 100.0000 | 344.6342 | 2.9117 | 2.6095 | 4.6927 | 5.9383 | 0.0000 |
| 500.0000 | 1.0000 | 382.3398 | 2.6141 | 2.4926 | 3.2442 | 4.3810 | 0.0000 |
| 500.0000 | 10.0000 | 341.2427 | 2.9334 | 2.6241 | 4.4592 | 7.3890 | 0.0000 |
| 500.0000 | 50.0000 | 354.0421 | 2.8270 | 2.6088 | 4.1576 | 5.4964 | 0.0000 |
| 500.0000 | 100.0000 | 354.4916 | 2.8259 | 2.6017 | 4.3252 | 5.6874 | 0.0000 |
| 1000.0000 | 1.0000 | 354.9530 | 2.8214 | 2.5385 | 3.7751 | 9.4859 | 0.0000 |
| 1000.0000 | 10.0000 | 372.0756 | 2.6874 | 2.5122 | 3.3206 | 6.9661 | 0.0000 |
| 1000.0000 | 50.0000 | 356.7941 | 2.8062 | 2.5125 | 3.6903 | 8.2652 | 0.0000 |
| 1000.0000 | 100.0000 | 364.2594 | 2.7473 | 2.5259 | 3.5400 | 8.8025 | 0.0000 |
| 5000.0000 | 1.0000 | 360.9447 | 2.7697 | 2.5570 | 3.7393 | 5.7658 | 0.0000 |
| 5000.0000 | 10.0000 | 348.5983 | 2.8715 | 2.5850 | 4.1349 | 6.9320 | 0.0000 |
| 5000.0000 | 50.0000 | 338.6175 | 2.9544 | 2.5921 | 4.3461 | 10.8305 | 0.0000 |
| 5000.0000 | 100.0000 | 352.4865 | 2.8360 | 2.5652 | 3.9165 | 6.0647 | 0.0000 |

## Rövid értelmezés

A batch size hatása a lokális mérésben az áteresztőképesség és a késleltetés változásán keresztül értelmezhető. Az eseményszám növelése determinisztikus ismétléssel történt, ezért a mérés a scoring feldolgozási költségét, nem pedig új adatminták detektálási minőségét vizsgálja.

## Dolgozatba emelhető összefoglaló

A prototípus batch scoring komponensén lokális/labor teljesítménymérés készült több eseményszám és batch size beállítás mellett. A mérés az esemény/másodperc alapú áteresztőképességet, az átlagos és percentilis késleltetést, valamint a hibás események számát rögzíti.

## Korlátok

- lokális/labor mérés.
- Hardver- és környezetfüggő eredmények.
- Nem hosszú idejű éles üzemi terhelés.
- Nem natív Wazuh indexelési teljesítmény.
- Batch scoring benchmark, nem teljes SIEM end-to-end benchmark.
