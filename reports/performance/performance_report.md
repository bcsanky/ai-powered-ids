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

- timestamp: `2026-05-04T18:30:05`
- python_version: `3.8.10 (default, Mar 18 2025, 20:04:55) 
[GCC 9.4.0]`
- platform: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.29`
- processor: `x86_64`
- cpu_count: `12`
- input_file: `examples/lab/lab_events.jsonl`
- preprocess_file: `data/processed/final/ae_minimal/preprocess.pkl`
- cpu_time_method: `time.process_time`
- memory_rss_method: `not_available`
- peak_memory_method: `resource.getrusage.ru_maxrss`

## Fő eredmények

- Legjobb áteresztőképesség: 399.14 esemény/másodperc, batch size 100, eseményszám 500.
- Legalacsonyabb p95 késleltetés: 3.0332 ms, batch size 1, eseményszám 500.
- Hibás események összesen: 0.

## CPU- és memóriahasználat

- A CPU-idő mérése `time.process_time()` alapján történt, ezért processzszintű CPU-időt mutat.
- A memória RSS érték psutil jelenléte esetén érhető el.
- Unix/Linux környezetben a csúcsmemória `resource.getrusage()` alapján is rögzíthető.
- Ha egy memóriaérték nem elérhető az adott platformon, az adott CSV mező üresen maradhat.
- Legalacsonyabb CPU-idő eseményenként: 2.4661 ms.
- Legnagyobb mért csúcsmemória: 147.9062 MB.

## Összesített táblázat

| total_events | batch_size | events_per_second | avg_latency_ms | p50_latency_ms | p95_latency_ms | p99_latency_ms | failed_events | process_cpu_time_s | cpu_time_per_event_ms | memory_rss_delta_mb | memory_rss_mb_before | memory_rss_mb_after | peak_memory_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100.0000 | 1.0000 | 382.8020 | 2.6121 | 2.5032 | 3.2764 | 3.7327 | 0.0000 | 0.2614 | 2.6136 | nincs adat | nincs adat | nincs adat | 143.0312 |
| 100.0000 | 10.0000 | 390.0274 | 2.5654 | 2.4116 | 3.5038 | 4.2048 | 0.0000 | 0.2566 | 2.5664 | nincs adat | nincs adat | nincs adat | 143.0312 |
| 100.0000 | 50.0000 | 382.4234 | 2.6214 | 2.5266 | 3.2959 | 3.8780 | 0.0000 | 0.2622 | 2.6224 | nincs adat | nincs adat | nincs adat | 143.1562 |
| 100.0000 | 100.0000 | 320.1848 | 3.1425 | 2.6872 | 5.0353 | 6.8694 | 0.0000 | 0.3144 | 3.1437 | nincs adat | nincs adat | nincs adat | 143.1562 |
| 500.0000 | 1.0000 | 384.5453 | 2.6006 | 2.4419 | 3.0332 | 6.2857 | 0.0000 | 1.3011 | 2.6023 | nincs adat | nincs adat | nincs adat | 143.5312 |
| 500.0000 | 10.0000 | 383.4945 | 2.6087 | 2.4501 | 3.5941 | 4.5447 | 0.0000 | 1.3049 | 2.6098 | nincs adat | nincs adat | nincs adat | 143.5312 |
| 500.0000 | 50.0000 | 395.5124 | 2.5279 | 2.4003 | 3.2557 | 4.2657 | 0.0000 | 1.2644 | 2.5288 | nincs adat | nincs adat | nincs adat | 143.5312 |
| 500.0000 | 100.0000 | 399.1433 | 2.5045 | 2.4033 | 3.0855 | 3.9694 | 0.0000 | 1.2527 | 2.5053 | nincs adat | nincs adat | nincs adat | 143.5312 |
| 1000.0000 | 1.0000 | 375.6807 | 2.6681 | 2.4407 | 3.3337 | 7.2131 | 0.0000 | 2.6701 | 2.6701 | nincs adat | nincs adat | nincs adat | 144.1562 |
| 1000.0000 | 10.0000 | 382.0030 | 2.6188 | 2.4548 | 3.3422 | 6.7913 | 0.0000 | 2.6200 | 2.6200 | nincs adat | nincs adat | nincs adat | 144.1562 |
| 1000.0000 | 50.0000 | 356.7862 | 2.8057 | 2.4908 | 3.8141 | 10.3179 | 0.0000 | 2.8070 | 2.8070 | nincs adat | nincs adat | nincs adat | 144.2812 |
| 1000.0000 | 100.0000 | 378.1033 | 2.6465 | 2.4336 | 3.4340 | 7.1131 | 0.0000 | 2.6475 | 2.6475 | nincs adat | nincs adat | nincs adat | 144.2812 |
| 5000.0000 | 1.0000 | 381.3835 | 2.6208 | 2.4301 | 3.3699 | 5.3162 | 0.0000 | 13.1133 | 2.6227 | nincs adat | nincs adat | nincs adat | 147.7812 |
| 5000.0000 | 10.0000 | 342.0606 | 2.9303 | 2.5594 | 4.4085 | 10.6131 | 0.0000 | 14.6583 | 2.9317 | nincs adat | nincs adat | nincs adat | 147.9062 |
| 5000.0000 | 50.0000 | 352.1586 | 2.8392 | 2.5238 | 4.1543 | 9.8529 | 0.0000 | 14.2020 | 2.8404 | nincs adat | nincs adat | nincs adat | 147.9062 |
| 5000.0000 | 100.0000 | 370.2655 | 2.6997 | 2.4995 | 3.5796 | 5.0849 | 0.0000 | 13.5042 | 2.7008 | nincs adat | nincs adat | nincs adat | 147.9062 |

## Rövid értelmezés

A batch size hatása a lokális mérésben az áteresztőképesség és a késleltetés változásán keresztül értelmezhető. Az eseményszám növelése determinisztikus ismétléssel történt, ezért a mérés a scoring feldolgozási költségét, nem pedig új adatminták detektálási minőségét vizsgálja.

## Dolgozatba emelhető összefoglaló

A prototípus batch scoring komponensén lokális/labor teljesítménymérés készült több eseményszám és batch size beállítás mellett. A mérés az esemény/másodperc alapú áteresztőképességet, az átlagos és percentilis késleltetést, valamint a hibás események számát rögzíti.

## Korlátok

- lokális/labor mérés.
- Hardver- és környezetfüggő eredmények.
- Nem hosszú idejű éles üzemi terhelés.
- Nem natív Wazuh indexelési teljesítmény.
- Batch scoring teljesítménymérés, nem teljes SIEM feldolgozási lánc mérése.
