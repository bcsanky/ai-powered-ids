# Batch scoring teljesítményriport

## Mérés célja

A mérés célja a meglévő AE-Minimal batch scoring feldolgozási lánc lokális/labor sebességének, késleltetésének és áteresztőképességének dokumentált vizsgálata.

## Bemenet és modell

- Bemeneti eseménykészlet: `examples/lab/lab_events.jsonl`
- Modell: AE-Minimal végleges modellkimenet.
- Preprocess fájl: `data/processed/final/ae_minimal/preprocess.pkl`
- Eseményszámok: 100, 500, 1000, 5000, 10000
- Batch size értékek: 1, 10, 50, 100

## Rendszerinformáció

- timestamp: `2026-05-04T18:58:38`
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

- Legjobb áteresztőképesség: 394.18 esemény/másodperc, batch size 100, eseményszám 500.
- Legalacsonyabb p95 késleltetés: 3.2856 ms, batch size 100, eseményszám 500.
- Hibás események összesen: 0.

## CPU- és memóriahasználat

- A CPU-idő mérése `time.process_time()` alapján történt, ezért processzszintű CPU-időt mutat.
- A memória RSS érték psutil jelenléte esetén érhető el.
- Unix/Linux környezetben a csúcsmemória `resource.getrusage()` alapján is rögzíthető.
- Ha egy memóriaérték nem elérhető az adott platformon, az adott CSV mező üresen maradhat.
- Legalacsonyabb CPU-idő eseményenként: 2.4057 ms.
- Legnagyobb mért csúcsmemória: 154.4844 MB.

## Összesített táblázat

| total_events | batch_size | events_per_second | avg_latency_ms | p50_latency_ms | p95_latency_ms | p99_latency_ms | failed_events | process_cpu_time_s | cpu_time_per_event_ms | memory_rss_delta_mb | memory_rss_mb_before | memory_rss_mb_after | peak_memory_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100.0000 | 1.0000 | 358.7651 | 2.8039 | 2.6258 | 3.6538 | 4.0499 | 0.0000 | 0.2806 | 2.8057 | nincs adat | nincs adat | nincs adat | 142.9844 |
| 100.0000 | 10.0000 | 366.3605 | 2.7389 | 2.6019 | 3.6176 | 4.0000 | 0.0000 | 0.2740 | 2.7401 | nincs adat | nincs adat | nincs adat | 142.9844 |
| 100.0000 | 50.0000 | 288.6347 | 3.6060 | 2.5998 | 8.5123 | 9.6500 | 0.0000 | 0.3608 | 3.6077 | nincs adat | nincs adat | nincs adat | 142.9844 |
| 100.0000 | 100.0000 | 359.3276 | 2.8448 | 2.4138 | 5.9307 | 6.4924 | 0.0000 | 0.2846 | 2.8457 | nincs adat | nincs adat | nincs adat | 143.1094 |
| 500.0000 | 1.0000 | 360.1481 | 2.8320 | 2.4876 | 5.6006 | 7.6687 | 0.0000 | 1.4169 | 2.8337 | nincs adat | nincs adat | nincs adat | 143.3594 |
| 500.0000 | 10.0000 | 327.3367 | 3.1086 | 2.4744 | 8.3347 | 10.5388 | 0.0000 | 1.5549 | 3.1099 | nincs adat | nincs adat | nincs adat | 143.4844 |
| 500.0000 | 50.0000 | 381.5198 | 2.6232 | 2.4310 | 3.4427 | 6.7857 | 0.0000 | 1.3121 | 2.6242 | nincs adat | nincs adat | nincs adat | 143.4844 |
| 500.0000 | 100.0000 | 394.1810 | 2.5398 | 2.4210 | 3.2856 | 3.7357 | 0.0000 | 1.2702 | 2.5405 | nincs adat | nincs adat | nincs adat | 143.4844 |
| 1000.0000 | 1.0000 | 346.5355 | 2.8987 | 2.5303 | 4.3071 | 10.4834 | 0.0000 | 2.9004 | 2.9004 | nincs adat | nincs adat | nincs adat | 144.1094 |
| 1000.0000 | 10.0000 | 343.6444 | 2.9105 | 2.5681 | 4.0410 | 10.9002 | 0.0000 | 2.9116 | 2.9116 | nincs adat | nincs adat | nincs adat | 144.1094 |
| 1000.0000 | 50.0000 | 359.4266 | 2.7818 | 2.4867 | 4.1177 | 8.7068 | 0.0000 | 2.7829 | 2.7829 | nincs adat | nincs adat | nincs adat | 144.1094 |
| 1000.0000 | 100.0000 | 367.6832 | 2.7283 | 2.5071 | 3.7326 | 6.3900 | 0.0000 | 2.7294 | 2.7294 | nincs adat | nincs adat | nincs adat | 144.1094 |
| 5000.0000 | 1.0000 | 360.4787 | 2.7785 | 2.4691 | 3.9853 | 8.9070 | 0.0000 | 13.9020 | 2.7804 | nincs adat | nincs adat | nincs adat | 147.7344 |
| 5000.0000 | 10.0000 | 365.3518 | 2.7369 | 2.4413 | 3.7165 | 11.6372 | 0.0000 | 13.6904 | 2.7381 | nincs adat | nincs adat | nincs adat | 147.7344 |
| 5000.0000 | 50.0000 | 379.6935 | 2.6331 | 2.4225 | 3.7061 | 6.8241 | 0.0000 | 13.1701 | 2.6340 | nincs adat | nincs adat | nincs adat | 147.7344 |
| 5000.0000 | 100.0000 | 365.8740 | 2.7390 | 2.4700 | 3.7735 | 9.2190 | 0.0000 | 13.6998 | 2.7400 | nincs adat | nincs adat | nincs adat | 147.7344 |
| 10000.0000 | 1.0000 | 367.0604 | 2.7240 | 2.4574 | 3.8179 | 9.6894 | 0.0000 | 27.2595 | 2.7260 | nincs adat | nincs adat | nincs adat | 154.4844 |
| 10000.0000 | 10.0000 | 354.1051 | 2.8238 | 2.4944 | 4.0363 | 11.1820 | 0.0000 | 28.2504 | 2.8250 | nincs adat | nincs adat | nincs adat | 154.4844 |
| 10000.0000 | 50.0000 | 357.5803 | 2.7984 | 2.5093 | 4.0343 | 9.9173 | 0.0000 | 27.9944 | 2.7994 | nincs adat | nincs adat | nincs adat | 154.4844 |
| 10000.0000 | 100.0000 | 368.8521 | 2.7102 | 2.4541 | 3.7719 | 9.2445 | 0.0000 | 27.1122 | 2.7112 | nincs adat | nincs adat | nincs adat | 154.4844 |

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
