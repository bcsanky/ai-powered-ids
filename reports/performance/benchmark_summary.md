# Batch scoring teljesítménymérési összefoglaló

A mérés a meglévő AE-Minimal scoring lánc lokális/labor teljesítményét vizsgálja. Az eseményszám növelése determinisztikus ismétléssel történt, kizárólag a feldolgozási idő méréséhez.

## Fő eredmények

- Legjobb áteresztőképesség: 405.51 esemény/másodperc, batch size 50, eseményszám 100.
- Legalacsonyabb p95 késleltetés: 2.7938 ms, batch size 10, eseményszám 100.
- Hibás események összesen: 0.
- Legalacsonyabb CPU-idő eseményenként: 2.4661 ms.
- Legnagyobb mért csúcsmemória: 147.9062 MB.

## Erőforrás-mérés

A CPU-idő mérése `time.process_time()` alapján történik. A memória RSS érték psutil jelenléte esetén érhető el; Unix/Linux környezetben a csúcsmemória `resource.getrusage()` alapján is rögzíthető. Ha egy memóriaadat nem érhető el az adott platformon, az oszlop üresen maradhat.

## Korlát

Ez lokális/labor mérés, nem éles üzemi teljesítménygarancia, nem hosszú idejű SOC-terhelés, és nem natív Wazuh indexelési teljesítménymérés.
