# Batch scoring teljesítménymérési összefoglaló

A mérés a meglévő AE-Minimal scoring lánc lokális/labor teljesítményét vizsgálja. Az eseményszám növelése determinisztikus ismétléssel történt, kizárólag a feldolgozási idő méréséhez.

## Fő eredmények

- Legjobb áteresztőképesség: 415.65 esemény/másodperc, batch size 100, eseményszám 500.
- Legalacsonyabb p95 késleltetés: 2.5994 ms, batch size 100, eseményszám 500.
- Hibás események összesen: 0.
- Legalacsonyabb CPU-idő eseményenként: 2.4057 ms.
- Legnagyobb mért csúcsmemória: 154.4844 MB.

## Erőforrás-mérés

A CPU-idő mérése `time.process_time()` alapján történik. A memória RSS érték psutil jelenléte esetén érhető el; Unix/Linux környezetben a csúcsmemória `resource.getrusage()` alapján is rögzíthető. Ha egy memóriaadat nem érhető el az adott platformon, az oszlop üresen maradhat.

## Korlát

Ez lokális/labor mérés, nem éles üzemi teljesítménygarancia, nem hosszú idejű SOC-terhelés, és nem natív Wazuh indexelési teljesítménymérés.
