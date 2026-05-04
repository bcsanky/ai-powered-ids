# Batch scoring teljesítménymérési összefoglaló

A mérés a meglévő AE-Minimal scoring lánc lokális/labor teljesítményét vizsgálja. Az eseményszám növelése determinisztikus ismétléssel történt, kizárólag a feldolgozási idő méréséhez.

## Fő eredmények

- Legjobb áteresztőképesség: 399.04 esemény/másodperc, batch size 1, eseményszám 100.
- Legalacsonyabb p95 késleltetés: 2.7723 ms, batch size 1, eseményszám 100.
- Hibás események összesen: 0.

## Korlát

Ez lokális/labor mérés, nem éles üzemi benchmark, nem hosszú idejű SOC-terhelés, és nem natív Wazuh indexelési teljesítménymérés.
