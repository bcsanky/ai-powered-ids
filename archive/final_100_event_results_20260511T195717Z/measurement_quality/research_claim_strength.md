# Kutatási állítás erősségének ellenőrzése

Összesített státusz: **PASS**

A minősítés kizárólag a meglévő real-lab comparison metrikákból következik.

| Ellenőrzés | Kategória | Státusz | Üzenet | Javaslat | Érték |
|---|---|---|---|---|---|
| metrics_available | Kutatási állítás | PASS | comparison metrikák elérhetők | Futtasd a real comparison pipeline-t tényleges mérés után. |  |
| best_hybrid_by_f1 | Kutatási állítás | PASS | legjobb hibrid F1 alapján |  | Hybrid OR |
| f1_delta_vs_wazuh | Kutatási állítás | PASS | F1 delta Wazuh-onlyhoz képest |  | 0.2201 |
| claim_category | Kutatási állítás | PASS | a vizsgált lab mérés alapján inkább recall és riasztási terhelés közötti kompromisszum látszik |  | TRADEOFF_ONLY |
