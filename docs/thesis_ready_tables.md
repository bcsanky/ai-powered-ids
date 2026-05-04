# Dolgozatba másolható táblázatok

## 1. Implementált komponensek

| Komponens | Megvalósítás | Bemenet | Kimenet | Dolgozatbeli szerep |
|---|---|---|---|---|
| Adatépítés | Python alapú CIC-IDS2017 feldolgozás | Nyers CSV fájlok | Train, validation, calibration és test adatrészek | 5. fejezet adatfeldolgozási leírása |
| AE-Minimal | `MLPRegressor` alapú autoencoder | Minimális flow jellemzők | Anomáliapontszám, predikciók, metrikák | Fő gépi tanulási mérési ág |
| AE-Context | Autoencoder kontextusjellemzőkkel | Flow jellemzők és egyszerű kontextusjellemzők | Anomáliapontszám, predikciók, metrikák | Kontextusos mérési ág |
| Baseline_stat | Távolságalapú statisztikai baseline | Feldolgozott AE-Minimal adatok | Baseline metrikák | Egyszerű viszonyítási alap |
| Rule_proxy | Szabályalapú proxy baseline | Feldolgozott flow adatok | Wazuh-szerű riasztási export és metrikák | Natív Wazuh export hiányának kontrollált pótlása |
| Hybrid | Offline unió alapú döntés | AE-Minimal és rule_proxy predikciók | Hibrid metrikák | Kontrollált offline összehasonlítás |
| Scoring szolgáltatás | FastAPI `/score` végpont | Eseményjellemzők | Anomáliapontszám és kockázati szint | Működési demonstráció |
| Batch scoring | Parancssori scoring JSONL/CSV bemenetre | Demonstrációs események | Pontozott események | Reprodukálható demonstráció |
| Szakértői jelentés | Markdown/HTML riport | Metrikák és pontozott események | Szakértői jelentés és dashboard-jellegű összefoglaló | Riportkészítési réteg |

## 2. Feature-készletek összehasonlítása

| Jellemző | AE-Minimal | AE-Context | Típus | Előfeldolgozás |
|---|---|---|---|---|
| `destination_port` | igen | igen | numerikus | `StandardScaler` |
| `flow_duration` | igen | igen | numerikus | `StandardScaler` |
| `total_fwd_packets` | igen | igen | numerikus | `StandardScaler` |
| `total_backward_packets` | igen | igen | numerikus | `StandardScaler` |
| `flow_bytes_per_sec` | igen | igen | numerikus | `StandardScaler` |
| `flow_packets_per_sec` | igen | igen | numerikus | `StandardScaler` |
| `protocol` | igen | igen | kategorikus | `OneHotEncoder` |
| `destination_port_frequency` | nem | igen | numerikus kontextusjellemző | tanító adatrészből illesztett gyakoriság, majd skálázás |
| `protocol_frequency` | nem | igen | numerikus kontextusjellemző | tanító adatrészből illesztett gyakoriság, majd skálázás |
| `is_rare_destination_port` | nem | igen | bináris kontextusjellemző | tanító adatrészből illesztett ritkasági szabály |
| `packet_ratio` | nem | igen | numerikus arány | soronként számított érték, majd skálázás |
| `bytes_packets_ratio` | nem | igen | numerikus arány | soronként számított érték, majd skálázás |

## 3. Konfigurációk és státuszuk

| Konfiguráció | Státusz | Értelmezés |
|---|---|---|
| AE-Minimal | validált | Fő autoencoder mérési ág |
| AE-Context | validált | Timestamp nélküli kontextusjellemzőkkel bővített autoencoder |
| Baseline_stat | validált | Egyszerű statisztikai baseline |
| Rule_proxy | validált | Kontrollált flow-alapú szabályproxy |
| Hybrid | validált, offline proxy-alapú | AE-Minimal és rule_proxy predikciók uniója |
| Natív Wazuh baseline | hiányzik | Nincs címkézett natív Wazuh export |

## 4. Fő metrikák konfigurációnként

| Konfiguráció | Státusz | Precision | Recall | F1 | FPR | FNR | Alert count | ROC-AUC | Mintaszám |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AE-Minimal | ok | 0,620287 | 1,000000 | 0,765651 | 1,000000 | 0,000000 | 448 628 | 0,605370 | 448 628 |
| AE-Context | ok | 0,620290 | 1,000000 | 0,765653 | 0,999988 | 0,000000 | 448 626 | 0,764273 | 448 628 |
| Baseline_stat | ok | 0,514707 | 0,032008 | 0,060267 | 0,049299 | 0,967992 | 17 305 | 0,475816 | 448 628 |
| Rule_proxy | ok | 0,525269 | 0,033391 | 0,062791 | 0,049299 | 0,966609 | 17 690 | 0,573446 | 448 628 |
| Hybrid | ok | 0,620287 | 1,000000 | 0,765651 | 1,000000 | 0,000000 | 448 628 | 0,640483 | 448 628 |
| Natív Wazuh baseline | hiányzik | [KITÖLTENDŐ: nincs validált export] | [KITÖLTENDŐ] | [KITÖLTENDŐ] | [KITÖLTENDŐ] | [KITÖLTENDŐ] | [KITÖLTENDŐ] | [KITÖLTENDŐ] | [KITÖLTENDŐ] |

## 5. Lab/replay esettanulmányok

| Szcenárió | Esemény | Normal | Medium | High | Critical | Átlagos pontszám | Max. pontszám | ML riasztás | Szabályriasztás |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| benign_activity | 5 | 0 | 0 | 5 | 0 | 0,402420 | 0,449409 | 5 | 0 |
| combined_suspicious | 5 | 0 | 0 | 0 | 5 | 0,326466 | 0,449258 | 5 | 5 |
| port_scan | 5 | 0 | 0 | 0 | 5 | 0,344663 | 0,448297 | 5 | 5 |
| ssh_bruteforce | 5 | 0 | 0 | 0 | 5 | 0,451053 | 0,451846 | 5 | 5 |

## 6. Performance összefoglaló

| Mutató | Érték |
|---|---:|
| Vizsgált eseményszámok | 100, 500, 1000, 5000, 10000 |
| Vizsgált batch size értékek | 1, 10, 50, 100 |
| Benchmark sorok száma | 60 |
| Legjobb áteresztőképesség | 415,65 esemény/másodperc |
| Legjobb throughput batch size | 100 |
| Legjobb throughput eseményszám | 500 |
| Legalacsonyabb p95 késleltetés | 2,5994 ms |
| Hibás események összesen | 0 |
| Legalacsonyabb CPU-idő eseményenként | 2,4057 ms |
| Legnagyobb csúcsmemória | 154,4844 MB |

## 7. Korlátok és kezelésük

| Korlát | Hatás | Kezelés a dolgozatban |
|---|---|---|
| CIC-IDS2017 reprezentativitás | Nem általánosítható minden hálózatra | Laboratóriumi érvényesség hangsúlyozása |
| Wazuh log és CIC flow eltérés | Natív Wazuh mérés nem igazolható export nélkül | Rule_proxy külön, korlátozott baseline-ként szerepel |
| Magas false positive arány | Nagy riasztási terhelés | FPR és alert count explicit értelmezése |
| Hybrid offline proxy-alapú | Nem éles eseménykorreláció | Offline összehasonlításként megfogalmazva |
| Replay kis elemszám | Nem statisztikai benchmark | Demonstrációs validációként kezelve |
| Teljesítménymérés hardverfüggése | Nem éles üzemi garancia | Lokális batch scoring mérésként megfogalmazva |
