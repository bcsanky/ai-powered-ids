# 5. Implementáció

## 5.1 A prototípus áttekintése

A megvalósított rendszer egy laboratóriumi prototípus, amelynek célja a szabályalapú és autoencoder-alapú anomáliadetektálás összehasonlítható vizsgálata. A prototípus nem éles üzemi SOC rendszer, hanem reprodukálható mérési és demonstrációs környezet. Ennek megfelelően a hangsúly az adatfeldolgozási lánc, a modellalapú pontozás, a viszonyítási alapok, a riportkészítés és az értékelési kimenetek követhető előállításán van.

A rendszer címe és szakmai fókusza az „AI-alapú kiberfenyegetés-felderítő és -elemző rendszer”. A megvalósítás Wazuh komponensekre, Python-alapú adatfeldolgozásra, sklearn `MLPRegressor` alapú autoencoder modellre, valamint FastAPI és batch scoring komponensekre épül. A prototípus azt mutatja be, hogyan illeszthető egy gépi tanulási anomáliadetektáló komponens egy IDS/SIEM jellegű architektúrába, és milyen korlátok mellett értékelhető kontrollált környezetben.

A validált mérési ágak a következők: AE-Minimal, AE-Context, statisztikai baseline, szabályalapú proxy baseline és offline hibrid kiértékelés. A natív Wazuh baseline nem tekinthető validált eredménynek, mivel nem áll rendelkezésre címkézett natív Wazuh export.

## 5.2 Fejlesztési és futtatási környezet

A prototípus fő feldolgozási részei Pythonban készültek. Az adatépítés, tanítás, kiértékelés, ábragenerálás, batch scoring, riportkészítés és teljesítménymérés külön parancssori belépési pontokon keresztül futtatható. A végleges konfigurációk az `experiments/final/` könyvtárban találhatók, és YAML formátumban rögzítik a mérési ágak azonosítóit, bemeneti adatait, kimeneti könyvtárait, jellemzőkészletét és tanítási beállításait.

A Wazuh Manager, Indexer és Dashboard Docker Compose alapú laboratóriumi komponensként szerepel a rendszerben. A Wazuh szerepe az IDS/SIEM architektúra demonstrálása és a szabályalapú kiértékelési ág kontextusba helyezése. A natív Wazuh teljesítményméréshez külön, címkézett Wazuh export szükséges, amely jelen mérési állapotban nem áll rendelkezésre.

A FastAPI scoring szolgáltatás a validált AE-Minimal modellre épülő pontozási végpontot valósítja meg. Ha a modellfájl, az előfeldolgozó vagy a küszöbfájl nem érhető el, a szolgáltatás nem ad vissza helyettesítő anomáliapontszámot, hanem hibát jelez. A batch scoring parancssori alternatívát ad ugyanarra a pontozási logikára, ami akkor is bemutathatóvá teszi a feldolgozási láncot, ha a konténeres környezet nem fut.

## 5.3 Adatfeldolgozási lánc

Az adatfeldolgozás elsődleges bemenete a CIC-IDS2017 nyers CSV adathalmaza. Az `ml/src/build_dataset.py` modul feladata a nyers oszlopnevek kanonizálása, a szükséges jellemzők kiválasztása, a címke normalizálása, a hibás vagy nem értelmezhető numerikus értékek kezelése, valamint a tanító, validációs, kalibrációs és teszt adatrészek előállítása.

A címkézés benign és támadó minták elkülönítésére épül. A benign minták a normál viselkedés tanulásához szükségesek, míg a támadó minták a kalibrációban és a tesztelésben jelennek meg. A végleges AE-Minimal és AE-Context adatkészletek felosztása a validált `dataset_metadata.json` állományok alapján:

| Adatrész | Mintaszám |
|---|---:|
| Tanító adatrész | 1 589 922 |
| Validációs adatrész | 340 698 |
| Kalibrációs adatrész | 448 628 |
| Teszt adatrész | 448 628 |

A kalibrációs és teszt adatrész egyaránt 278 278 támadó és 170 350 benign mintát tartalmaz. A tanító és validációs adatrész benign mintákon alapul az autoencoder tanítási logikája szerint.

Az előfeldolgozás kizárólag a tanító adatrészen illeszkedik. A numerikus jellemzők `StandardScaler` transzformáción mennek keresztül, a kategorikus `protocol` mező pedig `OneHotEncoder(handle_unknown="ignore")` kódolással kerül a modell bemenetére. Az előfeldolgozó `preprocess.pkl` fájlként mentődik, így a későbbi tanítási, kiértékelési és scoring lépések ugyanazt a transzformációt használhatják.

## 5.4 Jellemzőkészletek

Az AE-Minimal mérési ág a CIC-IDS2017 flow adatok hét alapjellemzőjét használja:

- `destination_port`
- `flow_duration`
- `total_fwd_packets`
- `total_backward_packets`
- `flow_bytes_per_sec`
- `flow_packets_per_sec`
- `protocol`

Ez a jellemzőkészlet szándékosan szűk, mert a cél egy stabilan reprodukálható, könnyen értelmezhető autoencoder baseline kialakítása. A numerikus mezők a forgalom időtartamát, csomagszámát és sebességét írják le, a `protocol` mező pedig kategorikus hálózati protokoll-információt ad hozzá.

Az AE-Context mérési ág ugyanezt az alapjellemzőkészletet egyszerű, timestamp nélküli kontextusjellemzőkkel egészíti ki:

- `destination_port_frequency`
- `protocol_frequency`
- `is_rare_destination_port`
- `packet_ratio`
- `bytes_packets_ratio`

A port- és protokollgyakorisági jellemzők kizárólag a tanító adatrészből illesztett statisztikákon alapulnak. A validációs, kalibrációs és teszt adatrészben olyan port vagy protokoll esetén, amely a tanító adatrészben nem szerepelt, a gyakorisági érték `0.0`. Ez csökkenti annak kockázatát, hogy validációs vagy teszteloszlási információ kerüljön vissza a tanítási folyamatba.

Fontos lehatárolás, hogy az AE-Context kontextusjellemzői nem időablakos, nem hostalapú és nem CTI-alapú jellemzők. Nem számítanak például forráscímenkénti kapcsolatszámot, egyedi célpontszámot vagy indikátortalálatot. A prototípus jelen változata stabil, flow-alapú, timestamp nélküli bővítést valósít meg.

## 5.5 Autoencoder modell

Az autoencoder modell sklearn `MLPRegressor` implementációra épül. A tanítás során a modell benign minták alapján tanulja meg a normál forgalmi mintázatok rekonstrukcióját. A teszteléskor minden minta előfeldolgozott jellemzővektora bemenetként kerül a modellbe, majd a bemenet és a rekonstrukció közötti eltérésből anomáliapontszám számítható.

A magas rekonstrukciós hiba azt jelzi, hogy a minta a tanult benign mintázatokhoz képest nehezebben rekonstruálható. Ez nem önmagában bizonyító erejű támadási állítás, hanem anomáliadetektálási jelzés, amely küszöbölés után riasztási döntéssé alakítható.

A prototípus három küszöbölési stratégiát kezel:

- `fixed`: konfigurációban rögzített küszöb.
- `percentile_95`: validációs pontszámeloszlás 95. percentilise.
- `f1_optimum`: kalibrációs adatrészen kiválasztott F1-optimum.

Az autoencoder kimenetei közé tartozik a `metrics_summary.csv`, a mintaszintű `predictions.csv`, a `threshold_curve.csv`, valamint a legnagyobb rekonstrukciós hibát mutató jellemzők összesítése. Ez utóbbi magyarázhatósági kiegészítésként szolgál: nem teljes interpretálhatósági módszer, de jelzi, hogy egy adott pontszámhoz mely jellemzőcsoportok járultak hozzá leginkább.

## 5.6 Baseline és hibrid komponensek

A statisztikai baseline a tanítóhalmaz középpontjától mért távolság alapján képez anomáliapontszámot. Ez egyszerű, nem neurális viszonyítási alapként jelenik meg az autoencoder eredmények mellett. A cél nem egy fejlett IDS-modell kiváltása, hanem annak bemutatása, hogy azonos előfeldolgozási lánc mellett egy egyszerű statisztikai pontszám milyen eredményt ad.

A szabályalapú proxy baseline akkor használható, ha nincs megfelelő címkézett natív Wazuh export. A proxy a feldolgozott AE-Minimal adatokból Wazuh-szerű riasztási exportot állít elő, majd az `eval.py` Wazuh módján keresztül értékelhető. A proxy pontszáma a numerikus jellemzők legnagyobb abszolút standardizált értékére épül, küszöbe pedig a validációs adatrészből származik. Ez kontrollált flow-alapú szabályproxy, nem natív Wazuh teljesítménymérés.

A hibrid komponens az AE-Minimal és a szabályalapú proxy predikcióit kombinálja. A döntési szabály unió alapú: akkor ad pozitív jelzést, ha bármelyik komponens támadást jelez. A hibrid eredmény validált, de offline proxy-alapú kiértékelésként értelmezendő, mivel azonos sorrendű teszthalmaz-predikciókra épül, nem éles eseménykorrelációra.

## 5.7 Scoring szolgáltatás és batch scoring

A scoring szolgáltatás célja a validált AE-Minimal modell eseményszintű pontozásának bemutatása. A FastAPI `/score` végpont bemenete egy eseményazonosítót, a szükséges flow jellemzőket, valamint opcionális szabályalapú jelzéseket tartalmaz. A kimenet többek között az alábbi mezőket adja vissza:

- `anomaly_score`
- `threshold_name`
- `threshold_value`
- `ml_alert`
- `rule_flag`
- `rule_level`
- `risk_level`
- `reason`

A `risk_level` értéke a modellalapú riasztás és a szabályalapú jelzések kombinációjából áll elő. A `normal`, `medium`, `high` és `critical` kategóriák célja nem incidensminősítés automatizálása, hanem a demonstrációs riport és a lab/replay esettanulmányok priorizálható megjelenítése.

A batch scoring script JSONL és CSV bemenetet támogat. Ugyanazt a modell-, előfeldolgozó- és küszöblogikát használja, mint a szolgáltatás, de parancssorból futtatható. Hiányzó modellfájl, előfeldolgozó vagy küszöb esetén hibával áll le, és nem állít elő helyettesítő pontszámot.

## 5.8 Riport és dashboard-jellegű összefoglaló

A riportkészítési komponens a mérési összehasonlításból és az eseményszintű scoring kimenetekből szakértői jelentést állít elő. A `security_report.md` és `security_report.html` a validált konfigurációk listáját, a fő összehasonlító metrikákat, a scoring események magas kockázatú elemeit, valamint a legfontosabb korlátokat foglalja össze. A `dashboard_summary.csv` tömör, táblázatos nézetet ad a riportban szereplő fő értékekről.

Ez a komponens dashboard jellegű demonstrációs réteg, nem éles incidensjelentés. A cél annak bemutatása, hogy a modell- és baseline-eredmények hogyan alakíthatók szakértői értelmezést támogató kimenetté.

## 5.9 Lab/replay demonstráció

A lab/replay demonstráció kontrollált eseménysorral mutatja be a scoring lánc működését. A bemenet négy szcenáriót tartalmaz:

- `benign_activity`
- `port_scan`
- `ssh_bruteforce`
- `combined_suspicious`

Az események célja a normál forgalom, port scan jellegű mintázat, SSH brute force jellegű viselkedés és kombinált gyanús helyzet bemutatása. Ezek demonstrációs események, nem natív Wazuh exportból származó bizonyítékok és nem éles SOC-validációs adatok.

A lab/replay kimenetek közé tartozik a pontozott eseménytábla, a szcenáriónkénti összefoglaló, az esettanulmányos Markdown jelentések, a timeline ábra és a kockázati szint eloszlását bemutató ábra. Ezek elsősorban az 5. fejezet implementációs és működési bemutatójához használhatók.

## 5.10 Teljesítménymérési komponens

A teljesítménymérési komponens a batch scoring feldolgozási lánc lokális/labor teljesítményét méri. A benchmark több eseményszám és több batch size mellett fut, miközben ugyanazt a már betöltött AE-Minimal modellt és előfeldolgozót használja. A vizsgált eseményszámok: 100, 500, 1000 és 5000. A vizsgált batch size értékek: 1, 10, 50 és 100.

A rögzített metrikák:

- `events_per_second`
- `avg_latency_ms`
- `p50_latency_ms`
- `p95_latency_ms`
- `p99_latency_ms`
- `failed_events`

A mérés során a bemeneti lab/replay események determinisztikus ismétlése történik. Ez a scoring feldolgozási költség mérésére szolgál, nem új detektálási minőségvizsgálatra. Az eredmény hardver- és környezetfüggő lokális/labor mérés, nem éles üzemi teljesítménygarancia és nem natív Wazuh indexelési benchmark.

## 5.11 Reprodukálhatóság

A reprodukálhatóság alapját a végleges YAML konfigurációk, a Makefile célok, a futtatási metaadatok és a dokumentált validációs lépések adják. A mérési eredményekhez tartozó fő konfigurációk az `experiments/final/` könyvtárban találhatók. A futtatások `run_metadata.json`, `train_config.json` és küszöbinformációs állományokat tartalmaznak, amelyek segítik a mérési környezet visszakövetését.

A verziókezelésben a forráskód, konfigurációk, tesztek és dokumentációk maradnak. A nagy méretű futási eredmények, modellfájlok, nagy CSV-k és PNG ábrák a beadási ZIP mellékletben őrizhetők meg. A dolgozatba beemelendő táblázatokhoz és ábrákhoz külön eredményjegyzék rögzíti, hogy mely futtatási könyvtárakból és kimeneti állományokból származnak.

A fejezetbe illesztendő ábra- és táblázathivatkozások:

- [IDE KERÜL: 5.1 ábra, prototípus architektúrája]
- [IDE KERÜL: 5.2 ábra, adatfeldolgozási lánc]
- [IDE KERÜL: 5.3 ábra, scoring és riportgenerálási folyamat]
- [IDE KERÜL: 5.1 táblázat, implementált komponensek]
