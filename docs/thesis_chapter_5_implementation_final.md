# 5. Implementáció

## 5.1 A prototípus áttekintése

A megvalósított rendszer laboratóriumi prototípus, amely az „AI-alapú kiberfenyegetés-felderítő és -elemző rendszer” mérnöki megvalósítását mutatja be. A prototípus célja Wazuh/SIEM jellegű környezet és autoencoder-alapú anomáliadetektálás együttes vizsgálata, valamint a szabályalapú és gépi tanulási megközelítések összehasonlítható értékelése.

A rendszer nem éles üzemi SOC-platform. A megvalósítás reprodukálható mérési és demonstrációs környezetként készült, amelyben a feldolgozási lánc, a tanítás, a kiértékelés, a scoring, a riportkészítés és a teljesítménymérés külön parancsokkal ismételhető. Ez a lehatárolás azért fontos, mert a diplomamunka célja nem egy teljes körű üzemeltetett IDS kiváltása, hanem annak vizsgálata, hogy a gépi tanulási anomáliadetektálás hogyan illeszthető egy IDS/SIEM jellegű architektúrába.

[IDE KERÜL: 5.1 ábra – Prototípus architektúrája]

## 5.2 Fejlesztési és futtatási környezet

A feldolgozási lánc Python-alapú. Az adatépítés, az autoencoder tanítása, a baseline kiértékelés, az ábrák előállítása, a batch scoring, a szakértői jelentés készítése és a lokális teljesítménymérés külön modulokban és Makefile célokban jelenik meg. A végleges kísérleti konfigurációk az `experiments/final/` könyvtárban találhatók.

A laboratóriumi IDS/SIEM környezet Wazuh Manager, Wazuh Indexer és Wazuh Dashboard komponensekre épül. A Wazuh szerepe a prototípusban architekturális és összehasonlítási: bemutatja, hogyan kapcsolódhat a gépi tanulási komponens egy SIEM jellegű rendszerhez. Natív Wazuh teljesítménymérés csak címkézett Wazuh exporttal lenne igazolható; ilyen export a validált mérési körben nem állt rendelkezésre.

A scoring réteg két formában készült el. A FastAPI szolgáltatás `/score` végpontja modellalapú pontozást végez, ha a szükséges modellfájl, előfeldolgozó és küszöbfájl elérhető. Emellett a batch scoring parancssori feldolgozást biztosít CSV vagy JSONL bemenetekre, ami a diplomamunka demonstrációs és riportkészítési részeiben is használható.

[IDE KERÜL: 5.1 táblázat – Implementált komponensek]

## 5.3 Adatfeldolgozási lánc

Az adatfeldolgozás bemenete a CIC-IDS2017 nyers CSV adathalmaza. Az adatépítési lépés először egységesíti az oszlopneveket, majd a szükséges flow jellemzőket kanonikus névre képezi le. A címke benign és attack osztályokra normalizálódik.

Az adatok tanító, validációs, kalibrációs és teszt adatrészre bomlanak. A validált futásban a tanító adatrész 1 589 922 mintát, a validációs adatrész 340 698 mintát, a kalibrációs és teszt adatrész pedig külön-külön 448 628 mintát tartalmazott. A teszt adatrészben 278 278 támadó és 170 350 benign minta szerepelt.

Az előfeldolgozó kizárólag a tanító adatrészen illeszkedik. A numerikus mezők `StandardScaler` transzformációt kapnak, a `protocol` kategorikus mező pedig `OneHotEncoder(handle_unknown="ignore")` kódolással kerül a modell bemenetére. Az illesztett előfeldolgozó `preprocess.pkl` fájlként mentődik, így ugyanaz a transzformáció használható a tanítás, a kiértékelés és a scoring során.

[IDE KERÜL: 5.2 ábra – Adatfeldolgozási lánc]

## 5.4 Jellemzőkészletek

Az AE-Minimal mérési ág hét alapvető flow jellemzőt használ: `destination_port`, `flow_duration`, `total_fwd_packets`, `total_backward_packets`, `flow_bytes_per_sec`, `flow_packets_per_sec` és `protocol`. Ez a jellemzőkészlet szándékosan szűk, hogy stabilan reprodukálható és könnyen értelmezhető autoencoder konfigurációt adjon.

Az AE-Context ugyanebből a minimális jellemzőkészletből indul ki, és timestamp nélküli kontextusjellemzőkkel egészíti ki. Az implementált kontextusjellemzők: `destination_port_frequency`, `protocol_frequency`, `is_rare_destination_port`, `packet_ratio` és `bytes_packets_ratio`. A port- és protokollgyakoriságok kizárólag a tanító adatrészből illeszkednek, így validációs vagy teszteloszlási információ nem kerül vissza a tanításba.

Ezek a kontextusjellemzők nem időablakos, nem hostalapú és nem CTI-alapú jellemzők. Nem számítanak például forráscímenkénti kapcsolatszámot, sikertelen bejelentkezési számot vagy indikátortalálatot. A bővítés célja egy stabil, CIC-IDS2017 flow adatokon működő kontextusos mérési ág létrehozása volt.

[IDE KERÜL: 5.2 táblázat – Feature-készletek összehasonlítása]

## 5.5 Autoencoder modell

Az anomáliadetektáló modell sklearn `MLPRegressor` alapú autoencoder. A modell benign mintákon tanulja meg a normál forgalom rekonstrukcióját. Teszteléskor a bemenet és a rekonstrukció közötti eltérés adja az anomáliapontszámot: magasabb rekonstrukciós hiba erősebb eltérést jelez a tanult normál mintázattól.

A prototípus három küszöbstratégiát támogat: `fixed`, `percentile_95` és `f1_optimum`. A `fixed` konfigurációban rögzített értéket használ, a `percentile_95` a validációs pontszámeloszlás percentilisén alapul, az `f1_optimum` pedig a kalibrációs adatrészen választ F1 szempontból kedvező küszöböt.

Az autoencoder eredményeihez magyarázhatósági kiegészítés is készül: a top feature reconstruction error azt mutatja, hogy az adott mintánál mely jellemzőcsoportok járultak hozzá leginkább a rekonstrukciós hibához. Ez nem teljes interpretálhatósági módszer, de segíti az anomáliapontszám szakmai értelmezését.

## 5.6 Baseline és hibrid komponensek

A statisztikai viszonyítási alap (baseline) a tanítóhalmaz középpontjától mért távolság alapján képez anomáliapontszámot. Ez egyszerű, nem neurális referencia, amely segít értelmezni, hogy az autoencoder mennyiben tér el egy egyszerűbb pontozási módszertől.

A rule_proxy kontrollált szabályalapú proxy baseline. Akkor használható, ha nincs címkézett natív Wazuh export. A proxy a feldolgozott flow adatokból Wazuh-szerű riasztási mezőket állít elő, de nem tekinthető natív Wazuh teljesítménymérésnek. A validált mérési körben a natív Wazuh ág hiányzóként szerepel.

A hibrid komponens az AE-Minimal és a rule_proxy predikcióit kombinálja. A hibrid döntés offline, azonos sorrendű teszthalmaz-predikciókra épül: ha bármelyik komponens támadást jelez, a hibrid predikció is pozitív lesz. Ez kontrollált offline összehasonlítás, nem éles eseménykorreláció.

[IDE KERÜL: 5.3 táblázat – Futtatási konfigurációk]

## 5.7 Scoring szolgáltatás és batch scoring

A scoring szolgáltatás célja, hogy a validált AE-Minimal modell eseményszintű pontozása szolgáltatásként is bemutatható legyen. A `/score` endpoint bemenete egy eseményazonosítót, a szükséges flow jellemzőket, valamint opcionális szabályalapú jelzéseket tartalmaz.

A kimenet többek között az `anomaly_score`, `threshold_name`, `threshold_value`, `ml_alert`, `risk_level` és `reason` mezőket adja vissza. Hiányzó modell, előfeldolgozó vagy küszöbfájl esetén a szolgáltatás nem ad helyettesítő pontszámot, hanem hibát jelez. Ez azért fontos, mert a demonstráció nem kelthet hamis biztonsági vagy működési benyomást.

A batch scoring ugyanezt a pontozási logikát teszi elérhetővé parancssorból. JSONL és CSV bemenetet támogat, és a lab/replay demonstráció, valamint a szakértői jelentés bemeneteként is használható.

[IDE KERÜL: 5.3 ábra – Batch scoring és riportgenerálási folyamat]

## 5.8 Riport és dashboard-jellegű összefoglaló

A riportkészítési komponens a mérési összehasonlítás és a pontozott események alapján szakértői jelentést készít Markdown és HTML formátumban. A `security_report.md` és `security_report.html` a validált konfigurációk fő metrikáit, a pontozott események magasabb kockázati szintjeit, valamint a legfontosabb korlátokat foglalja össze.

A `dashboard_summary.csv` tömör táblázatos formában tartalmazza a riport fő adatait. Ez a réteg dashboard-jellegű összefoglalóként értelmezhető, nem éles incidensjelentésként.

## 5.9 Lab/replay demonstráció

A lab/replay demonstráció kontrollált eseménysorral mutatja be a scoring lánc működését. A szcenáriók: `benign_activity`, `port_scan`, `ssh_bruteforce` és `combined_suspicious`. A demonstráció célja a kockázati szintek, az anomáliapontszámok és a riportkészítés szemléltetése.

A lab/replay eredményekhez timeline ábra, kockázati szint eloszlási ábra, szcenárióösszesítés és esettanulmányos jelentés készült. A demonstráció nem natív Wazuh export és nem éles SOC-validáció, hanem kontrollált eseménysor a prototípus működésének bemutatására.

## 5.10 Teljesítménymérési komponens

A teljesítménymérési komponens lokális batch scoring/inference mérést végez. A mérés 100, 500, 1000, 5000 és 10000 esemény feldolgozását vizsgálja 1, 10, 50 és 100 batch size mellett, három ismétléssel. A cél a scoring komponens feldolgozási költségének becslése.

A rögzített metrikák közé tartozik az áteresztőképesség, az átlagos késleltetés, a p50, p95 és p99 késleltetés, a processz CPU-idő, a CPU-idő eseményenként, a platformfüggően elérhető memóriaértékek és a hibás események száma. Ez nem event/perc alapú replay mérés, nem éles üzemi teljesítménygarancia, és nem teljes SIEM/Wazuh end-to-end terhelhetőségi mérés.

## 5.11 Reprodukálhatóság és mellékletek

A reprodukálhatóságot a végleges YAML konfigurációk, a Makefile célok, a futtatási metaadatok, a validációs dokumentumok és az eredményjegyzék biztosítják. A nagy méretű modell- és mérési állományok nem feltétlenül részei a Git verziókezelésnek; ezek a beadási ZIP mellékletben adhatók át.

A Word dokumentumba beemelendő fő állományok a végleges fejezetszövegek, a kész táblázatok, a dolgozatba rendezett ábrák, valamint a validált mérési eredményekhez tartozó CSV-k. A végleges PDF előtt minden `[KITÖLTENDŐ]` jelölést, ábra- és táblázathivatkozást ellenőrizni kell.
