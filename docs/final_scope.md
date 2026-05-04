# Végleges kísérleti lehatárolás

## 1. Rövid célmegfogalmazás

A szakdolgozat címe: **„AI-alapú kiberfenyegetés-felderítő és -elemző rendszer”**.

A munka célja egy olyan laboratóriumi prototípus megtervezése, megvalósítása és értékelése, amely hagyományos IDS/SIEM komponenseket egészít ki gépi tanuláson alapuló anomáliadetektálással. A prototípus Wazuh komponensekre, egy saját FastAPI alapú ML szolgáltatás-prototípus komponensre, valamint a CIC-IDS2017 adathalmazon végzett kísérletekre épül. A gépi tanulási komponens egy sklearn `MLPRegressor` alapú autoencoder, amely rekonstrukciós hibából számít anomáliapontszámot.

A cél nem egy éles üzemi IDS teljes körű kiváltása, hanem annak vizsgálata, hogy egy hibrid, szabályalapú és gépi tanulási megközelítés milyen módon illeszthető egy SIEM/IDS architektúrába, és milyen mérőszámokkal értékelhető kontrollált, reprodukálható kísérleti környezetben.

## 2. A végleges laboratóriumi prototípus pontos lehatárolása

A végleges laboratóriumi prototípus egy reprodukálható kísérleti rendszer, amely az alábbi fő elemekből áll:

- Wazuh Manager, Wazuh Indexer és Wazuh Dashboard Docker Compose alapú laboratóriumi környezetben.
- Egy saját FastAPI alapú ML szolgáltatás-prototípus komponens, amely az architekturális integrációs pontot reprezentálja.
- CIC-IDS2017 adatfeldolgozási lánc, amely nyers CSV fájlokból tanító, validációs, kalibrációs és teszt adathalmazokat állít elő.
- Autoencoder alapú anomáliadetektáló modell sklearn `MLPRegressor` implementációval.
- Rekonstrukciós hiba alapú anomáliapontszám.
- Három küszöbölési stratégia: fix küszöb, validációs percentilis alapú küszöb és kalibrációs halmazon optimalizált F1-küszöb.
- Statisztikai viszonyítási alap (baseline), amely a tanítóhalmaz középpontjától mért távolság alapján képez anomáliapontszámot.
- Wazuh-stílusú baseline, amely exportált Wazuh riasztásokat vagy Wazuh-szerű predikciós mezőket hasonlít össze a címkézett adatokkal.
- Szabályalapú proxy baseline, amely kontrollált offline kiértékelésben Wazuh-szerű exportot állít elő a feldolgozott CIC-IDS2017 flow adatokból.
- Offline hibrid kiértékelés, amely az AE-Minimal és a szabályalapú proxy baseline predikcióit azonos teszthalmaz-sorrend mellett kombinálja.
- Eredményfájlok, mérőszámok, AE ábrák és összehasonlító ábrák előállítása a szakdolgozati értékeléshez.

A laboratóriumi prototípus a detektálási és értékelési láncot demonstrálja. A hangsúly a reprodukálható mérési láncon, a konfigurációk összehasonlíthatóságán és a korlátok világos megnevezésén van.

## 3. Megvalósítási kör május 15-ig

A szakdolgozati prototípus lezárásáig az alábbi elemek tartoznak a megvalósítási körbe:

- A végleges kísérleti konfigurációk rögzítése az `experiments/final/` könyvtárban.
- A CIC-IDS2017 adathalmazból előállított feldolgozott adatszeletek létrehozása.
- Az `ae_minimal` konfiguráció teljes futtatása, beleértve az adatépítést, a modell tanítását, a küszöbök számítását és a tesztkiértékelést.
- Az `ae_context` konfiguráció futtatása egyszerű, timestamp nélküli kontextusjellemzőkkel.
- A `baseline_stat` konfiguráció futtatása és eredményeinek összehasonlítása az autoencoder eredményeivel.
- A `rule_proxy` konfiguráció futtatása kontrollált, flow-alapú szabályproxyként.
- A `baseline_wazuh` konfiguráció előkészítése natív Wazuh vagy Wazuh-szerű exportált predikciók kiértékelésére, ha ilyen címkézett export rendelkezésre áll.
- A `hybrid` konfiguráció futtatása offline kiértékelésként, az AE-Minimal és a rule_proxy predikciók kombinálásával.
- A fő mérőszámok táblázatos exportja: precision, recall, F1, false positive rate, confusion matrix és alert count.
- A szakdolgozathoz felhasználható ábrák és táblázatok előállítása: küszöbgörbe, ROC-görbe, pontszámeloszlás, konfúziós mátrix, legfontosabb jellemzők gyakorisága és összehasonlító metrikaábrák, ahol értelmezhető.
- A kísérleti lépések rövid futtatási dokumentációja és az eredmények értelmezése.

## 4. Mit nem valósítunk meg, és miért nem

A végleges laboratóriumi prototípus nem vállal teljes éles üzemi IDS implementációt. Ennek oka, hogy a szakdolgozat időkerete és a laboratóriumi validációs környezet nem teszi lehetővé egy termelési környezetben hosszú ideig futó, teljes körűen üzemeltetett detektáló rendszer megbízható értékelését.

Nem valósítunk meg teljes körű online tanulást vagy automatikus modellfrissítést. A concept drift kezeléséhez hosszabb idejű, időben változó valós forgalmi adatokra és külön validációs metodikára lenne szükség.

Nem valósítunk meg teljes körű CTI integrációt, például STIX/TAXII feedek automatikus feldolgozását, indikátorok életciklus-kezelését vagy több forrásból származó threat intelligence korrelációját. A CTI kezelés a prototípusban egyszerűsített, koncepcionális elemként jelenik meg.

Nem cél mély Wazuh szabálykészlet-fejlesztés vagy egyedi Wazuh szabályfejlesztési kampány végrehajtása. A Wazuh komponens elsősorban IDS/SIEM architekturális baseline és integrációs környezet.

Nem cél a CIC-IDS2017 flow jellemzőinek teljes megfeleltetése valós Wazuh logmezőknek. A két reprezentáció eltérő adatmodellt használ: a CIC-IDS2017 hálózati flow-jellemzőket tartalmaz, míg a Wazuh esemény- és logorientált adatokat kezel.

Nem állítjuk, hogy a szabályalapú proxy baseline natív Wazuh teljesítménymérés lenne. Ez kontrollált, flow-alapú proxy, amely a Wazuh-szerű kiértékelési útvonal használhatóságát és egy egyszerű szabályalapú referencia viselkedését mutatja be.

Nem valósítunk meg nagy skálájú teljesítménytesztet, magas rendelkezésre állású üzemeltetést, jogosultságkezelési auditot vagy éles üzemi megerősítést. Ezek fontos mérnöki feladatok, de túlmutatnak a szakdolgozat kísérleti fókuszán.

## 5. A végleges összehasonlított konfigurációk

### baseline_stat

A `baseline_stat` egy egyszerű statisztikai anomáliadetektáló baseline. A feldolgozott jellemzőtérben kiszámítja a tanítóhalmaz középpontját, majd a tesztminták ehhez viszonyított távolságából képez anomáliapontszámot. A küszöböt a validációs pontszámok percentilise alapján állítja be.

Ez a konfiguráció nem tekinthető fejlett IDS-nek, de hasznos referenciaérték: megmutatja, hogy egy egyszerű, nem neurális módszer milyen teljesítményt ér el ugyanazon adatelőkészítési folyamat mellett.

### baseline_wazuh

A `baseline_wazuh` a Wazuh vagy Wazuh-szerű riasztási eredmények kiértékelésére szolgál. A bemenet egy exportált fájl, amely tartalmazza a valós címkét, valamint a Wazuh riasztási vagy predikciós mezőit.

A cél nem annak állítása, hogy a Wazuh natívan ugyanazokat a CIC-IDS2017 flow jellemzőket használja, mint az autoencoder, hanem egy szabályalapú vagy riasztásalapú baseline beemelése az összehasonlításba. Az eredmények értelmezésénél figyelembe kell venni a logalapú és flow-alapú adatmodell közötti eltérést.

### rule_proxy

A `rule_proxy` egy szabályalapú proxy baseline. A feldolgozott AE-Minimal adatok numerikus jellemzőin soronként a legnagyobb abszolút standardizált értéket használja pontszámként. A küszöb a validációs adatrész 0,95 kvantilise alapján áll elő, ezért a teszt címkéi nem vesznek részt a szabály illesztésében.

A kimenet Wazuh-szerű szabályalapú export, amely tartalmaz riasztási mezőket, például `wazuh_alert`, `is_alert`, `y_pred`, `rule_level` és `score`. Ez kontrollált offline kiértékelés, nem natív Wazuh teljesítménymérés.

### ae_minimal

Az `ae_minimal` a fő autoencoder konfiguráció. A bemeneti jellemzőkészlet az implementált minimális jellemzőkből áll:

- `destination_port`
- `flow_duration`
- `total_fwd_packets`
- `total_backward_packets`
- `flow_bytes_per_sec`
- `flow_packets_per_sec`
- `protocol`

A numerikus jellemzőket standard skálázás, a kategorikus protokollmezőt one-hot encoding alakítja át. Az autoencoder csak benign mintákon tanul, a tesztelés pedig benign és támadó mintákat egyaránt tartalmaz. A detektálás alapja a rekonstrukciós hiba.

### ae_context

Az `ae_context` a minimális flow jellemzőkészletet egyszerű, timestamp nélküli kontextusjellemzőkkel egészíti ki. Az implementált kontextusjellemzők tanító adatrészből illesztett port- és protokollgyakoriságon, ritka célport jelzőn, valamint soronként számított forgalmi arányokon alapulnak:

- `destination_port_frequency`
- `protocol_frequency`
- `is_rare_destination_port`
- `packet_ratio`
- `bytes_packets_ratio`

Ez a megközelítés lehetőséget ad annak bemutatására, hogyan bővíthető a rendszer egyszerű kontextussal úgy, hogy a jellemzőképzés stabilan működjön a CIC-IDS2017 flow adatokon. A gyakorisági térképek nem használják a validációs, kalibrációs vagy teszt adatrész eloszlását. Fontos korlát, hogy ezek nem időablakos, hostalapú vagy CTI-alapú kontextusjellemzők.

### hybrid

A `hybrid` konfiguráció az AE-Minimal és a szabályalapú proxy baseline predikcióit kombinálja kontrollált offline kiértékelésben. A döntési szabály szerint akkor keletkezik riasztás, ha az AE-Minimal vagy a rule_proxy komponens támadást jelez:

```text
hybrid_pred = ae_pred == 1 OR rule_pred == 1
```

A hibrid pontszám a normalizált AE-pontszám és a normalizált rule_proxy pontszám maximuma. Fontos korlát, hogy ez a hibrid kiértékelés azonos teszthalmaz-sorrendre épül, nem éles eseménykorreláció.

## 6. Mérőszámok

Az összehasonlítás fő mérőszámai:

- **Precision**: a pozitívnak jelzett riasztások közül mennyi volt valóban támadás.
- **Recall**: a tényleges támadások mekkora részét találta meg a rendszer.
- **F1**: a precision és recall harmonikus átlaga, amely kiegyensúlyozott képet ad a két szempont között.
- **False positive rate**: a benign minták közül mekkora arányt jelzett tévesen támadásnak a rendszer.
- **Confusion matrix**: a true negative, false positive, false negative és true positive értékek táblázatos összefoglalása.
- **Alert count**: az előállított riasztások száma, amely az üzemeltetési terhelés becsléséhez fontos.

A mérőszámokat minden konfigurációnál azonos címkézési logika és azonos tesztelési elvek mellett kell értelmezni. A különböző adatmodellekből fakadó eltéréseket, különösen a Wazuh és CIC flow jellemzők esetében, az eredmények elemzésében külön jelezni kell.

## 7. A diplomamunkához készülő fájlok és ábrák

A szakdolgozat eredményfejezeteihez az alábbi fájlok és ábrák használhatók fel:

- `metrics_summary.csv`: precision, recall, F1, ROC-AUC, mintaszámok és küszöbinformációk.
- `confusion_matrix.csv`: numerikus confusion matrix.
- `confusion_matrix.png`: ábrázolt confusion matrix.
- `predictions.csv`: mintaszintű pontszámok és predikciók.
- `threshold_curve.csv`: küszöbértékekhez tartozó precision, recall és F1 értékek.
- `threshold_curve.png`: küszöbérzékenységi görbe.
- `roc_curve.png`: ROC-görbe, ahol a címkék és pontszámok alapján értelmezhető.
- `score_distribution.png`: benign és támadó minták anomáliapontszám-eloszlása.
- `loss_curve.png`: tanítási veszteséggörbe, ahol a modell története rendelkezésre áll.
- `top_feature_errors.csv`: autoencoder esetén a legnagyobb rekonstrukciós hibát adó jellemzőcsoportok.
- `run_metadata.json`: futtatási metaadatok, például konfiguráció, adathalmaz, sorszámok és kimeneti könyvtárak.
- `train_config.json`: a modell tanításához használt konfiguráció mentett példánya.
- `thresholds.json`: az autoencoder küszöbértékei és a kapcsolódó kalibrációs eredmények.

A szakdolgozatban ezekből elsősorban összehasonlító táblázatok, konfigurációnkénti confusion matrix ábrák, pontszámeloszlások és küszöbérzékenységi ábrák kerülnek felhasználásra.

## 8. Korlátok

### Dataset representativeness

A CIC-IDS2017 széles körben használt kutatási adathalmaz, de nem reprezentál minden modern vállalati, felhős vagy ipari környezetet. Az eredmények ezért kontrollált laboratóriumi következtetéseknek tekinthetők, nem pedig általános érvényű éles üzemi teljesítménygaranciának.

### Wazuh log és CIC flow jellemzők eltérése

A CIC-IDS2017 elsősorban hálózati flow-jellemzőket tartalmaz. A Wazuh ezzel szemben log- és eseményorientált IDS/SIEM platform. A két adatmodell közötti különbség miatt a Wazuh baseline és az AE eredményei csak óvatosan, a bemeneti reprezentáció eltérésének figyelembevételével hasonlíthatók össze.

### Lab-scale validation

A validáció laboratóriumi környezetben történik, nem hosszú ideig futó éles üzemi rendszerben. Emiatt az üzemeltetési megbízhatóság, skálázhatóság, válaszidő, jogosultságkezelés és incidenskezelési munkafolyamat csak korlátozottan értékelhető.

### Concept drift

A modell statikus tanítóhalmazon tanul. Valós hálózatokban a normál forgalom idővel változhat, ami concept driftet okozhat. Ennek kezelése újratanítási stratégiát, driftmonitorozást és időalapú validációt igényelne, amely a laboratóriumi prototípusban nem része a megvalósításnak.

### Egyszerűsített CTI kezelés

A CTI a prototípusban nem teljes értékű threat intelligence platformként jelenik meg. Nem történik több forrásból származó indikátorok automatikus normalizálása, deduplikációja, megbízhatósági súlyozása vagy életciklus-kezelése. A CTI kezelése ezért egyszerűsített, koncepcionális szintű komponens marad.
