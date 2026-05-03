# Végleges kísérleti lehatárolás

## 1. Rövid célmegfogalmazás

A szakdolgozat címe: **„AI-alapú kiberfenyegetés-felderítő és -elemző rendszer”**.

A munka célja egy olyan laboratóriumi prototípus megtervezése, megvalósítása és értékelése, amely hagyományos IDS/SIEM komponenseket egészít ki gépi tanuláson alapuló anomáliadetektálással. A prototípus Wazuh komponensekre, egy saját FastAPI alapú ML szolgáltatásvázra, valamint a CIC-IDS2017 adathalmazon végzett kísérletekre épül. A gépi tanulási komponens egy sklearn `MLPRegressor` alapú autoencoder, amely rekonstrukciós hibából számít anomáliapontszámot.

A cél nem egy éles üzemi IDS teljes körű kiváltása, hanem annak vizsgálata, hogy egy hibrid, szabályalapú és gépi tanulási megközelítés milyen módon illeszthető egy SIEM/IDS architektúrába, és milyen mérőszámokkal értékelhető kontrollált, reprodukálható kísérleti környezetben.

## 2. A végleges MVP pontos lehatárolása

A végleges MVP egy reprodukálható kísérleti prototípus, amely az alábbi fő elemekből áll:

- Wazuh Manager, Wazuh Indexer és Wazuh Dashboard Docker Compose alapú laboratóriumi környezetben.
- Egy saját FastAPI alapú ML service skeleton, amely az architekturális integrációs pontot reprezentálja.
- CIC-IDS2017 adatfeldolgozó pipeline, amely nyers CSV fájlokból tanító, validációs, kalibrációs és teszt adathalmazokat állít elő.
- Autoencoder alapú anomáliadetektáló modell sklearn `MLPRegressor` implementációval.
- Rekonstrukciós hiba alapú anomáliapontszám.
- Három küszöbölési stratégia: fix küszöb, validációs percentilis alapú küszöb és kalibrációs halmazon optimalizált F1-küszöb.
- Statisztikai baseline, amely a tanítóhalmaz középpontjától mért távolság alapján képez anomáliapontszámot.
- Wazuh-stílusú baseline, amely exportált Wazuh riasztásokat vagy Wazuh-szerű predikciós mezőket hasonlít össze a címkézett adatokkal.
- Eredményfájlok, mérőszámok és ábrák előállítása a szakdolgozati értékeléshez.

Az MVP a detektálási és értékelési láncot demonstrálja. A hangsúly a reprodukálható kísérleti pipeline-on, a konfigurációk összehasonlíthatóságán és a korlátok világos megnevezésén van.

## 3. Mit valósítunk meg május 15-ig

Május 15-ig az alábbi elemek megvalósítása és dokumentálása a cél:

- A végleges kísérleti konfigurációk rögzítése az `experiments/final/` könyvtárban.
- A CIC-IDS2017 adathalmazból előállított feldolgozott adatszeletek létrehozása.
- Az `ae_minimal` konfiguráció teljes futtatása, beleértve az adatépítést, a modell tanítását, a küszöbök számítását és a tesztkiértékelést.
- Az `ae_context` konfiguráció jelenlegi kóddal futtatható változatának elkészítése, a későbbi kontextusjellemzők egyértelmű jelölésével.
- A `baseline_stat` konfiguráció futtatása és eredményeinek összehasonlítása az autoencoder eredményeivel.
- A `baseline_wazuh` konfiguráció előkészítése Wazuh vagy Wazuh-szerű exportált predikciók kiértékelésére.
- A `hybrid` konfiguráció tervezési szintű rögzítése, amely az AE és Wazuh predikciók kombinálásának módját írja le.
- A fő mérőszámok táblázatos exportja: precision, recall, F1, false positive rate, confusion matrix és alert count.
- A szakdolgozathoz felhasználható ábrák és táblázatok előállítása: küszöbgörbe, ROC-görbe, score eloszlás, confusion matrix és modellveszteség-görbe, ahol értelmezhető.
- A kísérleti lépések rövid futtatási dokumentációja és az eredmények értelmezése.

## 4. Mit nem valósítunk meg, és miért nem

A végleges MVP nem vállal teljes éles üzemi IDS implementációt. Ennek oka, hogy a szakdolgozat időkerete és a laboratóriumi validációs környezet nem teszi lehetővé egy termelési környezetben hosszú ideig futó, teljes körűen üzemeltetett detektáló rendszer megbízható értékelését.

Nem valósítunk meg teljes körű online tanulást vagy automatikus modellfrissítést. A concept drift kezeléséhez hosszabb idejű, időben változó valós forgalmi adatokra és külön validációs metodikára lenne szükség.

Nem valósítunk meg teljes körű CTI integrációt, például STIX/TAXII feedek automatikus feldolgozását, indikátorok életciklus-kezelését vagy több forrásból származó threat intelligence korrelációját. A CTI kezelés a prototípusban egyszerűsített, koncepcionális elemként jelenik meg.

Nem cél mély Wazuh szabálykészlet-fejlesztés vagy egyedi Wazuh rule engineering kampány végrehajtása. A Wazuh komponens elsősorban IDS/SIEM architekturális baseline és integrációs környezet.

Nem cél a CIC-IDS2017 flow feature-jeinek teljes megfeleltetése valós Wazuh logmezőknek. A két reprezentáció eltérő adatmodellt használ: a CIC-IDS2017 hálózati flow-jellemzőket tartalmaz, míg a Wazuh esemény- és logorientált adatokat kezel.

Nem valósítunk meg nagy skálájú teljesítménytesztet, magas rendelkezésre állású üzemeltetést, jogosultságkezelési auditot vagy production hardeninget. Ezek fontos mérnöki feladatok, de túlmutatnak a szakdolgozat kísérleti fókuszán.

## 5. A végleges összehasonlított konfigurációk

### baseline_stat

A `baseline_stat` egy egyszerű statisztikai anomáliadetektáló baseline. A feldolgozott feature-térben kiszámítja a tanítóhalmaz középpontját, majd a tesztminták ehhez viszonyított távolságából képez anomáliapontszámot. A küszöböt a validációs pontszámok percentilise alapján állítja be.

Ez a konfiguráció nem tekinthető fejlett IDS-nek, de hasznos referenciaérték: megmutatja, hogy egy egyszerű, nem neurális módszer milyen teljesítményt ér el ugyanazon adatelőkészítési pipeline mellett.

### baseline_wazuh

A `baseline_wazuh` a Wazuh vagy Wazuh-szerű riasztási eredmények kiértékelésére szolgál. A bemenet egy exportált fájl, amely tartalmazza a valós címkét, valamint a Wazuh riasztási vagy predikciós mezőit.

A cél nem annak állítása, hogy a Wazuh natívan ugyanazokat a CIC-IDS2017 flow feature-öket használja, mint az autoencoder, hanem egy szabályalapú vagy riasztásalapú baseline beemelése az összehasonlításba. Az eredmények értelmezésénél figyelembe kell venni a logalapú és flow-alapú adatmodell közötti eltérést.

### ae_minimal

Az `ae_minimal` a fő autoencoder konfiguráció. A bemeneti feature-készlet a jelenleg implementált minimális jellemzőkből áll:

- `destination_port`
- `flow_duration`
- `total_fwd_packets`
- `total_backward_packets`
- `flow_bytes_per_sec`
- `flow_packets_per_sec`
- `protocol`

A numerikus jellemzőket standard skálázás, a kategorikus protokollmezőt one-hot encoding alakítja át. Az autoencoder csak benign mintákon tanul, a tesztelés pedig benign és támadó mintákat egyaránt tartalmaz. A detektálás alapja a rekonstrukciós hiba.

### ae_context

Az `ae_context` a kontextusjellemzőkkel bővített irányt képviseli. A jelenlegi MVP-ben a konfiguráció futtatható marad a már implementált minimális feature-készlettel. A tervezett kontextusjellemzők külön TODO szekcióban szerepelnek, például időablakos forrásszámosságok, célport-gyakoriságok és forgalmi aggregátumok.

Ez a megközelítés lehetőséget ad annak bemutatására, hogyan bővíthető a rendszer viselkedési kontextussal, miközben a végleges futtatás nem támaszkodik még nem implementált feature engineering lépésekre.

### hybrid

A `hybrid` konfiguráció az autoencoder és a Wazuh riasztások kombinálásának tervezett irányát írja le. A legegyszerűbb döntési szabály szerint akkor keletkezik riasztás, ha az AE vagy a Wazuh komponens támadást jelez. Egy későbbi implementációban a pontszámok normalizált kombinációja vagy súlyozott döntési logika is alkalmazható.

A szakdolgozatban a hibrid konfiguráció elsősorban architekturális és módszertani elemként jelenik meg. Teljes értékű összehasonlítása csak akkor végezhető el, ha az AE predikciók és a Wazuh riasztások stabil esemény- vagy flow-azonosító alapján összekapcsolhatók.

## 6. Mérőszámok

Az összehasonlítás fő mérőszámai:

- **Precision**: a pozitívnak jelzett riasztások közül mennyi volt valóban támadás.
- **Recall**: a tényleges támadások mekkora részét találta meg a rendszer.
- **F1**: a precision és recall harmonikus átlaga, amely kiegyensúlyozott képet ad a két szempont között.
- **False positive rate**: a benign minták közül mekkora arányt jelzett tévesen támadásnak a rendszer.
- **Confusion matrix**: a true negative, false positive, false negative és true positive értékek táblázatos összefoglalása.
- **Alert count**: az előállított riasztások száma, amely az üzemeltetési terhelés becsléséhez fontos.

A mérőszámokat minden konfigurációnál azonos címkézési logika és azonos tesztelési elvek mellett kell értelmezni. A különböző adatmodellekből fakadó eltéréseket, különösen a Wazuh és CIC flow feature-k esetében, az eredmények elemzésében külön jelezni kell.

## 7. A thesishez készülő fájlok és ábrák

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
- `top_feature_errors.csv`: autoencoder esetén a legnagyobb rekonstrukciós hibát adó feature-csoportok.
- `run_metadata.json`: futtatási metaadatok, például konfiguráció, adathalmaz, sorszámok és kimeneti könyvtárak.
- `train_config.json`: a modell tanításához használt konfiguráció mentett példánya.
- `thresholds.json`: az autoencoder küszöbértékei és a kapcsolódó kalibrációs eredmények.

A szakdolgozatban ezekből elsősorban összehasonlító táblázatok, konfigurációnkénti confusion matrix ábrák, pontszámeloszlások és küszöbérzékenységi ábrák kerülnek felhasználásra.

## 8. Korlátok

### Dataset representativeness

A CIC-IDS2017 széles körben használt kutatási adathalmaz, de nem reprezentál minden modern vállalati, felhős vagy ipari környezetet. Az eredmények ezért kontrollált laboratóriumi következtetéseknek tekinthetők, nem pedig általános érvényű éles üzemi teljesítménygaranciának.

### Wazuh log és CIC flow feature mismatch

A CIC-IDS2017 elsősorban hálózati flow-jellemzőket tartalmaz. A Wazuh ezzel szemben log- és eseményorientált IDS/SIEM platform. A két adatmodell közötti különbség miatt a Wazuh baseline és az AE eredményei csak óvatosan, a bemeneti reprezentáció eltérésének figyelembevételével hasonlíthatók össze.

### Lab-scale validation

A validáció laboratóriumi környezetben történik, nem hosszú ideig futó production rendszerben. Emiatt az üzemeltetési megbízhatóság, skálázhatóság, válaszidő, jogosultságkezelés és incidenskezelési workflow csak korlátozottan értékelhető.

### Concept drift

A modell statikus tanítóhalmazon tanul. Valós hálózatokban a normál forgalom idővel változhat, ami concept driftet okozhat. Ennek kezelése újratanítási stratégiát, driftmonitorozást és időalapú validációt igényelne, amely a jelenlegi MVP-ben nem része a megvalósításnak.

### Egyszerűsített CTI kezelés

A CTI a prototípusban nem teljes értékű threat intelligence platformként jelenik meg. Nem történik több forrásból származó indikátorok automatikus normalizálása, deduplikációja, megbízhatósági súlyozása vagy életciklus-kezelése. A CTI kezelése ezért egyszerűsített, koncepcionális szintű komponens marad.
