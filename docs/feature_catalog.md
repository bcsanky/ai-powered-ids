# Feature katalógus

Ez a dokumentum a szakdolgozati prototípus jelenleg implementált feature-készletét foglalja össze. A nyers feature-ök forrásigazsága az `experiments/experiment.yaml` konfiguráció és az `ml/src/schema.py` oszlopkanonizáló modul; a generált context feature-ök implementációja az `ml/src/build_dataset.py` adatépítő pipeline-ban található. A katalógus célja, hogy a megvalósítási fejezetben egyértelműen bemutassa, milyen bemeneti változókat használ az autoencoder alapú anomáliadetektálás, hogyan történik az előfeldolgozás, és milyen szerepet töltenek be ezek a jellemzők az IDS értékelésében.

A jelenlegi implementáció a CIC-IDS2017 adathalmaz flow-szintű mezőire épül. Az `AE-Minimal` konfiguráció a lenti hét alapjellemzőt használja. Az `AE-Context` konfiguráció ugyanezekből az alapjellemzőkből indul ki, és `features.context_enabled: true` esetén további, timestampet nem igénylő, determinisztikusan számított context feature-öket is előállít.

| Feature name | Type | Source | Description | Used in AE-Minimal | Used in AE-Context | Preprocessing | Thesis relevance |
|---|---|---|---|---|---|---|---|
| `destination_port` | Numerikus | CIC-IDS2017 flow mező; kanonikus aliasok: `destination_port`, `destination port`, `dst_port`, `dst port` | A céloldali port száma, amely a kommunikáció célzott szolgáltatására utal. | Igen | Igen, alap feature-ként | Numerikussá alakítás, végtelen értékek cseréje hiányzó értékre, hiányzó értékek eldobása, majd `StandardScaler` | Segít elkülöníteni a szolgáltatás- vagy portspecifikus támadási mintázatokat, például port scan vagy szolgáltatáscélzott forgalom esetén. |
| `flow_duration` | Numerikus | CIC-IDS2017 flow mező; kanonikus aliasok: `flow_duration`, `flow duration` | A hálózati flow időtartama. | Igen | Igen, alap feature-ként | Numerikussá alakítás, végtelen értékek cseréje hiányzó értékre, hiányzó értékek eldobása, majd `StandardScaler` | A forgalom időbeli viselkedését írja le; rövid vagy szokatlanul hosszú kapcsolatok anomáliára utalhatnak. |
| `total_fwd_packets` | Numerikus | CIC-IDS2017 flow mező; kanonikus aliasok: `total_fwd_packets`, `total fwd packets`, `total_forward_packets` | A forward irányú csomagok száma a flow-ban. | Igen | Igen, alap feature-ként | Numerikussá alakítás, végtelen értékek cseréje hiányzó értékre, hiányzó értékek eldobása, majd `StandardScaler` | A kapcsolat irányonkénti intenzitását jellemzi, és hozzájárulhat a normál és támadó forgalmi profilok elkülönítéséhez. |
| `total_backward_packets` | Numerikus | CIC-IDS2017 flow mező; kanonikus aliasok: `total_backward_packets`, `total backward packets`, `total_bwd_packets` | A backward irányú csomagok száma a flow-ban. | Igen | Igen, alap feature-ként | Numerikussá alakítás, végtelen értékek cseréje hiányzó értékre, hiányzó értékek eldobása, majd `StandardScaler` | A válaszoldali forgalom mennyiségét írja le; az aszimmetrikus vagy rendellenes válaszmintázatok detektálásában lehet szerepe. |
| `flow_bytes_per_sec` | Numerikus | CIC-IDS2017 flow mező; kanonikus aliasok: `flow_bytes_per_sec`, `flow bytes/s`, `flow bytes per sec` | A flow adatsebessége byte/másodperc egységben. | Igen | Igen, alap feature-ként | Numerikussá alakítás, végtelen értékek cseréje hiányzó értékre, hiányzó értékek eldobása, majd `StandardScaler` | A forgalom volumenét és sebességét reprezentálja; túlzott vagy szokatlan adatsebesség támadási viselkedésre utalhat. |
| `flow_packets_per_sec` | Numerikus | CIC-IDS2017 flow mező; kanonikus aliasok: `flow_packets_per_sec`, `flow packets/s`, `flow packets per sec` | A flow csomagsebessége csomag/másodperc egységben. | Igen | Igen, alap feature-ként | Numerikussá alakítás, végtelen értékek cseréje hiányzó értékre, hiányzó értékek eldobása, majd `StandardScaler` | Kiemelten releváns volumetrikus vagy gyors ismétlődésű támadások esetén, mivel a csomagintenzitás eltérhet a normál forgalomtól. |
| `protocol` | Kategorikus | CIC-IDS2017 flow mező; kanonikus alias: `protocol` | A hálózati protokoll azonosítója vagy megnevezése. | Igen | Igen, alap feature-ként | Hiányzó érték esetén `unknown`, szöveges típusra alakítás, majd `OneHotEncoder(handle_unknown="ignore")` | A protokoll szerinti elkülönítés segít a forgalmi mintázatok értelmezésében, és csökkenti annak kockázatát, hogy eltérő protokollok azonos numerikus térben tévesen összeolvadjanak. |
| `destination_port_frequency` | Numerikus, generált context | `destination_port` alapján számított relatív gyakoriság a feldolgozott adathalmazban | Megmutatja, hogy az adott célport milyen arányban fordul elő a teljes tisztított flow-készletben. | Nem | Igen, `context_enabled: true` esetén | Generálás után numerikus feature-ként `StandardScaler` | Segít elkülöníteni a gyakori szolgáltatásokat a ritkább célportoktól, ami portspecifikus anomáliák értelmezésénél hasznos. |
| `protocol_frequency` | Numerikus, generált context | `protocol` alapján számított relatív gyakoriság a feldolgozott adathalmazban | Megmutatja, hogy az adott protokoll milyen arányban szerepel az adathalmazban. | Nem | Igen, `context_enabled: true` esetén | Generálás után numerikus feature-ként `StandardScaler` | A protokoll ritkaságát vagy gyakoriságát adja hozzá a modell bemenetéhez, közvetlen timestamp-függés nélkül. |
| `is_rare_destination_port` | Numerikus, generált context | `destination_port_frequency` és konfigurált `rare_destination_port_threshold` alapján | Bináris jelző, értéke 1, ha a célport relatív gyakorisága a küszöb alatt van. | Nem | Igen, `context_enabled: true` esetén | Generálás után numerikus feature-ként `StandardScaler` | Egyszerű ritka port indikátor, amely segíthet a szokatlan szolgáltatáscélzás felismerésében. |
| `packet_ratio` | Numerikus, generált context | `total_fwd_packets / max(total_backward_packets, 1)` | A forward és backward csomagszámok arányát írja le. | Nem | Igen, `context_enabled: true` esetén | Generálás után numerikus feature-ként `StandardScaler` | Az irányaszimmetria jellemzésével kiegészíti a nyers csomagszámokat. |
| `bytes_packets_ratio` | Numerikus, generált context | `flow_bytes_per_sec / max(flow_packets_per_sec, epsilon)` | Az adatsebesség és csomagsebesség arányából számított, hozzávetőleges byte/csomag jellegű mutató. | Nem | Igen, `context_enabled: true` esetén | Generálás után numerikus feature-ként `StandardScaler` | A forgalom intenzitását nemcsak sebességként, hanem byte/csomag jellegű arányként is megjeleníti. |

## Implementációs megjegyzések

Az oszlopnevek feldolgozása az `ml/src/schema.py` modulban definiált aliaslista alapján történik. A pipeline először normalizálja az oszlopneveket, majd a CIC-IDS2017 eredeti mezőit a kanonikus feature-nevekre nevezi át. Ez lehetővé teszi, hogy az eltérő írásmódú CSV oszlopnevek, például szóközös vagy aláhúzásos változatok, ugyanarra a belső feature-névre képeződjenek le.

A numerikus feature-ök esetén a pipeline explicit numerikus konverziót végez. A nem értelmezhető értékek hiányzó értékké alakulnak, az infinities értékek szintén hiányzó értékként kezelődnek, majd a hiányos sorok a numerikus feature-ök alapján kiesnek. A skálázás `StandardScaler` segítségével történik, és az előfeldolgozó kizárólag a tanítóhalmazon illeszkedik.

A `protocol` kategorikus feature one-hot encodinggal kerül a modell bemenetére. Az ismeretlen kategóriák kezelése `ignore`, ami biztosítja, hogy a validációs, kalibrációs vagy teszt adatokban megjelenő új protokollértékek ne okozzanak futási hibát.

Az autoencoder tanítása benign mintákon történik, ezért a feature-ök szerepe nem közvetlen osztályozási szabályok formájában jelenik meg, hanem a normál forgalom rekonstrukciós mintázatának megtanulásában. A tesztelés során a magas rekonstrukciós hiba anomáliapontszámként értelmezhető.

## AE-Context státusz

Az `AE-Context` irány célja, hogy a minimális flow feature-öket egyszerű, stabilan számítható kontextussal egészítse ki. A jelenlegi implementáció timestamp nélküli CIC-IDS2017 flow adatokon működik, ezért nem használ időablakos aggregációkat. A context feature engineering a `features.context_enabled: true` kapcsolóval aktiválható, és az elkészült generált feature-ök automatikusan bekerülnek a numerikus feature-listába.

Ez a megkülönböztetés fontos a szakdolgozatban: az `AE-Context` már használ egyszerű gyakorisági és arányalapú kontextusfeature-öket, de továbbra sem állítható róla, hogy időablakos, hostalapú vagy CTI-alapú kontextusmodellezést valósítana meg.

## Későbbi bővítési lehetőségek

A későbbi fejlesztések során az alábbi feature-típusokkal bővíthető a jelenlegi katalógus:

- **Time-window connection count**: adott forrásból vagy forrás-cél párból indított kapcsolatok száma meghatározott időablakban.
- **Unique destination count**: egy forrás által elért egyedi célcímek vagy célportok száma adott időablakban.
- **Failed login count**: sikertelen bejelentkezési események száma host-, user- vagy forráscím-alapon aggregálva.
- **Környezetfüggő rare service indikátorok**: a jelenlegi ritka célport jelző továbbfejlesztése host-, subnet- vagy időablak-specifikus gyakoriságokkal.
- **IOC hit flag**: bináris jelzés arra, hogy az esemény vagy flow kapcsolódik-e ismert indikátorhoz, például gyanús IP-címhez, domainhez vagy hash értékhez.

Ezek a bővítések növelhetik a modell kontextusérzékenységét, de külön feature engineering lépéseket, adatforrás-integrációt és új validációs kísérleteket igényelnek. A jelenlegi MVP csak a táblázatban szereplő timestamp nélküli context feature-öket tekinti implementált képességnek.
