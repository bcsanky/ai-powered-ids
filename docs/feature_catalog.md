# Feature katalógus

Ez a dokumentum a szakdolgozati prototípus jelenleg implementált feature-készletét foglalja össze. A forrásigazság az `experiments/experiment.yaml` konfiguráció és az `ml/src/schema.py` oszlopkanonizáló modul. A katalógus célja, hogy a megvalósítási fejezetben egyértelműen bemutassa, milyen bemeneti változókat használ az autoencoder alapú anomáliadetektálás, hogyan történik az előfeldolgozás, és milyen szerepet töltenek be ezek a jellemzők az IDS értékelésében.

A jelenlegi implementáció a CIC-IDS2017 adathalmaz flow-szintű mezőire épül. Az `AE-Minimal` konfiguráció a lenti hét jellemzőt használja. Az `AE-Context` konfiguráció jelenleg futtatható kompatibilis változatként ugyanebből a feature-készletből indul ki; a valódi kontextusjellemzők későbbi, opcionális bővítésként szerepelnek, és a jelenlegi kódban még nem tekinthetők implementáltnak.

| Feature name | Type | Source | Description | Used in AE-Minimal | Used in AE-Context | Preprocessing | Thesis relevance |
|---|---|---|---|---|---|---|---|
| `destination_port` | Numerikus | CIC-IDS2017 flow mező; kanonikus aliasok: `destination_port`, `destination port`, `dst_port`, `dst port` | A céloldali port száma, amely a kommunikáció célzott szolgáltatására utal. | Igen | Igen, jelenleg kompatibilis minimális feature-ként; kontextusbővítés tervezett / opcionális | Numerikussá alakítás, végtelen értékek cseréje hiányzó értékre, hiányzó értékek eldobása, majd `StandardScaler` | Segít elkülöníteni a szolgáltatás- vagy portspecifikus támadási mintázatokat, például port scan vagy szolgáltatáscélzott forgalom esetén. |
| `flow_duration` | Numerikus | CIC-IDS2017 flow mező; kanonikus aliasok: `flow_duration`, `flow duration` | A hálózati flow időtartama. | Igen | Igen, jelenleg kompatibilis minimális feature-ként; kontextusbővítés tervezett / opcionális | Numerikussá alakítás, végtelen értékek cseréje hiányzó értékre, hiányzó értékek eldobása, majd `StandardScaler` | A forgalom időbeli viselkedését írja le; rövid vagy szokatlanul hosszú kapcsolatok anomáliára utalhatnak. |
| `total_fwd_packets` | Numerikus | CIC-IDS2017 flow mező; kanonikus aliasok: `total_fwd_packets`, `total fwd packets`, `total_forward_packets` | A forward irányú csomagok száma a flow-ban. | Igen | Igen, jelenleg kompatibilis minimális feature-ként; kontextusbővítés tervezett / opcionális | Numerikussá alakítás, végtelen értékek cseréje hiányzó értékre, hiányzó értékek eldobása, majd `StandardScaler` | A kapcsolat irányonkénti intenzitását jellemzi, és hozzájárulhat a normál és támadó forgalmi profilok elkülönítéséhez. |
| `total_backward_packets` | Numerikus | CIC-IDS2017 flow mező; kanonikus aliasok: `total_backward_packets`, `total backward packets`, `total_bwd_packets` | A backward irányú csomagok száma a flow-ban. | Igen | Igen, jelenleg kompatibilis minimális feature-ként; kontextusbővítés tervezett / opcionális | Numerikussá alakítás, végtelen értékek cseréje hiányzó értékre, hiányzó értékek eldobása, majd `StandardScaler` | A válaszoldali forgalom mennyiségét írja le; az aszimmetrikus vagy rendellenes válaszmintázatok detektálásában lehet szerepe. |
| `flow_bytes_per_sec` | Numerikus | CIC-IDS2017 flow mező; kanonikus aliasok: `flow_bytes_per_sec`, `flow bytes/s`, `flow bytes per sec` | A flow adatsebessége byte/másodperc egységben. | Igen | Igen, jelenleg kompatibilis minimális feature-ként; kontextusbővítés tervezett / opcionális | Numerikussá alakítás, végtelen értékek cseréje hiányzó értékre, hiányzó értékek eldobása, majd `StandardScaler` | A forgalom volumenét és sebességét reprezentálja; túlzott vagy szokatlan adatsebesség támadási viselkedésre utalhat. |
| `flow_packets_per_sec` | Numerikus | CIC-IDS2017 flow mező; kanonikus aliasok: `flow_packets_per_sec`, `flow packets/s`, `flow packets per sec` | A flow csomagsebessége csomag/másodperc egységben. | Igen | Igen, jelenleg kompatibilis minimális feature-ként; kontextusbővítés tervezett / opcionális | Numerikussá alakítás, végtelen értékek cseréje hiányzó értékre, hiányzó értékek eldobása, majd `StandardScaler` | Kiemelten releváns volumetrikus vagy gyors ismétlődésű támadások esetén, mivel a csomagintenzitás eltérhet a normál forgalomtól. |
| `protocol` | Kategorikus | CIC-IDS2017 flow mező; kanonikus alias: `protocol` | A hálózati protokoll azonosítója vagy megnevezése. | Igen | Igen, jelenleg kompatibilis minimális feature-ként; kontextusbővítés tervezett / opcionális | Hiányzó érték esetén `unknown`, szöveges típusra alakítás, majd `OneHotEncoder(handle_unknown="ignore")` | A protokoll szerinti elkülönítés segít a forgalmi mintázatok értelmezésében, és csökkenti annak kockázatát, hogy eltérő protokollok azonos numerikus térben tévesen összeolvadjanak. |

## Implementációs megjegyzések

Az oszlopnevek feldolgozása az `ml/src/schema.py` modulban definiált aliaslista alapján történik. A pipeline először normalizálja az oszlopneveket, majd a CIC-IDS2017 eredeti mezőit a kanonikus feature-nevekre nevezi át. Ez lehetővé teszi, hogy az eltérő írásmódú CSV oszlopnevek, például szóközös vagy aláhúzásos változatok, ugyanarra a belső feature-névre képeződjenek le.

A numerikus feature-ök esetén a pipeline explicit numerikus konverziót végez. A nem értelmezhető értékek hiányzó értékké alakulnak, az infinities értékek szintén hiányzó értékként kezelődnek, majd a hiányos sorok a numerikus feature-ök alapján kiesnek. A skálázás `StandardScaler` segítségével történik, és az előfeldolgozó kizárólag a tanítóhalmazon illeszkedik.

A `protocol` kategorikus feature one-hot encodinggal kerül a modell bemenetére. Az ismeretlen kategóriák kezelése `ignore`, ami biztosítja, hogy a validációs, kalibrációs vagy teszt adatokban megjelenő új protokollértékek ne okozzanak futási hibát.

Az autoencoder tanítása benign mintákon történik, ezért a feature-ök szerepe nem közvetlen osztályozási szabályok formájában jelenik meg, hanem a normál forgalom rekonstrukciós mintázatának megtanulásában. A tesztelés során a magas rekonstrukciós hiba anomáliapontszámként értelmezhető.

## AE-Context státusz

Az `AE-Context` irány célja, hogy a minimális flow feature-öket időbeli, forrás- és céloldali, illetve biztonsági kontextussal egészítse ki. A jelenlegi repository alapján ezek a kontextusfeature-ök még nem részei az implementált `ml/src/build_dataset.py` feldolgozási logikának. Ezért a jelenlegi `AE-Context` konfiguráció csak kompatibilis, futtatható változatként használja a minimális feature-készletet, a kontextusjellemzők pedig tervezett / opcionális bővítésként kezelendők.

Ez a megkülönböztetés fontos a szakdolgozatban: a jelenlegi kísérleti eredmények nem állíthatják, hogy a modell már időablakos vagy CTI-alapú kontextusjellemzőket használ. Ezek a bővítések a rendszer továbbfejlesztési irányát jelölik.

## Későbbi bővítési lehetőségek

A későbbi fejlesztések során az alábbi feature-típusokkal bővíthető a jelenlegi katalógus:

- **Time-window connection count**: adott forrásból vagy forrás-cél párból indított kapcsolatok száma meghatározott időablakban.
- **Unique destination count**: egy forrás által elért egyedi célcímek vagy célportok száma adott időablakban.
- **Failed login count**: sikertelen bejelentkezési események száma host-, user- vagy forráscím-alapon aggregálva.
- **Rare port indicator**: bináris vagy rangsorolt jelzés arra, hogy a célport ritka-e a megfigyelt környezetben.
- **IOC hit flag**: bináris jelzés arra, hogy az esemény vagy flow kapcsolódik-e ismert indikátorhoz, például gyanús IP-címhez, domainhez vagy hash értékhez.

Ezek a bővítések növelhetik a modell kontextusérzékenységét, de külön feature engineering lépéseket, adatforrás-integrációt és új validációs kísérleteket igényelnek. A jelenlegi MVP ezért ezeket nem tekinti implementált képességnek.
