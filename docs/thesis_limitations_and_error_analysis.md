# Hibaanalízis és korlátok

## Detektálási hibák

A detektálási hibák értelmezésében a false positive és false negative eseteket külön kell kezelni. A false positive riasztások benign mintákat jelölnek támadásként, ami éles üzemi környezetben magas elemzői terheléshez vezethet. A false negative esetek ezzel szemben tényleges támadó minták elmulasztását jelentik, ami biztonsági kockázatot hordoz.

Az AE-Minimal és AE-Context eredményeknél a kiválasztott `f1_optimum` küszöb nagyon magas recall értéket eredményezett, ugyanakkor a hamis pozitív arány is rendkívül magas. Ez azt mutatja, hogy az F1-optimalizálás önmagában nem elegendő üzemeltetési szempontból, ha a riasztásszám túl nagy.

A statisztikai baseline és a szabályalapú proxy baseline alacsonyabb riasztásszámot adott, de sok támadó mintát nem jelölt pozitívként. Ez a false negative oldal korlátját mutatja. A hibaanalízisben ezért a küszöbérzékenységet, a pontszámeloszlások átfedését és a riasztásszámot együtt kell értelmezni.

## Jellemzőképzési korlátok

Az AE-Minimal jellemzőkészlete szándékosan szűk, ezért csak alapvető flow-szintű információt tartalmaz. A célport, időtartam, csomagszám, byte- és csomagsebesség, valamint protokoll mezők hasznosak, de nem fedik le a teljes hálózati kontextust.

Az AE-Context bővítés egyszerű gyakorisági és arányalapú jellemzőket használ. Ezek stabilan számíthatók timestamp nélküli CIC-IDS2017 flow adatokon, de nem helyettesítik az időablakos, hostalapú vagy CTI-alapú kontextust. Nem készül például forráscímenkénti kapcsolatszám, egyedi célpontszám vagy indikátortalálati jelző.

A gyakorisági kontextusjellemzők kizárólag a tanító adatrészből illeszkednek, ami csökkenti az adatelszivárgás kockázatát. Ennek mellékhatása, hogy validációs, kalibrációs vagy teszt adatrészben megjelenő új portok és protokollok `0.0` gyakoriságot kapnak. Ez helyes mérési szempontból, de valós környezetben külön kezelést igényelhet.

## Dataset korlátok

A CIC-IDS2017 kutatási célra széles körben használt adathalmaz, de nem reprezentál minden modern vállalati, felhős vagy ipari hálózatot. Az adathalmaz osztályeloszlása, támadástípusai és mérési körülményei eltérhetnek egy valós szervezet forgalmától.

A modell és a baseline-ok eredményei ezért kontrollált laboratóriumi érvényességűek. Nem állítható, hogy azonos metrikák érhetők el más hálózatban, más forgalmi eloszlás mellett vagy időben változó normál viselkedés esetén.

Az osztályeloszlás a küszöbválasztásra is hatással van. Ha a kalibrációs vagy teszt adatrész támadó mintákban gazdag, az F1-optimalizálás a recall irányába tolhatja a döntési határt. Ez növelheti a benign minták téves riasztási arányát.

## Wazuh és proxy baseline korlátok

A Wazuh log- és eseményorientált IDS/SIEM platform, míg a CIC-IDS2017 flow-alapú hálózati jellemzőket tartalmaz. A két adatmodell közvetlen megfeleltetése korlátozott. Emiatt natív Wazuh teljesítménymérés csak olyan exporttal lenne védhető, amely tartalmazza a valós címkéket és a Wazuh riasztási vagy predikciós mezőit.

Jelen mérési állapotban nincs validált natív Wazuh export. A szabályalapú proxy baseline kontrollált, flow-alapú referencia, amely Wazuh-szerű riasztási mezőket állít elő, de nem azonos a Wazuh natív szabálymotorjának teljesítményével.

A proxy baseline célja az összehasonlító mérési lánc kiegészítése egy egyszerű szabályalapú referenciával. Nem szabad úgy értelmezni, mint Wazuh szabálykészlet-validációt, natív Wazuh dashboard-riasztási teljesítményt vagy teljes SIEM mérési eredményt.

## Hibrid értékelés korlátai

A hibrid kiértékelés az AE-Minimal és a szabályalapú proxy baseline predikcióinak unióját használja. A döntés offline, azonos sorrendű teszthalmaz-predikciókra épül. Ez kontrollált összehasonlításra alkalmas, de nem jelent éles eseménykorrelációt.

Éles környezetben a hibrid döntéshez stabil eseményazonosító, időbélyeg, forrás- és célobjektum, valamint log- és flow-események közötti korrelációs szabályok kellenének. Ezek a prototípus jelen változatában nem részei a mérésnek.

A hibrid eredmény értelmezésénél azt is figyelembe kell venni, hogy az unió alapú döntés növelheti a recall értéket, de gyakran növeli a riasztásszámot és a hamis pozitív arányt is.

## Lab/replay korlátok

A lab/replay demonstráció kis elemszámú kontrollált eseménysorra épül. A szcenáriók a prototípus működésének bemutatását szolgálják: normál aktivitás, port scan jellegű események, SSH brute force jellegű események és kombináltan gyanús minták.

Ezek az események nem natív Wazuh exportból származó bizonyítékok, és nem tekinthetők éles SOC-validációnak. A kis elemszám miatt statisztikai következtetésre nem alkalmasak. A szerepük az, hogy bemutassák a scoring kimeneteket, a kockázati kategóriákat, az indoklást és a riportkészítési folyamatot.

## Teljesítménymérés korlátok

A teljesítménymérés lokális/labor batch scoring mérés. A benchmark a meglévő AE-Minimal modellt és előfeldolgozót használja, és nem indít új tanítást. Az események ismétlése determinisztikus, és kizárólag a feldolgozási kapacitás mérésére szolgál.

Az eredmények hardver- és környezetfüggők. Nem tartalmaznak natív Wazuh indexelést, dashboard terhelést, hosszú idejű stressztesztet, hálózati késleltetést vagy többfelhasználós SOC munkafolyamatot. Ezért nem szabad éles üzemi teljesítménygaranciaként értelmezni őket.

A CPU-használat processz CPU-időként értelmezendő, nem teljes gépszintű terhelésként. A memóriahasználati értékek platformfüggők: ahol psutil elérhető, RSS alapú értékek rögzíthetők, Unix/Linux környezetben pedig csúcsmemória is mérhető. Ha valamelyik memóriaérték nem áll rendelkezésre, az nem érvényteleníti a latency és throughput mérést, de a memóriaelemzést korlátozza.

## Etikai és adatvédelmi megfontolások

A mérés nyilvános kutatási adathalmazra és saját demonstrációs eseményekre épül. A prototípus nem tartalmaz éles felhasználói logokat, személyes azonosítókat vagy valós incidensadatokat.

Éles környezetben a SIEM- és IDS-logok érzékeny adatokat tartalmazhatnak, például IP-címeket, felhasználóneveket, hostneveket, hitelesítési eseményeket és incidensleírásokat. Ilyen környezetben hozzáférés-kezelés, naplózási kontroll, adatminimalizálás, megőrzési szabályok és adatvédelmi hatásvizsgálat szükséges.

A gépi tanulási komponens döntéseit nem célszerű automatikus szankcionáló döntésként használni. A prototípus kimenetei szakértői elemzést támogató jelzések, amelyek emberi felülvizsgálattal értelmezendők.
