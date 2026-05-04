# 6. Eredmények és értékelés

## 6.1 Értékelési cél és módszertan

Az értékelés célja a laboratóriumi prototípus detektálási és mérési láncának összehasonlítható vizsgálata. A fő kérdés az, hogy az autoencoder-alapú anomáliadetektálás, az egyszerű statisztikai baseline, a szabályalapú proxy baseline és az offline hibrid döntés milyen mérőszámokkal jellemezhető azonos tesztelési környezetben.

A validált konfigurációk: AE-Minimal, AE-Context, baseline_stat, rule_proxy és hybrid. A natív Wazuh ág nem validált, mert nem áll rendelkezésre címkézett natív Wazuh export. Ezért a natív Wazuh teljesítménymérés helyett kontrollált szabályalapú proxy baseline készült.

A fő metrikák a precision, recall, F1, false positive rate, false negative rate, alert count, ROC-AUC és a konfúziós mátrix. A precision a pozitív jelzések pontosságát, a recall a tényleges támadó minták megtalálási arányát, az F1 pedig ezek kiegyensúlyozott összegzését adja. A false positive rate és az alert count üzemeltetési szempontból fontos, mert a téves riasztások és a riasztási mennyiség közvetlenül hat az elemzői terhelésre.

## 6.2 Adathalmaz és felosztás

A mérés a CIC-IDS2017 flow-alapú adathalmazon készült. Az adatfeldolgozás tanító, validációs, kalibrációs és teszt adatrészeket állított elő. Az előfeldolgozó kizárólag a tanító adatrészen illeszkedett.

| Adatrész | Mintaszám | Benign | Támadás |
|---|---:|---:|---:|
| Tanító adatrész | 1 589 922 | [KITÖLTENDŐ: train benign bontás, ha a végleges táblázat igényli] | [KITÖLTENDŐ: train attack bontás, ha a végleges táblázat igényli] |
| Validációs adatrész | 340 698 | [KITÖLTENDŐ: validation benign bontás, ha a végleges táblázat igényli] | [KITÖLTENDŐ: validation attack bontás, ha a végleges táblázat igényli] |
| Kalibrációs adatrész | 448 628 | 170 350 | 278 278 |
| Teszt adatrész | 448 628 | 170 350 | 278 278 |

A fő összehasonlítás a teszt adatrészre épült, amely 448 628 mintát tartalmazott.

## 6.3 AE-Minimal eredményei

Az AE-Minimal a hét alap flow jellemzőt használta. A fő összehasonlításban az `f1_optimum` küszöbsor szerepelt. A validált futtatási könyvtár: `results/final/final-ae-minimal-v1/ae_v1_20260504_113852/`.

| Metrika | Érték |
|---|---:|
| Precision | 0,620287 |
| Recall | 1,000000 |
| F1 | 0,765651 |
| False positive rate | 1,000000 |
| False negative rate | 0,000000 |
| Alert count | 448 628 |
| ROC-AUC | 0,605370 |

Az AE-Minimal minden támadó mintát pozitívként jelölt, ugyanakkor minden benign tesztmintát is riasztásként sorolt be. Ez maximális recall értéket ad, de nagyon magas hamis pozitív aránnyal jár. A konfiguráció ezért jól szemlélteti, hogy a magas recall önmagában nem elegendő, ha a riasztásszám üzemeltetési szempontból túl nagy.

Felhasználható ábrák: AE-Minimal konfúziós mátrix, score distribution, threshold curve és top feature frequency.

## 6.4 AE-Context eredményei

Az AE-Context a minimális flow jellemzőket tanító adatrészből illesztett port- és protokollgyakoriságokkal, ritka célport jelzővel, valamint forgalmi arányjellemzőkkel bővítette. Ezek nem időablakos, nem hostalapú és nem CTI-alapú kontextusjellemzők.

| Metrika | Érték |
|---|---:|
| Precision | 0,620290 |
| Recall | 1,000000 |
| F1 | 0,765653 |
| False positive rate | 0,999988 |
| False negative rate | 0,000000 |
| Alert count | 448 626 |
| ROC-AUC | 0,764273 |

Az AE-Context precision, recall és F1 értéke nagyon közel áll az AE-Minimal eredményéhez. A ROC-AUC viszont magasabb, ami arra utal, hogy a pontszámok rangsorolási képessége kedvezőbb lehet. A kiválasztott küszöb mellett azonban a riasztási mennyiség továbbra is nagyon magas, ezért a javulás nem fogalmazható meg általános üzemeltetési előnyként. Az eredmények alapján a két jellemzőkészlet hatása összehasonlíthatóvá vált.

## 6.5 Baseline eredmények

A baseline_stat egyszerű távolságalapú statisztikai viszonyítási alap. Eredményei:

| Metrika | Érték |
|---|---:|
| Precision | 0,514707 |
| Recall | 0,032008 |
| F1 | 0,060267 |
| False positive rate | 0,049299 |
| False negative rate | 0,967992 |
| Alert count | 17 305 |
| ROC-AUC | 0,475816 |

A rule_proxy kontrollált szabályalapú proxy baseline. Eredményei:

| Metrika | Érték |
|---|---:|
| Precision | 0,525269 |
| Recall | 0,033391 |
| F1 | 0,062791 |
| False positive rate | 0,049299 |
| False negative rate | 0,966609 |
| Alert count | 17 690 |
| ROC-AUC | 0,573446 |

Mindkét baseline kevesebb riasztást állított elő, mint az autoencoder konfigurációk, de recall értékük alacsony. A natív Wazuh teljesítménymérés helyett kontrollált szabályalapú proxy baseline készült, ezért a rule_proxy nem értelmezhető natív Wazuh mérésként.

## 6.6 Hibrid eredmények

A hibrid kiértékelés validált, de kontrollált offline összehasonlításként értelmezhető, nem éles eseménykorrelációként. A döntés az AE-Minimal és a rule_proxy predikcióinak unióján alapul.

| Metrika | Érték |
|---|---:|
| Precision | 0,620287 |
| Recall | 1,000000 |
| F1 | 0,765651 |
| False positive rate | 1,000000 |
| False negative rate | 0,000000 |
| Alert count | 448 628 |
| ROC-AUC | 0,640483 |

A hibrid eredmény ebben a mérési körben az AE-Minimal riasztási viselkedését örökli: a recall maximális, de a false positive rate is maximális. A hibrid ezért nem bizonyít éles üzemi javulást, hanem azt mutatja meg, hogyan kombinálható kontrollált offline módon a modellalapú és szabályproxy döntés.

## 6.7 Lab/replay esettanulmányok

A lab/replay demonstráció négy szcenáriót tartalmazott: `benign_activity`, `port_scan`, `ssh_bruteforce` és `combined_suspicious`. Mindegyik szcenárió 5 eseményt tartalmazott.

| Szcenárió | Esemény | Normal | Medium | High | Critical | Átlagos pontszám | Max. pontszám | ML riasztás | Szabályriasztás |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| benign_activity | 5 | 0 | 0 | 5 | 0 | 0,402420 | 0,449409 | 5 | 0 |
| combined_suspicious | 5 | 0 | 0 | 0 | 5 | 0,326466 | 0,449258 | 5 | 5 |
| port_scan | 5 | 0 | 0 | 0 | 5 | 0,344663 | 0,448297 | 5 | 5 |
| ssh_bruteforce | 5 | 0 | 0 | 0 | 5 | 0,451053 | 0,451846 | 5 | 5 |

A demonstráció alapján a scoring lánc képes eseményszintű kockázati kategóriát előállítani. Ugyanakkor a benign demonstrációs események is magas kockázati szintet kaptak, ami a küszöbök és a demonstrációs minták óvatos értelmezését indokolja. Ez replay-alapú demonstráció, nem éles SOC-validáció.

## 6.8 Teljesítménymérés

A teljesítménymérés batch scoring/inference mérés volt, nem event/perc alapú replay. A mérés 100, 500, 1000, 5000 és 10000 esemény feldolgozását vizsgálta 1, 10, 50 és 100 batch size mellett, három ismétléssel. Összesen 60 mérési sor készült.

| Mutató | Érték |
|---|---:|
| Legjobb mért áteresztőképesség | 415,65 esemény/másodperc |
| Ehhez tartozó batch size | 100 |
| Ehhez tartozó eseményszám | 500 |
| Legalacsonyabb p95 késleltetés | 2,5994 ms |
| CPU-idő eseményenként, minimum | 2,4057 ms |
| Csúcsmemória, maximum | 154,4844 MB |
| Hibás események összesen | 0 |

A mérés a scoring komponens feldolgozási költségét mutatja, nem a teljes SIEM/Wazuh end-to-end terhelhetőségét. A CPU mérés processz CPU-időként értelmezendő, nem teljes rendszer CPU százalékként. A memóriaértékek RSS, delta és csúcsmemória jellegűek, ahol az adott platformon elérhetők.

## 6.9 Hibaanalízis

Az AE-Minimal és AE-Context konfigurációk a választott F1-optimum küszöb mellett minden támadó mintát megtaláltak, de nagyon magas hamis pozitív arányt eredményeztek. Ez false positive szempontból kedvezőtlen, mert a benign minták nagy része riasztássá alakul.

A baseline_stat és a rule_proxy ezzel szemben alacsonyabb riasztásszámot adtak, de a recall értékük nagyon alacsony volt. Ez false negative szempontból kedvezőtlen, mert a támadó minták többsége negatív besorolást kapott.

A threshold érzékenység ezért központi tényező. Alacsonyabb küszöb növelheti a recall értéket, de növelheti a hamis pozitív riasztások számát is. Magasabb küszöb csökkentheti a riasztási terhelést, de növelheti a kihagyott támadások arányát. Az AE-Context magasabb ROC-AUC értéke rangsorolási szempontból kedvezőbb viselkedést jelezhet, de a kiválasztott küszöb mellett ez nem vezetett alacsony riasztásszámhoz.

## 6.10 Korlátok és érvényességi feltételek

A CIC-IDS2017 flow-alapú adathalmaz nem reprezentál minden modern vállalati vagy felhős környezetet. A Wazuh log- és eseményorientált adatmodellje eltér a CIC flow jellemzőitől, ezért natív Wazuh mérés csak címkézett Wazuh exporttal lenne védhető. A rule_proxy nem natív Wazuh teljesítménymérés.

A hibrid kiértékelés offline és proxy-alapú, nem éles eseménykorreláció. A lab/replay demonstráció kis elemszámú és kontrollált, ezért nem tekinthető éles SOC-validációnak. A teljesítménymérés hardverfüggő lokális batch scoring mérés, nem éles üzemi teljesítménygarancia.

További korlát a concept drift kezelése és a CTI egyszerűsítése. A prototípus nem valósít meg automatikus újratanítást, időbeli driftmonitorozást vagy teljes értékű threat intelligence életciklus-kezelést.

## 6.11 Összegző értékelés

A prototípus validáltan előállította az AE-Minimal, AE-Context, baseline_stat, rule_proxy és hybrid mérési eredményeket. A mérés megmutatta, hogy a gépi tanulási anomáliadetektálás mérnöki szinten integrálható IDS/SIEM jellegű feldolgozási láncba, és összehasonlítható egyszerű viszonyítási alapokkal.

Az eredmények ugyanakkor azt is megmutatták, hogy a magas recall önmagában nem elegendő. A false positive rate, az alert count és a küszöbválasztás döntően befolyásolja a gyakorlati használhatóságot. A prototípus a kutatási kérdésre kontrollált laboratóriumi környezetben ad választ: bemutatja az AI-alapú anomáliadetektálás integrálhatóságát és mérhetőségét, de nem bizonyít éles üzemi SOC-teljesítményt.
