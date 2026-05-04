# 6. Eredmények és értékelés

Megjegyzés: a végleges számozást a Word-dokumentum tartalomjegyzéke szerint kell igazítani.

## 6.1 Kísérleti cél és értékelési szempontok

Az értékelés célja nem egyetlen modell abszolút bizonyítása, hanem a megvalósított laboratóriumi prototípus detektálási és mérési láncának összehasonlítható vizsgálata. A mérés azt elemzi, hogy az autoencoder-alapú megközelítés, a statisztikai viszonyítási alap (baseline), a szabályalapú proxy baseline és az offline hibrid kombináció milyen mérőszámokkal jellemezhető ugyanazon CIC-IDS2017 alapú tesztelési környezetben.

A validált összehasonlított konfigurációk:

- `ae_minimal`
- `ae_context`
- `baseline_stat`
- `rule_proxy`
- `hybrid`

A natív Wazuh baseline nem szerepel validált mérési eredményként, mert nem áll rendelkezésre megfelelő címkézett natív Wazuh export. Ennek helyét a kontrollált, flow-alapú szabályproxy tölti be, amelyet külön korláttal kell értelmezni.

Az értékelés fő metrikái:

- precision
- recall
- F1
- false positive rate
- false negative rate
- true positive rate
- true negative rate
- alert count
- ROC-AUC
- confusion matrix

Az alert count különösen fontos üzemeltetési szempontból, mert a pozitív jelzések mennyiségét mutatja. A false positive rate a benign minták téves riasztási arányát jelzi, ezért a gyakorlati használhatóság értelmezéséhez kiemelten fontos.

## 6.2 Adathalmaz és felosztás

A mérés a CIC-IDS2017 flow-alapú adathalmazon készült. Az adatfeldolgozási lánc a nyers CSV fájlokból tanító, validációs, kalibrációs és teszt adatrészt állít elő. Az előfeldolgozó kizárólag a tanító adatrészen illeszkedik, majd ugyanaz a transzformáció kerül alkalmazásra a validációs, kalibrációs és teszt adatrészre.

A validált AE-Minimal és AE-Context futtatások adatrészméretei:

| Adatrész | Mintaszám | Benign | Támadás |
|---|---:|---:|---:|
| Tanító adatrész | 1 589 922 | [IDE KERÜL: benign mintaszám, ha külön rögzítve van] | [IDE KERÜL: támadó mintaszám, ha külön rögzítve van] |
| Validációs adatrész | 340 698 | [IDE KERÜL: benign mintaszám, ha külön rögzítve van] | [IDE KERÜL: támadó mintaszám, ha külön rögzítve van] |
| Kalibrációs adatrész | 448 628 | 170 350 | 278 278 |
| Teszt adatrész | 448 628 | 170 350 | 278 278 |

A fő mérési összehasonlítás a teszt adatrészre épül. A teszt adatrész 448 628 mintát tartalmaz, ebből 278 278 támadó és 170 350 benign minta.

## 6.3 AE-Minimal eredményei

Az AE-Minimal konfiguráció a hét alap flow jellemzőt használja. A fő összehasonlításban az `f1_optimum` küszöbsor szerepel. A validált futtatási könyvtár:

```text
results/final/final-ae-minimal-v1/ae_v1_20260504_113852/
```

A fő metrikák:

| Metrika | Érték |
|---|---:|
| Precision | 0,620287 |
| Recall | 1,000000 |
| F1 | 0,765651 |
| False positive rate | 1,000000 |
| False negative rate | 0,000000 |
| Alert count | 448 628 |
| ROC-AUC | 0,605370 |

A konfúziós mátrix alapján az AE-Minimal ebben a küszöbbeállításban minden támadó mintát pozitívként jelöl, ugyanakkor minden benign tesztmintát is riasztásként sorol be. Ez magas recall értéket eredményez, de nagyon magas hamis pozitív aránnyal jár. Az eredmény jól szemlélteti, hogy az F1-optimalizált küszöb a vizsgált osztályeloszlás mellett erősen a recall irányába tolódhat.

A dolgozatban felhasználható ábrák:

- `reports/final/thesis_figures/ae_minimal_confusion_matrix.png`
- `results/final/final-ae-minimal-v1/ae_v1_20260504_113852/score_distribution.png`
- `results/final/final-ae-minimal-v1/ae_v1_20260504_113852/top_feature_frequency.png`

## 6.4 AE-Context eredményei

Az AE-Context konfiguráció az AE-Minimal alapjellemzőit egyszerű, timestamp nélküli kontextusjellemzőkkel egészíti ki. A gyakorisági jellemzők a tanító adatrészből illesztett port- és protokollgyakoriságokon alapulnak, míg a `packet_ratio` és `bytes_packets_ratio` soronként számított arányok.

A validált futtatási könyvtár:

```text
results/final/final-ae-context-v1/ae_v1_20260504_131741/
```

A fő metrikák:

| Metrika | Érték |
|---|---:|
| Precision | 0,620290 |
| Recall | 1,000000 |
| F1 | 0,765653 |
| False positive rate | 0,999988 |
| False negative rate | 0,000000 |
| Alert count | 448 626 |
| ROC-AUC | 0,764273 |

Az AE-Context eredményei az AE-Minimalhoz nagyon hasonló precision, recall és F1 értéket adnak, ugyanakkor a ROC-AUC érték magasabb. Ez arra utal, hogy a pontszámok rangsorolási képességében a kontextusjellemzők kedvezőbb elkülönítést adhatnak, de a kiválasztott `f1_optimum` küszöb mellett a riasztási mennyiség továbbra is nagyon magas. A különbséget ezért óvatosan kell értelmezni: nem állítható általános javulás minden üzemeltetési szempontból, de az eredmények alapján az AE-Context összehasonlíthatóvá vált az AE-Minimal ággal.

A dolgozatban felhasználható ábrák:

- `reports/final/thesis_figures/ae_context_confusion_matrix.png`
- `results/final/final-ae-context-v1/ae_v1_20260504_131741/score_distribution.png`
- `results/final/final-ae-context-v1/ae_v1_20260504_131741/top_feature_frequency.png`

## 6.5 Baseline eredmények

A statisztikai baseline egyszerű távolságalapú anomáliapontszámot használ. A validált eredmény:

| Metrika | Érték |
|---|---:|
| Precision | 0,514707 |
| Recall | 0,032008 |
| F1 | 0,060267 |
| False positive rate | 0,049299 |
| False negative rate | 0,967992 |
| Alert count | 17 305 |
| ROC-AUC | 0,475816 |

A statisztikai baseline jóval kevesebb riasztást állít elő, mint az autoencoder konfigurációk, de recall értéke alacsony. Ez azt jelzi, hogy a vizsgált egyszerű távolságalapú szabály a támadó minták nagy részét nem jelöli pozitívként.

A szabályalapú proxy baseline validált, de nem natív Wazuh teljesítménymérés. A mérés fő értékei:

| Metrika | Érték |
|---|---:|
| Precision | 0,525269 |
| Recall | 0,033391 |
| F1 | 0,062791 |
| False positive rate | 0,049299 |
| False negative rate | 0,966609 |
| Alert count | 17 690 |
| ROC-AUC | 0,573446 |

A szabályproxy hasonló riasztási mennyiséget ad, mint a statisztikai baseline, és valamivel magasabb F1 értéket ér el. A natív Wazuh baseline továbbra is hiányzik, ezért a dolgozatban azt kell rögzíteni, hogy natív Wazuh export hiányában kontrollált szabályalapú proxy baseline készült.

## 6.6 Hibrid eredmények

A hibrid kiértékelés az AE-Minimal és a szabályalapú proxy baseline predikcióinak unióját használja. A validált futtatási könyvtár:

```text
results/final/final-hybrid-v1/hybrid_20260504_144659/
```

A fő metrikák:

| Metrika | Érték |
|---|---:|
| Precision | 0,620287 |
| Recall | 1,000000 |
| F1 | 0,765651 |
| False positive rate | 1,000000 |
| False negative rate | 0,000000 |
| Alert count | 448 628 |
| ROC-AUC | 0,640483 |

Az unió alapú döntés ebben a mérésben az AE-Minimal riasztási viselkedését örökli, ezért a recall maximális, a false positive rate viszont szintén maximális. A hibrid eredmény validált, de offline proxy-alapú, azonos sorrendű teszthalmaz-predikciókra épül. Nem tekinthető éles eseménykorrelációs hibrid IDS-validációnak.

## 6.7 Lab/replay esettanulmányok

A lab/replay demonstráció négy kontrollált szcenáriót tartalmaz: benign aktivitás, port scan jellegű mintázat, SSH brute force jellegű mintázat és kombináltan gyanús eseménysor. A pontozott demonstrációs események alapján készült szcenárióösszesítés:

| Szcenárió | Események száma | Normal | Medium | High | Critical | ML riasztás | Szabályriasztás |
|---|---:|---:|---:|---:|---:|---:|---:|
| benign_activity | 5 | 0 | 0 | 5 | 0 | 5 | 0 |
| port_scan | 5 | 0 | 0 | 0 | 5 | 5 | 5 |
| ssh_bruteforce | 5 | 0 | 0 | 0 | 5 | 5 | 5 |
| combined_suspicious | 5 | 0 | 0 | 0 | 5 | 5 | 5 |

A demonstráció alapján a scoring lánc képes eseményszintű kockázati kategóriát és indoklást előállítani. Ugyanakkor a benign demonstrációs események is magas kockázati szintet kaptak, ami jól mutatja a küszöb- és mintaspecifikus értelmezés fontosságát. Ez a rész nem mérési benchmark és nem éles SOC-validáció, hanem a prototípus működésének bemutatása.

A dolgozatban felhasználható ábrák:

- `reports/final/thesis_figures/lab_timeline.png`
- `reports/final/thesis_figures/lab_risk_level_distribution.png`

## 6.8 Teljesítménymérés

A batch scoring komponens lokális/labor teljesítménymérése 100, 500, 1000 és 5000 eseményen, valamint 1, 10, 50 és 100 batch size beállítással készült. A mérés három ismétlést használt, és nem indított új modell-tanítást. A bemeneti események determinisztikus ismétlése kizárólag a feldolgozási kapacitás mérésére szolgált.

A fő eredmények:

| Mutató | Érték |
|---|---:|
| Legjobb mért áteresztőképesség | 399,04 esemény/másodperc |
| Ehhez tartozó batch size | 1 |
| Ehhez tartozó eseményszám | 100 |
| Legalacsonyabb p95 késleltetés | 2,7723 ms |
| Hibás események összesen | 0 |

A teljesítményriport összesített táblázata alapján az esemény/másodperc értékek a vizsgált környezetben nagyságrendileg 338 és 399 között mozogtak. A mérés CPU-időt, CPU-idő/esemény értéket, valamint platformfüggően elérhető memória RSS és csúcsmemória adatokat is rögzít. Az eredmények lokális hardver- és környezetfüggő mérések, nem éles üzemi teljesítménygaranciák, és nem tartalmazzák a teljes SIEM indexelési vagy incidenskezelési lánc költségét.

A dolgozatban felhasználható ábrák:

- `reports/final/thesis_figures/performance_throughput_by_batch_size.png`
- `reports/final/thesis_figures/performance_latency_by_batch_size.png`
- `reports/final/thesis_figures/performance_scoring_time_distribution.png`

## 6.9 Hibaanalízis

Az AE-Minimal és AE-Context konfigurációk a kiválasztott F1-optimum küszöb mellett minden támadó mintát pozitívként jelöltek, de rendkívül magas hamis pozitív arányt eredményeztek. Ennek egyik oka, hogy a kalibrációs halmazon optimalizált F1 a vizsgált osztályeloszlás mellett a recall maximalizálása felé tolhatja a döntési határt.

A statisztikai baseline és a szabályalapú proxy ezzel szemben kevesebb riasztást adtak, de alacsony recall érték mellett. Ez false negative szempontból kedvezőtlen, mert a támadó minták nagy része negatív besorolást kapott.

A threshold érzékenység ezért központi értelmezési kérdés. A küszöb alacsonyabbra állítása növelheti a recall értéket, de növeli a hamis pozitív riasztások számát is. A magasabb küszöb csökkentheti az üzemeltetési terhelést, de több támadó minta maradhat rejtve. A pontszámeloszlások és küszöbgörbék alapján kell megmutatni, hol helyezkedik el ez a kompromisszum.

Az AE-Context ROC-AUC értéke magasabb, mint az AE-Minimalé, ami a pontszámok rangsorolásában kedvezőbb viselkedésre utalhat. Ugyanakkor a kiválasztott riasztási küszöb mellett a gyakorlati riasztási mennyiség továbbra is nagyon magas, ezért a kontextusjellemzők hatását óvatosan kell értelmezni.

## 6.10 Érvényességi feltételek és korlátok

A mérés CIC-IDS2017 flow-alapú adatokon készült. Ez kutatási célra széles körben használt adathalmaz, de nem reprezentál minden modern vállalati, felhős vagy ipari hálózati környezetet. Az eredmények ezért kontrollált laboratóriumi következtetések.

A Wazuh logorientált adatmodellje és a CIC-IDS2017 flow jellemzői között jelentős eltérés van. Emiatt natív Wazuh teljesítménymérést csak akkor lehetne állítani, ha rendelkezésre állna címkézett Wazuh export. Jelen állapotban a rule_proxy csak kontrollált szabályalapú proxy baseline.

A hibrid kiértékelés offline proxy-alapú, és azonos teszthalmaz-sorrendet feltételez. Nem tartalmaz éles eseményazonosítón, időbélyegen vagy logkorreláción alapuló eseményösszekapcsolást.

A lab/replay demonstráció kis elemszámú kontrollált eseménysor. A port scan és SSH brute force jellegű minták bemutatják a scoring lánc működését, de önmagukban nem bizonyítanak éles üzemi detektálási teljesítményt.

A teljesítménymérés lokális hardver- és környezetfüggő batch scoring mérés. Nem tartalmaz natív Wazuh indexelést, dashboard terhelést, hosszú idejű stressztesztet vagy többfelhasználós SOC munkafolyamatot.

További korlátok:

- concept drift kezelése nincs implementálva;
- CTI kezelés egyszerűsített;
- a timestamp nélküli kontextusjellemzők nem helyettesítik az időablakos vagy hostalapú kontextust;
- az eredmények erősen függnek a küszöbválasztástól és az adathalmaz osztályeloszlásától.

## 6.11 Összefoglaló értékelés

A prototípus validáltan előállította az AE-Minimal, AE-Context, statisztikai baseline, szabályalapú proxy baseline és offline hibrid mérési eredményeket. A rendszer képes volt egységes metrikaszámításra, konfúziós mátrixok és összehasonlító ábrák előállítására, eseményszintű scoring demonstrációra, lab/replay esettanulmányokra és lokális teljesítménymérésre.

A mérés azt bizonyítja, hogy a gépi tanulási anomáliadetektálás mérnöki szinten integrálható egy IDS/SIEM jellegű prototípus feldolgozási láncába, és összehasonlíthatóvá tehető egyszerű viszonyítási alapokkal. A mérés nem bizonyítja, hogy a prototípus éles üzemi SOC rendszerként használható, és nem helyettesíti a natív Wazuh méréshez szükséges címkézett logexportot.

A dolgozat kutatási kérdéséhez kapcsolódóan az eredmények azt mutatják, hogy az autoencoder-alapú pontozás, a szabályalapú referencia és a hibrid döntési logika reprodukálható kísérleti környezetben vizsgálható. A fő gyakorlati tanulság a küszöbválasztás, a hamis pozitív arány és a riasztásszám kiemelt szerepe: a magas recall önmagában nem elegendő, ha közben az üzemeltetési terhelés túl nagy.

[IDE KERÜL: végleges fejezeti összegzés a témavezetői visszajelzések és a dolgozat szerkezete alapján]
