# Adateredet és no-demo mérési szabályok

Ez a dokumentum rögzíti, hogy a diplomamunkában real-lab mérési eredményként csak tényleges lab futásból származó bemenetek és azokból számolt eredmények használhatók. Demonstrációs, kézzel összeállított, sample vagy tesztfixture adat nem szerepelhet real-lab kutatási eredményként.

## Mi számít real-lab eredménynek?

Real-lab eredménynek csak az tekinthető, amelynek bemenetei:

- `data/lab/lab_ground_truth.csv`: tényleges lab futás közben rögzített eseményablakok;
- `data/lab/lab_features.csv`: tényleges flow vagy Zeek bemenetből előállított feature fájl;
- `data/wazuh/alerts.jsonl`: tényleges Wazuh/OpenSearch export;
- `reports/real_measurement/measurement_provenance.json`: a bemenetek és fő kimenetek SHA256 azonosítóit rögzítő provenance fájl.

A provenance fájl nem igazolja önmagában a mérés szakmai helyességét, de rögzíti, hogy mely bemenetekből és eredményfájlokból készült az adott mérési csomag.

## Mi számít demo inputnak?

Demo inputnak számítanak többek között:

- `examples/lab/**`;
- `examples/scoring/**`;
- sablonok a `templates/lab/**` könyvtárban;
- bármely olyan fájl, amelynek útvonala vagy neve `sample`, `demo` vagy `fixture` jellegű eredetre utal.

Ezek használhatók bemutatási vagy fejlesztési célra, de nem használhatók real-lab mérési eredmény forrásaként.

## Mi számít automatikusan előállított offline eredménynek?

Offline vagy demonstrációs eredményfájl lehet például:

- `reports/lab/**`;
- `reports/final/**`;
- `reports/performance/**`;
- `results/performance/**`;
- `figures/final/**`;
- `reports/scored_events.jsonl`.

Ezek a prototípus működésének bemutatására vagy korábbi offline mérések dokumentálására alkalmasak, de nem helyettesítik a natív Wazuh exporttal készült real-lab eredményt.

## Miért nem használható az examples/lab real-lab eredményként?

Az `examples/lab` könyvtár kontrollált demonstrációs eseményeket tartalmazhat. Ezek célja a scoring és riportgenerálási lánc bemutatása, nem pedig a kutatási eredményként értelmezett Wazuh-only vs AE-only vs hibrid összehasonlítás. Real-lab eredményhez tényleges lab futásból származó ground truth, Wazuh alert export és flow/Zeek feature bemenet szükséges.

## Measurement provenance létrehozása

Tényleges mérés után:

```bash
make real-measurement-provenance
```

A cél létrehozza:

```text
reports/real_measurement/measurement_provenance.json
```

Ha bármely input hiányzik, vagy az útvonal demo/sample/fixture eredetre utal, a parancs hibával leáll.

## Repository hygiene audit

A repóban található demo és automatikusan előállított kimenetek áttekintéséhez:

```bash
make repo-hygiene-audit
```

Kimenetek:

- `reports/repo_hygiene/generated_artifact_audit.csv`
- `reports/repo_hygiene/generated_artifact_audit.md`
- `reports/repo_hygiene/generated_artifact_audit.json`

## No-demo guard

A real-lab eredmények no-demo ellenőrzéséhez:

```bash
make check-no-demo-real-results
```

Ha nincs real-lab comparison eredmény, a parancs sikeresen jelzi, hogy nincs ellenőrizhető eredmény. Ha van `results/real_comparison/metrics_comparison.csv`, akkor érvényes provenance fájl szükséges.

## Cleanup terv

A generált és demo kimenetek verziókezelési kezeléséhez:

```bash
make cleanup-generated-outputs-plan
```

A terv nem töröl fájlokat. Csak javaslatot ad arra, hogy egy állomány megtartható-e verziókezelésben, külön demo jelölést igényel-e, vagy inkább a beadási mellékletben / lokális futtatási könyvtárban maradjon.

## Mit szabad a dolgozatba beemelni?

Real-lab eredményként csak olyan táblázat, ábra vagy riport emelhető be, amelyhez:

- létezik `measurement_provenance.json`;
- a provenance `measurement_source` értéke `real_lab`;
- a bemeneti útvonalak nem `examples`, `templates`, `tests`, `sample`, `demo` vagy `fixture` eredetűek;
- a bemeneti SHA256 mezők kitöltöttek;
- a post-run QA `READY` vagy `READY_WITH_LIMITATIONS` státuszt ad.

## Mit tilos real-lab eredményként beemelni?

Nem emelhető be real-lab eredményként:

- demo vagy sample bemenetből készült eredmény;
- unit teszt fixture-ből származó eredmény;
- kézzel összeállított metrika;
- provenance nélküli real-lab jellegű comparison táblázat;
- Wazuh export nélküli natív Wazuh mérési állítás.

## Live integration kimenetek kezelése

A live integration réteg enriched alert és dashboard-ready kimeneteket állít elő. Ezek csak akkor használhatók dolgozati integrációs demonstrációként, ha ugyanarra a verified real-lab provenance-re épülnek, mint a Wazuh-only, AE-only és hibrid metrikai összehasonlítás.

Fontos értelmezési szabályok:

- Az enriched alert kimenet nem benchmark, hanem végponttól végpontig tartó mérnöki integrációs bizonyíték.
- Az unmatched alert arányt mindig közölni kell, mert megmutatja, hogy a Wazuh alert események mekkora része kapott AE scoringot.
- Dashboard screenshot vagy dashboard payload csak akkor használható, ha ugyanabból a verified real-lab mérésből származik.
- Provenance nélkül a live integration kimenet nem nevezhető végleges real-lab dolgozati bizonyítéknak.

Futtatás tényleges mérés után:

```bash
make final-live-integration
```

OpenSearch/dashboard demóhoz:

```bash
make final-live-integration-opensearch OPENSEARCH_PASSWORD=<jelszo>
```

## Verziózott demo fájlok és generált outputok kezelése

Az `examples/lab` és `examples/scoring` könyvtár verziózott demo inputokat tartalmazhat, mert ezek a fejlesztési és demonstrációs használatot segítik. Ezeket azonban minden dokumentációban demo bemenetként kell megjelölni, és nem használhatók real-lab mérési eredmény forrásaként.

A `reports/*`, `results/*` és `figures/final/*` alatti mérési outputok nem kerülhetnek Gitbe. Ezek futási kimenetek, amelyeket tényleges mérés után provenance és manifest alapján kell kezelni. A real-lab eredményeket nem a Git repository bizonyítja, hanem a mérési csomag:

- inputfájlok SHA256 azonosítói;
- `measurement_provenance.json`;
- `measurement_manifest.*`;
- post-run QA státusz;
- szükség esetén anonimizált beadási melléklet.

Gitben lévő demo input nem bizonyít kutatási eredményt. Kutatási eredményként csak verified real_lab provenance-hez kötött, tényleges lab futásból számolt metrika, ábra vagy riport használható.

## Lab session dokumentumok szerepe

A `reports/lab_session/` alatti session doctor riportok, session tervek, operátori naplósablonok és session összefoglalók nem mérési eredmények. Ezek a tényleges lab futás végrehajtását teszik reprodukálhatóbbá és auditálhatóbbá.

Fontos megkötések:

- A session plan nem helyettesíti a ground truth állományt.
- Az operátori napló nem helyettesíti a Wazuh exportot vagy a flow/Zeek bemenetet.
- A scenario marker export csak akkor valid ground truth forrás, ha a markerek tényleges lab futás közben lettek rögzítve.
- Eredménynek csak a provenance-ben rögzített inputokból számolt output tekinthető.
- A session réteg nem futtat támadó parancsot, és nem hoz létre mérési inputot.

## Final acceptance check szerepe

A final acceptance check célja annak ellenőrzése, hogy a rendszer a tényleges lab mérés előtt konzisztens és védett állapotban van-e. Ez a réteg nem hoz létre mérési bemenetet, nem exportál Wazuh alertet, és nem számol metrikát.

A failure mode ellenőrzések azt bizonyítják, hogy a rendszer hiányzó Wazuh export, hiányzó lab feature, demo input vagy provenance hiány esetén nem ad végleges READY státuszt. Ez azért fontos, mert a helyes hibázás ugyanúgy része a mérnöki megbízhatóságnak, mint a sikeres futás.

A real-lab pipeline-nak input nélkül hibáznia kell, mert ellenkező esetben fennállna a veszélye, hogy üres vagy demonstrációs fájlokból látszólagos eredmény születik. Ugyanebből az okból a demo inputot real-lab inputként minden guard rétegnek el kell utasítania.
