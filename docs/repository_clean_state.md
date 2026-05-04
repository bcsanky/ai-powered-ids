# Repository clean state

Ez a dokumentum rögzíti, hogy a repóban milyen állományok maradhatnak verziókezelésben, és mely futási kimeneteket kell Gitből kizárni.

## Gitben maradó állományok

Verziókezelésben maradhatnak:

- forráskód: `ml/src/**`, `infra/**`;
- tesztek: `ml/tests/**`;
- konfigurációk: `experiments/**`;
- dokumentáció: `docs/**`, `README.md`, `results/README.md`, `reports/README.md`, `figures/README.md`;
- sablonok: `templates/lab/**`;
- explicit demo inputok: `examples/lab/**`, `examples/scoring/**`, egyértelmű demo jelöléssel.

## Gitből kizárt állományok

Nem kerülhetnek Git verziókezelésbe:

- futási riportok: `reports/final/`, `reports/lab/`, `reports/performance/`;
- real-lab futási kimenetek: `reports/real_measurement/`, `reports/real_measurement_qa/`, `reports/live_integration/`;
- mérési eredmények: `results/final/`, `results/performance/`, `results/real_comparison/`, `results/wazuh_real/`, `results/ae_lab/`, `results/hybrid_real/`;
- dolgozati ábrakimenetek: `figures/final/`;
- raw vagy érzékeny mérési inputok: `data/lab/lab_ground_truth.csv`, `data/lab/lab_features.csv`, `data/wazuh/alerts*.jsonl`;
- pcap, Zeek és flow nyersállományok;
- anonimizálási mapping fájlok.

## Miért nem verziózzuk a real-lab raw inputokat?

A real-lab bemenetek környezetfüggőek és érzékeny adatokat tartalmazhatnak: IP-címeket, hostneveket, Wazuh rule eseményeket vagy időbélyegeket. Ezeket provenance hash-sel és beadási mellékletben kell kezelni, nem általános Git tartalomként.

## Miért nem verziózzuk a generált metrikákat?

A metrikák futási kimenetek. Ha bekerülnek Gitbe, könnyen összekeverhetők demo, offline vagy korábbi állapotból származó eredményekkel. A dolgozatba kerülő eredményt a tényleges mérés után kell előállítani, majd provenance és manifest alapján kell archiválni.

## Valódi mérés csomagolása

Tényleges real-lab mérés után:

```bash
make final-real-measurement-package-with-provenance
```

Ez létrehozza a metrikákat, riportokat, provenance fájlt és manifestet. A beadási mellékletbe a manifest alapján kiválasztott állományok kerülhetnek.

## Ellenőrző parancsok

```bash
make repo-hygiene-check
make check-no-demo-real-results
make final-validate
```

A `repo-hygiene-check` ellenőrzi, hogy nincs-e tiltott verziózott futási kimenet. A `check-no-demo-real-results` megakadályozza, hogy demo/sample/fixture eredetű input real-lab bizonyítékként szerepeljen.

## Beadási melléklet

A beadási mellékletbe kerülhetnek:

- verified real_lab provenance fájl;
- mérési manifest és hash-ek;
- validált metrika CSV-k;
- dolgozatba beemelt ábrák;
- anonimizált riportok;
- szükség esetén raw inputok, ha adatvédelmi és intézményi szempontból megengedett.

Anonimizálni kell minden olyan állományt, amely IP-címet, hostnevet, felhasználónevet vagy környezetazonosító adatot tartalmazhat.

