# Final acceptance runbook

## Cél

A final acceptance réteg a tényleges lab mérés előtti release-candidate állapotot ellenőrzi. Nem állít elő mérési adatot, nem exportál Wazuh alertet, nem számol új metrikát, és nem helyettesíti a valós lab futtatást.

## Mit ellenőriz?

- A szükséges Makefile célok meglétét és a végső ellenőrzési sorrendet.
- Azt, hogy hiányzó input esetén a real-lab pipeline hibával álljon le.
- Azt, hogy demonstrációs vagy fixture jellegű input ne kerülhessen real-lab bemenetként feldolgozásra.
- Azt, hogy provenance nélkül ne lehessen READY státuszú dolgozati beemelhetőség.
- Azt, hogy a futási output könyvtárak Gitből kizártak legyenek.
- Azt, hogy a dokumentáció ne sugalljon nem létező vagy nem igazolt real-lab eredményt.

## Mit nem csinál?

- Nem hoz létre `data/lab/lab_ground_truth.csv` fájlt.
- Nem hoz létre `data/lab/lab_features.csv` fájlt.
- Nem hoz létre Wazuh alert exportot.
- Nem indít traininget.
- Nem indít benchmarkot.
- Nem ír metrikát a fő `results/` vagy `reports/` mérési könyvtárakba.

## Futtatás

```bash
make final-acceptance
```

Részenként:

```bash
make final-acceptance-make-targets
make final-acceptance-failure-modes
make final-acceptance-provenance-policy
make final-acceptance-docs
make final-acceptance-readiness
make final-acceptance-brief
```

## Státuszok

- `READY_FOR_REAL_LAB_RUN`: a kritikus ellenőrzések sikeresek, a rendszer készen áll a tényleges lab mérésre.
- `READY_WITH_WARNINGS`: nincs kritikus hiba, de van kézzel kezelendő figyelmeztetés.
- `NOT_READY`: legalább egy kritikus ellenőrzés hibás; a lab mérés előtt javítani kell.

## Kapcsolódó célok

- `make repo-hygiene-check`: tracked futási outputok és no-demo guard ellenőrzése.
- `make final-validate`: kód- és tesztvalidáció.
- `make final-real-measurement-package-with-provenance`: tényleges mérés után futtatható mérési csomag.
- `make final-thesis-integration`: verified provenance után futtatható dolgozati fejezetrész-készítés.

## NOT_READY esetén

Először a `reports/final_acceptance/release_candidate_readiness.md` és az egyes részriportok alapján kell javítani a hibát. A final acceptance státusz csak azt jelzi, hogy a rendszer mérésre kész-e; nem jelenti azt, hogy a dolgozat vagy a real-lab mérés már elkészült.

## Execution brief

A `reports/final_acceptance/real_lab_execution_brief.md` rövid operátori sorrendet ad a mérés napjára. Ez nem futtat parancsokat automatikusan, hanem ellenőrzött lépéssort ad a tényleges lab futtatáshoz.

