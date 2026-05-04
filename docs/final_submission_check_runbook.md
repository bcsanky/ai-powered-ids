# Final submission check runbook

## Cél

A final submission check a beadás előtti követelmény-megfelelőségi és dokumentációs QA réteg. Nem mérési pipeline, nem hoz létre Wazuh exportot, nem számol metrikát, és nem helyettesíti a tényleges real-lab mérést.

## Mit ellenőriz?

- A feladatlap fő tartalmi pontjaihoz van-e bizonyíték vagy dokumentált runtime kimeneti terv.
- A dolgozati fejezetszerkezet lefedi-e az implementációt, módszertant, eredményeket, korlátokat és összegzést.
- Nincs-e túlzó vagy nem igazolt állítás a dokumentációban.
- A beadási mellékletbe szánt fájlok kezelése elkülöníti-e a forráskódot, konfigurációkat, riportokat és érzékeny raw inputokat.
- A bírálói kockázati kérdésekhez készült-e rövid válaszlista.

## Futtatás

```bash
make final-submission-check
```

Részenként:

```bash
make final-submission-requirements
make final-submission-structure
make final-submission-no-overclaiming
make final-submission-artifact-plan
make final-submission-readiness
make final-submission-risk-questions
```

## Státuszok

- `READY_FOR_REAL_MEASUREMENT`: a mérnöki pipeline beadás előtti állapotban van, de verified real-lab mérés még szükséges.
- `READY_FOR_THESIS_INTEGRATION`: van real-lab eredmény, de a Word dokumentumba emelés vagy kézi ellenőrzés még hátravan.
- `READY_FOR_SUBMISSION_REVIEW`: a fő QA ellenőrzések alapján beadási review-ra előkészíthető.
- `NOT_READY`: kritikus követelmény, túlzó állítás vagy beadási terv hiba miatt javítás szükséges.

## Provenance hiánya esetén

Ha nincs `measurement_provenance.json`, a rendszer nem lehet `READY_FOR_SUBMISSION_REVIEW` állapotban. Ilyenkor a helyes státusz jellemzően `READY_FOR_REAL_MEASUREMENT`, vagy hiba esetén `NOT_READY`.

## Word dokumentumban kézzel ellenőrizendő

- fejezetszámozás;
- ábra- és táblázatszámok;
- kereszthivatkozások;
- irodalomjegyzék;
- rövidítések jegyzéke;
- minden metrika egyezése a CSV forrásokkal;
- minden provenance-hez kötött állítás feltételessége.

## Bírálói kérdések használata

A `reports/final_submission_check/biraloi_risk_questions.md` védési és beadási felkészüléshez készült. A válaszok rövidek és óvatosak, de a tényleges metrikák ismeretében kézzel pontosíthatók.

