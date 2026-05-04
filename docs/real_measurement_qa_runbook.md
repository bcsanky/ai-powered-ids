# Real-lab QA és dolgozati beemelési runbook

Ez a dokumentum a real-lab mérés előtti preflight ellenőrzés, a mérés utáni minőségkapu és a dolgozatba másolható táblázatok előállításának lépéseit foglalja össze. A QA-réteg nem készít mérési adatot, nem módosítja a metrika CSV-ket, és kizárólag a meglévő bemenetekből, illetve eredményfájlokból dolgozik.

## Preflight ellenőrzés

A tényleges lab mérés megkezdése előtt futtatandó:

```bash
make real-measurement-preflight
```

A preflight ellenőrzi:

- a szükséges projektfájlok és konfigurációk meglétét;
- az AE-Minimal modellhez szükséges modell- és preprocess állományok elérhetőségét;
- a lab sablonokat;
- a real-lab méréshez tartozó runbookokat;
- a szükséges input könyvtárakat;
- az opcionális bemeneti fájlok állapotát;
- OpenSearch környezeti változókat, ha erre külön szükség van.

Az opcionális mérési bemenetek hiánya mérés előtt nem hiba, hanem figyelmeztetés. A script a hiányzó alapkönyvtárakat létrehozza, és ezt a riportban jelzi.

Kimenetek:

- `reports/real_measurement_qa/preflight_report.md`
- `reports/real_measurement_qa/preflight_summary.csv`
- `reports/real_measurement_qa/preflight_metadata.json`

## Post-run QA

A tényleges mérési lánc lefutása után futtatandó:

```bash
make real-measurement-postrun-qa
```

Ez ellenőrzi:

- a kötelező Wazuh-only, AE lab, hibrid és comparison eredményfájlokat;
- a comparison táblában szereplő konfigurációkat;
- az `n_samples` konzisztenciát;
- a fő metrikák tartományát;
- a riasztásszám nemnegativitását;
- azt, hogy legalább egy benign és egy attack esemény szerepel-e;
- a Wazuh-only, AE-Minimal lab és hibrid sorok értelmezhetőségét.

## READY státuszok értelmezése

- `READY`: minden kötelező fájl és metrikai ellenőrzés rendben van.
- `READY_WITH_LIMITATIONS`: az eredmények értelmezhetők, de kisebb figyelmeztetés maradt.
- `NOT_READY`: hiányzik kulcsfájl, hibás a metrika, vagy a comparison táblázat nem alkalmas dolgozati beemelésre.

A `NOT_READY` státusz azt jelenti, hogy a real-lab mérés eredményeit nem szabad végleges mérési eredményként szerepeltetni a 6. fejezetben.

## Dolgozati táblázatok előállítása

Post-run QA után:

```bash
make real-measurement-thesis-tables
```

Kimenetek:

- `reports/real_measurement_qa/thesis_table_real_comparison.md`
- `reports/real_measurement_qa/thesis_table_wazuh_only.md`
- `reports/real_measurement_qa/thesis_table_ae_lab.md`
- `reports/real_measurement_qa/thesis_table_hybrid_strategies.md`
- `reports/real_measurement_qa/thesis_interpretation_bullets.md`

Ezek a fájlok Wordbe másolható Markdown táblázatokat és rövid értelmezési pontokat tartalmaznak. A számok a meglévő CSV-kből származnak, a hiányzó értékek `nincs adat` jelölést kapnak.

## Védési jegyzetek

A védéshez használható rövid kérdés-válasz jegyzet:

```bash
make real-measurement-defense-notes
```

Kimenet:

```text
reports/real_measurement_qa/defense_notes_real_measurement.md
```

A jegyzet röviden összefoglalja, mit mért a real-lab lánc, miért kellett Wazuh-only baseline, mit jelent az AE-only ág, hogyan értelmezhetők a hibrid stratégiák, és milyen korlátokat kell szóban is hangsúlyozni.

## Teljes dolgozati beemelési csomag

A mérés után futtatható cél:

```bash
make final-real-measurement-thesis-ready
```

Ez sorrendben futtatja:

- post-run QA;
- dolgozati táblázatok előállítása;
- védési jegyzetek előállítása.

Ha a teljes mérési csomagot is elő kell állítani a QA-val együtt:

```bash
make final-real-measurement-package-with-qa
```

## A 6. fejezet frissítése

A `thesis_readiness.md` alapján kell eldönteni, hogy a real-lab eredmények bekerülhetnek-e a diplomamunka 6. fejezetébe. Ha a státusz `READY` vagy `READY_WITH_LIMITATIONS`, a táblázatok és értelmezési pontok beemelhetők a megfelelő korlátokkal. Ha a státusz `NOT_READY`, a real-lab mérés csak folyamatleírásként vagy hiányzó mérésként említhető.

Mindig rögzíteni kell:

- a mérés lab környezetben készült;
- az AE-only ág offline scoring;
- a hibrid eredmény event_id alapú illesztésen alapul;
- az eredmények nem jelentenek hosszú idejű éles SOC-validációt.
