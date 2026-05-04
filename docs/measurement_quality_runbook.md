# Mérési minőségellenőrzési runbook

Ez a runbook a real-lab mérés utáni minőségi ellenőrző réteget írja le. A mérési quality gate nem futtat új mérést, nem exportál Wazuh alertet, nem készít lab feature állományt, és nem számol új predikciót. Csak a már meglévő, provenance-szel igazolt inputokat és eredményfájlokat olvassa.

## Mikor kell futtatni?

A quality gate a tényleges mérési csomag elkészülése után futtatható:

```bash
make final-real-measurement-package-with-provenance
make final-measurement-quality
```

Ha nincs `reports/real_measurement/measurement_provenance.json`, vagy az nem `real_lab` forrást rögzít, az ellenőrzés nem adhat használható mérési státuszt.

## Mit ellenőriz?

- A ground truth eseményszámát, benign/attack arányát és scenario-lefedettségét.
- A ground truth, lab feature, Wazuh-only, AE-only és hibrid predikciók event_id egyezését.
- Az AE scoring lefedettségét és az anomáliapontszámok meglétét.
- A Wazuh alert matching értelmezhetőségét.
- A Wazuh-only, AE-Minimal lab és hibrid metrikák tartományait és konzisztenciáját.
- A TTD értékek meglétét, negatív vagy túl nagy értékeit.
- A kutatási állítás erősségét a comparison metrikák alapján.

## Futtatás

```bash
make final-measurement-quality
```

Részellenőrzések külön is futtathatók:

```bash
make measurement-quality-scenario-coverage
make measurement-quality-feature-alert-alignment
make measurement-quality-metric-consistency
make measurement-quality-ttd
make measurement-quality-claim-strength
make measurement-quality-summary
make measurement-quality-thesis-notes
```

## Minőségi státuszok

- `MEASUREMENT_STRONG`: a fő minőségi feltételek teljesülnek, és nincs kritikus warning vagy fail.
- `MEASUREMENT_USABLE_WITH_LIMITATIONS`: a mérés használható, de korlátokat külön meg kell nevezni.
- `MEASUREMENT_WEAK`: a mérés értelmezhető lehet, de az elemszám, lefedettség vagy állítási erő gyenge.
- `MEASUREMENT_NOT_READY`: kritikus hiány vagy konzisztenciahiba miatt nem használható kutatási állítás alátámasztására.

## Kutatási állítás kategóriái

- `CLAIM_SUPPORTED_WITH_LIMITATIONS`: a vizsgált lab mérés alapján F1 szerint óvatos, korlátokkal kezelt hibrid javulás figyelhető meg.
- `TRADEOFF_ONLY`: a recall javulhat, de FPR vagy riasztási terhelés romlik, ezért kompromisszumként kell értelmezni.
- `CLAIM_NOT_SUPPORTED`: a hibrid F1 nem jobb a Wazuh-only viszonyítási alapnál, vagy a különbség nem éri el a konfigurált küszöböt.
- `INSUFFICIENT_MEASUREMENT`: a metrikák hiányosak vagy nem elég konzisztens a mérés.

## Kimenetek

- `reports/measurement_quality/scenario_coverage.md`
- `reports/measurement_quality/feature_alert_alignment.md`
- `reports/measurement_quality/metric_consistency.md`
- `reports/measurement_quality/ttd_quality.md`
- `reports/measurement_quality/research_claim_strength.md`
- `reports/measurement_quality/measurement_quality_summary.md`
- `reports/measurement_quality/thesis_measurement_quality_notes.md`

Ezek futási kimenetek, Gitbe nem kerülnek. Nem új mérési eredmények, hanem a meglévő real-lab mérés minőségét és értelmezhetőségét ellenőrző riportok.

## Kapcsolat a 6. fejezettel

A `thesis_measurement_quality_notes.md` rövid, dolgozatba illeszthető megjegyzéseket ad a mérési lefedettségről, a szcenáriók korlátairól, a Wazuh/AE/hibrid összehasonlítás megbízhatóságáról és a kutatási állítás erősségéről. A szöveget a 6. fejezet hibaanalízis és korlátok részéhez érdemes felhasználni.

## Miért nem generál adatot?

A quality gate célja éppen az, hogy megakadályozza a provenance nélküli vagy nem kellően erős mérési eredmények túlértelmezését. Ezért csak meglévő CSV/JSON kimenetekből dolgozik, és provenance nélkül nem minősíti a mérést használhatónak.
