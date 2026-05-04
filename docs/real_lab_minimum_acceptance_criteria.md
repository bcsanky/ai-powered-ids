# Real-lab minimum acceptance criteria

Ez a dokumentum a real-lab mérés minimális elfogadási feltételeit rögzíti. Nem mérési eredmény, hanem döntési segédlet. Nyilvános vagy idegen IP-t tilos célozni; rövid szabály: tilos idegen IP.

Kötelező hivatkozások: provenance, Wazuh, `lab_ground_truth.csv`, `lab_features.csv`, `alerts.jsonl`, `final-real-measurement-package-with-provenance`, `final-measurement-quality`, tilos idegen IP.

## Minimális eseményszám

- Legalább 20 esemény.
- Legalább 5 benign esemény.
- Legalább 5 attack esemény.
- Legalább 4 különböző scenario.
- Legalább 2 attack scenario.

## Minimális fájlok

- `data/lab/lab_ground_truth.csv`
- `data/lab/lab_features.csv`
- `data/wazuh/alerts.jsonl`
- `reports/real_measurement/measurement_provenance.json`
- `results/real_comparison/metrics_comparison.csv`

## Kötelező kapuk

- `make final-real-measurement-package-with-provenance`
- `make final-live-integration`
- `make final-measurement-quality`
- `make final-thesis-integration`
- `make final-submission-check`
- `make repo-hygiene-check`
- `make final-validate`

## Elfogadható státuszok

Measurement quality:

- `MEASUREMENT_STRONG`: dolgozati eredményként használható, kézi ellenőrzéssel.
- `MEASUREMENT_USABLE_WITH_LIMITATIONS`: használható, de a korlátokat a 6. fejezetben világosan rögzíteni kell.
- `MEASUREMENT_WEAK`: további lab futás szükséges.
- `MEASUREMENT_NOT_READY`: nem használható eredményfejezethez.

Claim strength:

- `CLAIM_SUPPORTED_WITH_LIMITATIONS`: óvatos, a vizsgált lab mérésre korlátozott javulási állítás megengedett.
- `TRADEOFF_ONLY`: csak kompromisszum írható le, például recall javulás magasabb FPR vagy riasztásszám mellett.
- `CLAIM_NOT_SUPPORTED`: nem szabad javulást állítani; ezt kell leírni.
- `INSUFFICIENT_MEASUREMENT`: további mérés vagy hibajavítás szükséges.

Submission readiness:

- Valós mérés nélkül legfeljebb `READY_FOR_REAL_MEASUREMENT`.
- Dolgozati integrációhoz legalább `READY_FOR_THESIS_INTEGRATION`.
- Végső beadási review-hoz `READY_FOR_SUBMISSION_REVIEW`.

## Ha nem teljesül

- `CLAIM_NOT_SUPPORTED`: a dolgozatban nem szerepelhet javulási állítás.
- `MEASUREMENT_WEAK`: további mérési kör vagy több scenario szükséges.
- `NOT_READY`: nem használható a 6. fejezet eredményrészeként.

