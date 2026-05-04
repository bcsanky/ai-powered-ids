# Submission bundle runbook

Ez a runbook a beadási mellékletként használható ZIP-csomag előállítását írja le. A csomagoló réteg nem mérési pipeline: nem exportál Wazuh alertet, nem állít elő lab feature-t, nem számol metrikát, és nem módosít meglévő eredményfájlokat.

## Cél

A `submission_bundle` workflow egy ellenőrzött csomagot készít a repository olyan részeiből, amelyek biztonságosan mellékelhetők:

- forráskód;
- unit tesztek;
- végleges konfigurációk;
- lab sablonok;
- dokumentáció és runbookok;
- egyértelműen demo inputként jelölt példák;
- verified real_lab provenance esetén engedélyezett riportok és manifestek.

## Mi nem kerülhet bele

A ZIP nem tartalmazhat raw vagy érzékeny mérési inputot:

- `data/**`;
- `raw/**`;
- Wazuh alert exportok;
- PCAP/PCAPNG állományok;
- Zeek raw log könyvtárak;
- tanított modellbinárisok;
- `.env`, tanúsítvány, kulcs vagy token jellegű fájlok;
- redaction mapping fájl.

Az `examples/lab` és `examples/scoring` könyvtárak csak demonstrációs példaként kerülhetnek a csomagba. Ezek nem mérési eredmények.

## Futtatás

```bash
make final-submission-bundle
```

A target először lefuttatja a beadási QA, final acceptance és repo hygiene ellenőrzéseket, majd elkészíti:

- `reports/submission_bundle/submission_candidates.md`;
- `reports/submission_bundle/submission_candidate_validation.md`;
- `reports/submission_bundle/submission_manifest.md`;
- `dist/submission/SUBMISSION_README.md`;
- `dist/submission/ai_powered_ids_submission_bundle.zip`;
- `reports/submission_bundle/submission_zip_inspection.md`;
- `reports/submission_bundle/submission_bundle_report.md`.

## Provenance kapcsolat

Provenance nélkül a csomag forráskódot, konfigurációt, dokumentációt, sablonokat és demo példákat tartalmazhat. Runtime mérési riportok és eredménytáblák csak akkor kerülhetnek be, ha a `reports/real_measurement/measurement_provenance.json` verified real_lab státuszú.

## Kézi ellenőrzés beadás előtt

- A ZIP inspection PASS legyen.
- A ZIP ne tartalmazzon raw Wazuh alertet, PCAP-ot vagy titkos állományt.
- A `SUBMISSION_README.md` legyen a ZIP gyökerében.
- Ha real-lab riport is kerül a csomagba, ellenőrizni kell a provenance és manifest hash-eket.
- Érzékeny IP-címeket vagy hostneveket tartalmazó riportoknál anonimizált változatot kell használni.
