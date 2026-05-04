# Real-lab troubleshooting guide

Ez a dokumentum a mérésnapi hibák gyors diagnosztikáját támogatja. Nem hoz létre mérési inputot, nem készít metrikát, és nem helyettesíti a provenance ellenőrzést. Nyilvános vagy idegen IP-t tilos célozni; rövid szabály: tilos idegen IP.

Kötelező hivatkozások: provenance, Wazuh, `lab_ground_truth.csv`, `lab_features.csv`, `alerts.jsonl`, `final-real-measurement-package-with-provenance`, `final-measurement-quality`, tilos idegen IP.

## Wazuh agent nem látszik

Ellenőrizd:

- target VM fut-e;
- Wazuh agent szolgáltatás fut-e;
- agent regisztráció megtörtént-e;
- időszinkron rendben van-e.

Javasolt parancsok:

```bash
make live-smoke
make lab-session-doctor
```

## OpenSearch nem elérhető

Ellenőrizd:

- `OPENSEARCH_URL`;
- felhasználónév;
- jelszó csak shellben legyen;
- TLS beállítás labor környezetben indokolt-e.

Javasolt parancs:

```bash
make live-smoke-opensearch OPENSEARCH_PASSWORD=<PASSWORD>
```

## Wazuh export üres

Ellenőrizd:

- helyes-e `WAZUH_EXPORT_START` és `WAZUH_EXPORT_END`;
- az események tényleg a megadott időablakban történtek-e;
- Wazuh index pattern helyes-e;
- target agent küldött-e logot.

Ha üres az export, ne pótold kézzel az `alerts.jsonl` fájlt. Javítsd az exportot vagy ismételd a mérést.

## Ground truth timestamp eltér

Ellenőrizd:

- marker start/end tényleges futás közben történt-e;
- UTC timestamp használat rendben van-e;
- `lab_ground_truth.csv` validálható-e;
- nincs-e lezáratlan esemény.

Javasolt parancs:

```bash
python -m ml.src.wazuh_baseline.build_ground_truth --input data/lab/lab_ground_truth.csv --output data/processed/wazuh_real/lab_ground_truth_validated.csv
```

## Zeek/flow feature nem illeszkedik event_id-hez

Ellenőrizd:

- Zeek vagy flow CSV időablaka lefedi-e a marker időket;
- source/target IP értékek egyeznek-e;
- `lab_features.csv` minden event_id-hez tartalmaz-e sort.

Javasolt parancsok:

```bash
make lab-build-features-zeek
make lab-validate-real-inputs
```

vagy:

```bash
make lab-build-features-flow-csv
make lab-validate-real-inputs
```

## AE scoring nem talál modellt

Ellenőrizd:

- `artifacts/final/final-ae-minimal-v1` létezik-e;
- `data/processed/final/ae_minimal/preprocess.pkl` létezik-e;
- `thresholds.json` elérhető-e.

Javasolt parancs:

```bash
make live-smoke-model-artifacts
```

## Provenance hiányzik

Ha nincs `reports/real_measurement/measurement_provenance.json`, akkor a mérés nem tekinthető verified real-lab eredménynek.

Javasolt parancs:

```bash
make real-measurement-provenance
make check-no-demo-real-results
```

## check-no-demo-real-results hibázik

Ellenőrizd:

- nem `examples/`, `templates/`, `tests/` alatti input szerepel-e;
- nincs-e sample/demo/fixture path;
- az input SHA256 mezők megvannak-e.

## final-measurement-quality NOT_READY

Lehetséges ok:

- kevés esemény;
- kevés attack vagy benign esemény;
- event_id eltérés;
- hiányzó AE score;
- hiányzó Wazuh prediction;
- invalid provenance.

Megoldás: javítsd az inputot vagy futtass új valós lab mérést. Ne írj kézzel metrikát.

## Thesis integration NOT_READY

Ellenőrizd:

- metrics comparison elkészült-e;
- measurement provenance valid-e;
- quality gate nem NOT_READY-e.

Javasolt parancs:

```bash
make final-thesis-integration
```

## Live integration unmatched arány magas

Ellenőrizd:

- van-e event_id az alertben;
- timestamp + IP mapping működik-e;
- ground truth időablakok elég pontosak-e;
- `--allow-time-only-match` használata indokolt-e.

Magas unmatched arány esetén a dashboard-ready kimenet csak korlátozott integrációs demonstrációként használható.
