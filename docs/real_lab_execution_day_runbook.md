# Real-lab execution day runbook

Ez a dokumentum a tényleges real-lab mérés napjára készült operátori runbook. Célja, hogy a Wazuh-only, AE-only és Wazuh+AE hibrid mérés kézzel, reprodukálhatóan, provenance-kompatibilisen és demo adatok nélkül fusson le.

Kötelező hivatkozások: provenance, Wazuh, `lab_ground_truth.csv`, `lab_features.csv`, `alerts.jsonl`, `final-real-measurement-package-with-provenance`, `final-measurement-quality`, tilos idegen IP.

## Scope

A mérés azt bizonyítja, hogy a laboratóriumi prototípus képes ugyanazon címkézett lab eseményeken összehasonlítani:

- natív Wazuh-only baseline eredményt;
- AE-Minimal lab scoring eredményt;
- Wazuh+AE hibrid döntési stratégiákat;
- end-to-end live integration kimenetet.

A mérés nem bizonyít hosszú idejű üzemi SOC-validációt, ipari általánosíthatóságot, általános érvényű teljesítményjavulást vagy üzemi bevezethetőséget.

## Biztonsági előfeltételek

- A támadásszimuláció kizárólag saját, izolált lab hálózatban végezhető.
- Nyilvános vagy idegen IP-t tilos célozni. Rövid szabály: tilos idegen IP.
- A target VM a saját lab része legyen, Wazuh agenttel.
- A brute force jellegű teszt csak saját teszt felhasználóra és alacsony próbálkozásszámmal történhet.
- A mérés előtt legyen időszinkron az attacker VM, target VM, Wazuh manager/OpenSearch és opcionális Zeek/flow collector között.
- Jelszót, tokent, raw Wazuh alertet, PCAP-ot és Zeek raw logot nem szabad Gitbe commitolni.

Tiltott műveletek:

- idegen vagy nyilvános IP scan vagy login próbálgatása;
- automatikus támadássorozat futtatása a repository scriptjeiből;
- demo/sample/fixture input real-lab eredményként használata;
- kézzel írt metrika vagy mesterséges Wazuh alert létrehozása;
- `examples/lab` vagy `examples/scoring` használata mérési bemenetként.

## Minimális lab topológia

- Attacker VM: csak izolált lab hálózatban, például `ATTACKER_IP`.
- Target VM: Wazuh agenttel, például `TARGET_IP`.
- Wazuh manager / OpenSearch: riasztások tárolása és exportja, például `WAZUH_MANAGER_IP`.
- Opcionális Zeek/flow collector: `conn.log` vagy flow CSV előállításához.

## Mérés előtti előkészítés

1. Ellenőrizd a branch-et és a tiszta mérnöki állapotot.
2. Futtasd:

```bash
make live-smoke
make final-acceptance
make final-submission-check
make repo-hygiene-check
```

3. Készíts session tervet:

```bash
make lab-session-prep LAB_SESSION_ID=<SESSION_ID> ATTACKER_IP=<ATTACKER_IP> TARGET_IP=<TARGET_IP> WAZUH_MANAGER_IP=<WAZUH_MANAGER_IP>
```

4. Nyisd meg és kövesd a `docs/real_lab_scenario_script.md` forgatókönyvet.

## Mérés alatti lépések

Minden szcenáriónál:

1. Marker start.
2. Kézi művelet saját izolált labban.
3. Marker end.
4. Rövid operátori megjegyzés az after action review sablonba.

A marker export a mérés végén történjen:

```bash
python -m ml.src.lab_session.scenario_marker_helper export --output data/lab/lab_ground_truth.csv
```

Ez a `lab_ground_truth.csv` csak akkor tekinthető valid ground truth forrásnak, ha a markerek tényleges futás közben lettek rögzítve.

## Mérés utáni export és pipeline

Wazuh/OpenSearch export:

```bash
make wazuh-export-opensearch WAZUH_EXPORT_START=<ISO_START> WAZUH_EXPORT_END=<ISO_END> OPENSEARCH_PASSWORD=<PASSWORD>
```

Vagy manuális export normalizálása:

```bash
make wazuh-export-from-file WAZUH_RAW_EXPORT=<RAW_EXPORT_JSON> WAZUH_EXPORT_START=<ISO_START> WAZUH_EXPORT_END=<ISO_END>
```

Feature build:

```bash
make lab-build-features-zeek
```

vagy:

```bash
make lab-build-features-flow-csv
```

Mérési pipeline:

```bash
make final-real-measurement-package-with-provenance
make final-live-integration
make final-measurement-quality
make final-thesis-integration
make final-submission-check
make final-submission-bundle
```

## Minőségi ellenőrzések

Minimálisan ellenőrizendő:

- `reports/real_measurement/measurement_provenance.json` létezik;
- `results/real_comparison/metrics_comparison.csv` elkészült;
- `reports/measurement_quality/measurement_quality_summary.md` nem NOT_READY;
- `reports/measurement_quality/research_claim_strength.md` nem állít többet, mint amit a metrikák igazolnak;
- `reports/thesis_integration/` kimenetek csak verified real_lab provenance mellett véglegesíthetők.

## Archiválás

Archiválandó, de nem automatikusan Gitbe commitolandó:

- measurement provenance;
- measurement manifest;
- quality gate riportok;
- thesis integration kimenetek;
- submission bundle ZIP;
- after action review.

Nem szabad Gitbe commitolni:

- `data/lab/lab_ground_truth.csv`;
- `data/lab/lab_features.csv`;
- `data/wazuh/alerts.jsonl`;
- Zeek raw log;
- PCAP/PCAPNG;
- secret, jelszó, token vagy tanúsítvány.
