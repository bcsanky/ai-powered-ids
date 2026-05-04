# Real-lab one day execution plan

Ez egy egynapos, reális mérési terv a tényleges real-lab futtatáshoz. Nem mérési eredmény, nem ground truth, és nem provenance. Nyilvános vagy idegen IP-t tilos célozni; rövid szabály: tilos idegen IP.

Kötelező hivatkozások: provenance, Wazuh, `lab_ground_truth.csv`, `lab_features.csv`, `alerts.jsonl`, `final-real-measurement-package-with-provenance`, `final-measurement-quality`, tilos idegen IP.

## 0. óra: környezet ellenőrzése

- Branch és repo állapot ellenőrzése.
- `make live-smoke`
- `make final-acceptance`
- `make final-submission-check`
- `make repo-hygiene-check`

Kimenet: környezeti readiness ismert, de még nincs mérési eredmény.

## 1. óra: Wazuh/agent/logging ellenőrzés

- Target VM Wazuh agent státusz.
- OpenSearch/Wazuh index elérhetőség.
- Időszinkron ellenőrzés.
- Rövid benign log esemény kézi ellenőrzése.

Ha nincs Wazuh alert, ne folytasd a mérési részt; először logging hibát kell javítani.

## 2. óra: benign szcenáriók

- `benign_ssh_login`
- `benign_package_update`
- ismétlés, ha több benign esemény kell a minimumhoz.

Minden eseményhez marker start/end szükséges.

## 3. óra: támadó szcenáriók

- `port_scan`
- `ssh_failed_logins`
- `ssh_bruteforce`
- `file_integrity_change`
- `privilege_change`

Minden támadó jellegű lépés saját izolált labra korlátozott, emberi operátor által végzett művelet.

## 4. óra: exportok

- Marker export: `data/lab/lab_ground_truth.csv`.
- Wazuh export: `data/wazuh/alerts.jsonl`.
- Zeek conn.log vagy flow CSV rendelkezésre állásának ellenőrzése.

Visszalépési terv üres Wazuh export esetén:

1. Ellenőrizd az időablakot.
2. Ellenőrizd az index patternt.
3. Ellenőrizd a target agent logküldését.
4. Ha továbbra is üres, ismételd a releváns szcenáriót, de ne pótold kézzel az alertet.

## 5. óra: feature build és real measurement package

```bash
make lab-session-after-capture
make lab-build-features-zeek
```

vagy:

```bash
make lab-build-features-flow-csv
```

Majd:

```bash
make final-real-measurement-package-with-provenance
```

## 6. óra: quality gate és live integration

```bash
make final-measurement-quality
make final-live-integration
```

Ellenőrizd:

- measurement quality státusz;
- claim strength;
- live integration unmatched arány;
- `metrics_comparison.csv`.

## 7. óra: thesis integration és artifact check

```bash
make final-thesis-integration
make final-submission-check
make final-submission-bundle
```

Ellenőrizd, hogy a dolgozati szövegek nem állítanak túlzó javulást.

## 8. óra: after action review

- Töltsd ki a `docs/real_lab_after_action_review_template.md` alapján az operátori összefoglalót.
- Archiváld a manifestet és provenance fájlt.
- Ellenőrizd, hogy raw/sensitive input nem került Gitbe vagy ZIP-be.
