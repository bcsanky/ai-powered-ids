# Live smoke runbook

Ez a runbook a tényleges real-lab mérés előtti technikai readiness ellenőrzést írja le. A live smoke réteg nem mérési pipeline: nem állít elő Wazuh alertet, nem készít lab feature fájlt, nem számol metrikát, és nem futtat benchmarkot.

## Cél

A `make live-smoke` parancs azt ellenőrzi, hogy a mérési környezet technikailag alkalmas lehet-e a lab futtatás megkezdésére. Az ellenőrzés a konténeres környezetet, az ML service health végpontját, a modellállományokat, az OpenSearch kapcsolatot, a real-lab input útvonalakat és a fő Makefile workflow-k dry-run állapotát vizsgálja.

## Mit ellenőriz?

- Docker és Docker Compose parancs elérhetősége.
- Az `infra/docker-compose.yml` értelmezhetősége.
- Futó Wazuh/OpenSearch/mlservice konténerminták, ha a stack már fut.
- Az ML service `/health` végpontja, scoring kérés nélkül.
- Az AE-Minimal modell, preprocess és küszöbfájl betölthetősége.
- OpenSearch root endpoint és index count lekérdezés, ha a kapcsolati adatok meg vannak adva.
- A real-lab input útvonalak nem demo, sablon vagy teszt eredetűek.
- A fő mérési workflow-k `make -n` dry-run módban elérhetők.

## Mit nem csinál?

- Nem indít vagy állít le konténert.
- Nem hív `/score` végpontot.
- Nem exportál Wazuh alertet.
- Nem ír OpenSearch indexbe.
- Nem hoz létre `data/lab/lab_ground_truth.csv`, `data/lab/lab_features.csv` vagy `data/wazuh/alerts.jsonl` fájlt.
- Nem hoz létre mérési metrikát vagy dolgozati eredményt.

## Futtatás

```bash
make live-smoke
```

Opcionálisan az ML service kötelezővé tehető:

```bash
make live-smoke ML_SERVICE_REQUIRED=true
```

OpenSearch kapcsolat ellenőrzéséhez jelszó szükséges. Ha nincs megadva, a cél csak WARN riportot készít, hálózati lekérdezés nélkül:

```bash
make live-smoke-opensearch OPENSEARCH_PASSWORD=<jelszo>
```

## Kimenetek

- `reports/live_smoke/docker_environment_check.md`
- `reports/live_smoke/ml_service_health_check.md`
- `reports/live_smoke/model_artifacts_check.md`
- `reports/live_smoke/opensearch_connection_check.md`
- `reports/live_smoke/real_input_paths_check.md`
- `reports/live_smoke/make_workflow_dry_run.md`
- `reports/live_smoke/live_smoke_readiness.md`
- `reports/live_smoke/operator_smoke_brief.md`

Ezek futási kimenetek, Gitbe nem kerülnek.

## Státuszok

- `READY_FOR_LAB_EXECUTION`: nincs FAIL, és a környezet technikailag készen állhat a mérésre.
- `READY_WITH_WARNINGS`: nincs FAIL, de van olyan figyelmeztetés, amelyet mérés előtt érdemes ellenőrizni.
- `NOT_READY`: legalább egy kritikus ellenőrzés hibás.

## WARN esetén

WARN akkor is előfordulhat, ha a mérés előtt még természetesen hiányzik egy inputfájl, például a Wazuh alert export vagy a lab feature CSV. Ilyenkor a következő lépés a tényleges lab futás és az inputok előállítása.

## FAIL esetén

FAIL esetén a mérés megkezdése előtt javítani kell a hibát. Tipikus ok lehet tiltott real-lab input útvonal, hiányzó modellállomány, hibás Makefile workflow vagy kötelezőként jelölt, de el nem érhető szolgáltatás.

## Kapcsolódó célok

Ajánlott mérés előtti sorrend:

```bash
make final-acceptance
make final-submission-check
make live-smoke
make lab-session-prep
```

A tényleges mérés után:

```bash
make lab-session-after-capture
make final-real-measurement-package-with-provenance
make final-live-integration
make final-thesis-integration
```

A live smoke ellenőrzés nem helyettesíti a provenance fájlt, és nem bizonyít detektálási teljesítményt. Csak azt dokumentálja, hogy a környezet technikailag készen állhat a lab mérés megkezdésére.
