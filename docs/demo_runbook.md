# Demonstrációs futtatási leírás

Ez a leírás a validált AE-Minimal modellre épülő scoring szolgáltatás, batch scoring és szakértői riport bemutatásához használható. A cél a diplomamunka 5. fejezetében bemutatható prototípus-működés reprodukálható futtatása.

## Előfeltételek

- Python környezet a projekt függőségeivel.
- Elérhető AE-Minimal modellfájl, preprocess állomány és küszöbfájl.
- Elérhető végleges összehasonlító táblázat: `results/final/comparison/metrics_comparison.csv`.
- Opcionálisan Docker Compose a FastAPI szolgáltatás indításához.

## Batch scoring futtatása

A mintabemenetek a `examples/scoring/` könyvtárban találhatók. A JSONL alapú demonstrációs pontozás:

```bash
make score-sample-events
```

Elvárt kimenet:

```text
reports/scored_events.jsonl
```

A kimeneti állomány eseményenként tartalmazza az anomáliapontszámot, a kiválasztott küszöböt, az ML riasztást, a szabályalapú jelzést és a kockázati szintet.

## FastAPI szolgáltatás indítása Dockerrel

```bash
docker compose -f infra/docker-compose.yml up -d mlservice
```

Health ellenőrzés:

```bash
curl http://localhost:8000/health
```

Ha a modellfájlok nem érhetők el a szolgáltatás számára, a health válaszban `model_loaded: false` jelenik meg. Ebben az esetben a `/score` végpont nem ad vissza becsült anomáliapontszámot.

## FastAPI szolgáltatás indítása lokálisan

```bash
PYTHONPATH=.:infra/mlservice uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Pontozási példa:

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "event_id": "demo-001",
    "features": {
      "destination_port": 80,
      "flow_duration": 12345,
      "total_fwd_packets": 10,
      "total_backward_packets": 8,
      "flow_bytes_per_sec": 1200.5,
      "flow_packets_per_sec": 15.2,
      "protocol": "6"
    },
    "rule_flag": false,
    "rule_level": 0
  }'
```

## Riport generálása

```bash
make generate-security-report
```

Elvárt kimenetek:

```text
reports/final/security_report.md
reports/final/security_report.html
reports/final/dashboard_summary.csv
```

## Kimenetek felhasználása

- `reports/scored_events.jsonl`: eseményszintű scoring demonstráció az 5. fejezethez.
- `reports/final/security_report.md`: szakértői jelentés szöveges formában.
- `reports/final/security_report.html`: dashboard jellegű összefoglaló.
- `reports/final/dashboard_summary.csv`: riportoldali metrikaösszesítés.

Ezek a kimenetek a prototípus működését szemléltetik. A diplomamunka 6. fejezetében a benchmark jellegű mérési eredmények továbbra is a `results/final/` alatti validált futtatási könyvtárakból származnak.
