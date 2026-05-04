# Május 7-i scoring és riport validáció

## Dátum és branch

- Dátum: 2026-05-07
- Branch: `thesis/final`

## Napi cél

A napi cél a validált AE-Minimal modellre épülő bemutatható prototípus-réteg lezárása volt:

- FastAPI scoring végpont valós modellalapú pontozással;
- batch scoring JSONL és CSV bemeneti mintákhoz;
- szakértői jelentés és dashboard jellegű összefoglaló;
- demonstrációs futtatási leírás;
- implementációs jegyzetek a diplomamunka 5. fejezetéhez.

## FastAPI scoring állapota

A `/score` végpont nem használ fix mintaértéket. A pontozás az AE-Minimal modellfájlon, a preprocess állományon és a küszöbfájlon alapul. Ha ezek közül valamelyik nem érhető el, a `/health` válaszban `model_loaded: false` jelenik meg, a `/score` pedig hibával tér vissza.

A scoring válasz tartalmazza az eseményazonosítót, az anomáliapontszámot, a kiválasztott küszöb nevét és értékét, az ML riasztást, a szabályalapú jelzést, a szabályszintet, a kockázati szintet és a rövid szakmai indoklást.

## Sample input/output fájlok

Elkészültek a demonstrációs mintabemenetek:

- `examples/scoring/sample_events.jsonl`
- `examples/scoring/sample_events.csv`
- `examples/scoring/README.md`

A batch scoring kimenete:

- `reports/scored_events.jsonl`

## Szakértői riport kimenetek

Elkészültek a riportkimenetek:

- `reports/final/security_report.md`
- `reports/final/security_report.html`
- `reports/final/dashboard_summary.csv`

A riport a validált összehasonlító metrikákból és a pontozott demonstrációs eseményekből dolgozik. Nem éles SOC jelentés, hanem szakdolgozati demonstrációs összefoglaló.

## Futtatott validációk

```bash
make final-validate
make score-sample-events
make generate-security-report
make final-day7
```

Eredmény:

- `make final-validate`: sikeres, 34 teszt lefutott, 1 nem blokkoló sklearn konvergencia figyelmeztetéssel.
- `make score-sample-events`: sikeres, 6 demonstrációs esemény pontozása megtörtént.
- `make generate-security-report`: sikeres, Markdown, HTML és CSV riportkimenet létrejött.
- `make final-day7`: sikeres, a validációs, scoring és riportlépések egymás után lefutottak.
- Lokális FastAPI futtatási próba: a host Python környezetben a `fastapi` csomag nem volt telepítve, ezért a szolgáltatás közvetlen ASGI tesztje nem futott le.
- Dockeres futtatási próba: a `docker` parancs nem érhető el ebben a környezetben, ezért konténeres ellenőrzés nem történt.

## Létrejött vagy frissített fájlok

- `ml/src/scoring_runtime.py`
- `ml/src/score_events.py`
- `ml/src/generate_security_report.py`
- `infra/mlservice/app/config.py`
- `infra/mlservice/app/schemas.py`
- `infra/mlservice/app/scoring.py`
- `infra/mlservice/app/main.py`
- `infra/mlservice/requirements.txt`
- `infra/docker-compose.yml`
- `ml/tests/test_scoring_service_logic.py`
- `ml/tests/test_security_report.py`
- `examples/scoring/sample_events.jsonl`
- `examples/scoring/sample_events.csv`
- `examples/scoring/README.md`
- `docs/demo_runbook.md`
- `docs/thesis_implementation_notes.md`
- `docs/may07_validation.md`

## Nyitott korlátok

- A prototípus nem éles üzemi SOC rendszer.
- Natív Wazuh teljesítménymérés csak megfelelő címkézett Wazuh exporttal végezhető.
- A batch scoring demonstrációs célú, nem benchmark mérés.
- A riport szakdolgozati demonstráció, nem éles incidensjelentés.
- Dockeres FastAPI futtatás külön környezeti ellenőrzést igényel; a batch scoring azonos modellalapú pontozási logikát validál.

## Következő lépések május 8-ra

- A diplomamunka 5. fejezetének implementációs szövegét össze kell kötni a demonstrációs futtatási leírással.
- A 6. fejezetben rögzíteni kell, hogy mely validált futtatási könyvtárakból kerülnek be a táblázatok és ábrák.
- A dashboard vagy riport bemutatását szükség esetén képernyőképekkel lehet kiegészíteni.
