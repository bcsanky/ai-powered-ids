# Implementációs jegyzetek a diplomamunkához

Ez a dokumentum összefoglalja, hogy a prototípus mely komponensei írhatók le a diplomamunka implementációs és eredményfejezeteiben.

## Implementált komponensek

- Adatépítés CIC-IDS2017 flow adatokból, konfigurálható YAML fájlok alapján.
- AE-Minimal autoencoder mérési ág minimális flow jellemzőkészlettel.
- AE-Context autoencoder mérési ág egyszerű, timestamp nélküli kontextusjellemzőkkel.
- Statisztikai baseline a tanítóhalmaz középpontjától mért távolság alapján.
- Szabályalapú proxy baseline, ha nincs címkézett natív Wazuh export.
- Offline hibrid kiértékelés az AE-Minimal és a rule_proxy predikcióinak uniójával.
- FastAPI scoring szolgáltatás AE-Minimal modellalapú pontozással.
- Batch scoring parancssori feldolgozás JSONL és CSV bemenetre.
- Szakértői riport és dashboard jellegű összefoglaló.
- Lab/replay eseménysor pontozása kontrollált port scan, SSH brute force jellegű és kombináltan gyanús mintákon.
- Case study riportok és dolgozatba rendezett ábrakészlet.
- Batch scoring teljesítménymérési komponens lokális/labor méréshez.

## 5. fejezetben leírható témák

- A Wazuh komponensek és az ML szolgáltatás szerepe a rendszerarchitektúrában.
- Az adatépítési folyamat és a jellemzőképzés.
- Az autoencoder tanítási és küszöbválasztási folyamata.
- A FastAPI scoring végpont működése és hibakezelése hiányzó modell esetén.
- A batch scoring célja és bemeneti/kimeneti sémája.
- A szakértői jelentés szerepe mint demonstrációs riportkészítési réteg.
- Scoring szolgáltatás és batch scoring külön alfejezetben.
- Eseményalapú demonstráció kontrollált replay adatsorral.
- Szakértői riport és dashboard összefoglaló.
- Esettanulmányok port scan és SSH brute force jellegű mintázatokra.
- Teljesítménymérési komponens: benchmark bemenet, eseményszámok, batch size értékek és mérési metrikák.

## 6. fejezetben leírható témák

- AE-Minimal, AE-Context, baseline_stat, rule_proxy és hybrid összehasonlítása.
- Precision, recall, F1, hamis pozitív arány, riasztásszám és ROC-AUC értelmezése.
- Konfúziós mátrixok, pontszámeloszlások és ROC-görbék bemutatása.
- A kontextusjellemzők hatásának óvatos értelmezése.
- A rule_proxy és hibrid eredmények lehatárolása: ezek kontrollált offline kiértékelések, nem natív Wazuh teljesítménymérések.
- Lokális/labor teljesítménymérés: events_per_second, átlagos késleltetés, p95/p99 késleltetés és failed_events.

## Korlátok és lehatárolások

- A mérés CIC-IDS2017 flow-alapú adathalmazon történt.
- A Wazuh logorientált adatai és a CIC-IDS2017 flow jellemzői között adatmodellbeli eltérés van.
- A rule_proxy kontrollált flow-alapú szabályproxy, nem natív Wazuh futtatás.
- A hibrid kiértékelés azonos teszthalmaz-sorrenden alapul, nem éles eseménykorreláció.
- A scoring szolgáltatás laboratóriumi prototípus-réteg, nem éles üzemi SOC rendszer.
- A szakértői riport demonstrációs összefoglaló, nem éles incidensjelentés.
- A lab/replay esettanulmány kis elemszámú kontrollált eseménysor, nem natív Wazuh export és nem éles SOC-validáció.
- A teljesítménymérés hardver- és környezetfüggő lokális mérés, nem éles üzemi teljesítménygarancia.
