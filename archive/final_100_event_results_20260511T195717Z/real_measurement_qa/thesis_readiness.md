# Real-lab dolgozati beemelhetőség

Státusz: **READY**

Használható a 6. fejezetben: **igen**

Legjobb F1 szerinti hibrid konfiguráció: `Hybrid OR`.
Javult-e a Wazuh-only eredményhez képest: **igen**.
Romlott-e a false positive rate: **igen**.
Nőtt-e a riasztásszám: **igen**.

## Értelmezés
A vizsgált lab mérés alapján a legjobb hibrid konfigurációnál F1 javulás figyelhető meg a Wazuh-only baseline-hoz képest.

## Korlátok
- A mérés lab környezetben készült, nem hosszú idejű éles SOC-validáció.
- A hibrid eredmény az event_id alapú illesztés és az időszinkron pontosságától függ.
- Az AE-only ág offline scoring, ezért natív detektálási idő csak a Wazuh riasztásoknál értelmezhető.
- A metrikák csak az adott lab eseménykészletre vonatkoznak.
