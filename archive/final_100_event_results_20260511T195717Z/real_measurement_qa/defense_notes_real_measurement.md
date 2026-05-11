# Real-lab mérés védési jegyzet

## Mit mértünk?
Ugyanazon címkézett lab eseményeken hasonlítottuk össze a Wazuh-only szabályalapú baseline-t, az AE-Minimal offline lab pontozást és a hibrid Wazuh+AE stratégiákat.

## Miért kellett Wazuh-only baseline?
A Wazuh-only eredmény adja azt a szabályalapú viszonyítási pontot, amelyhez az AE-only és a hibrid döntések mérnöki szempontból hasonlíthatók.

## Mit jelent az AE-only eredmény?
Az AE-only ág azt mutatja meg, hogy a végleges AE-Minimal modell a lab feature-ök alapján, Wazuh riasztási információ nélkül milyen besorolást ad.

## Mit jelent a hibrid stratégia?
A hibrid stratégiák a Wazuh riasztást és az AE pontozást kombinálják. Az OR, weighted és priority változat eltérő döntési szabályt képvisel.

## Javult-e a Wazuh eredmény?
A vizsgált lab mérés alapján a `Hybrid OR` F1 értéke magasabb volt a Wazuh-only eredménynél.
Wazuh F1: 0.3514; legjobb hibrid F1: 0.5714.

## Miért lehet magas a false positive rate?
A szabályalapú és hibrid döntések érzékenyek lehetnek a lab eseményablakok időzítésére, a feature mapping pontosságára és a Wazuh rule-ok konfigurációjára.

## Miért nem éles SOC bizonyítás?
A mérés kontrollált lab környezetben készült, rövid mérési ablakokkal. Nem hosszú idejű, változó terhelésű éles SOC-validáció.

## Mi a legfontosabb mérnöki eredmény?
A prototípus ugyanazon event_id készleten képes Wazuh-only, AE-only és hibrid eredményeket előállítani, majd ellenőrzött metrikákkal összehasonlítani.

## Milyen továbbfejlesztés lenne indokolt?
Nagyobb lab eseménykészlet, hosszabb mérési időablak, stabilabb eseménykorrelációs kulcsok és élesebb Wazuh exportfolyamat növelné a mérés külső érvényességét.

## Beemelhetőségi kivonat
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

## Ellenőrző kérdések
- Mit mértünk?
- Miért kellett Wazuh-only baseline?
- Mit jelent az AE-only eredmény?
- Mit jelent a hibrid stratégia?
- Javult-e a Wazuh eredmény?
- Miért lehet magas a false positive rate?
- Miért nem éles SOC bizonyítás?
- Mi a legfontosabb mérnöki eredmény?
- Milyen továbbfejlesztés lenne indokolt?
