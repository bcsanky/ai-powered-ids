# Május 6-i szabályalapú proxy és hibrid validáció

Validáció dátuma: 2026-05-06  
Branch: `thesis/final`

## Napi cél

A napi cél a szabályalapú proxy baseline és az offline hibrid kiértékelés minimális, védhető változatának lezárása volt. A validáció célja, hogy a diplomamunka eredményfejezetében az AE-Minimal, AE-Context, statisztikai baseline, rule_proxy és hibrid mérési ágak egységes összehasonlításban szerepelhessenek.

## Valódi Wazuh export ellenőrzése

A következő lehetséges bemenetek ellenőrzése megtörtént:

- `data/processed/final/wazuh/wazuh_eval.csv`
- `data/processed/final/wazuh/wazuh_eval.parquet`
- `data/processed/final/wazuh/wazuh_eval.jsonl`
- `data/processed/final/wazuh/wazuh_alerts.csv`
- `data/processed/final/wazuh/wazuh_alerts.parquet`
- `data/processed/final/wazuh/wazuh_alerts.jsonl`

Nem állt rendelkezésre megfelelő címkézett natív Wazuh export. Emiatt natív Wazuh teljesítménymérés nem történt. A mérési lánc kontrollált, flow-alapú szabályproxy baseline-t használ Wazuh-szerű exporttal.

## Szabályalapú proxy baseline indoklása

A rule_proxy célja egy egyszerű, reprodukálható szabályalapú összehasonlítási pont létrehozása a feldolgozott CIC-IDS2017 flow adatokon. A proxy pontszám a numerikus, már előfeldolgozott jellemzők abszolút értékének soronkénti maximuma. A küszöb a validációs adatrész alapján készül, ezért a teszt címkéi nem vesznek részt a szabály illesztésében.

Ez kontrollált offline kiértékelés, nem natív Wazuh futás és nem Wazuh szabálykészlet teljesítménymérése.

## Futtatott parancsok

```bash
git branch --show-current
git status --short
make final-validate
make dataset CONFIG=experiments/final/ae_minimal.yaml
make final-rule-proxy
make final-hybrid
make final-compare
```

Az AE-Minimal adatépítés azért futott újra, mert a lokális `data/processed/final/ae_minimal/` bemeneti könyvtár nem volt jelen. Ez nem indított új AE tanítást, csak a szabályproxyhoz szükséges feldolgozott adatszeleteket állította elő.

## Rule_proxy export eredménye

Proxy export:

```text
data/processed/final/rule_proxy/wazuh_like_rule_eval.csv
```

Metaadat:

```text
data/processed/final/rule_proxy/rule_proxy_metadata.json
```

Fő metaadatértékek:

- `method`: `max_abs_standardized_feature_rule`
- `threshold_quantile`: `0.95`
- `threshold_value`: `3.4674376114313468`
- `fitted_on`: `validation split`
- `evaluated_on`: `test split`
- megjegyzés: kontrollált szabályalapú proxy baseline, nem natív Wazuh export

## Rule_proxy baseline eredménykönyvtár

```text
results/final/final-rule-proxy-v1/baseline_wazuh_20260504_144645/
```

Fő kimeneti állományok:

- `metrics_summary.csv`
- `predictions.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `score_distribution.png`
- `threshold_curve.csv`
- `threshold_curve.png`
- `roc_curve.png`
- `run_metadata.json`

## Hybrid eredménykönyvtár

```text
results/final/final-hybrid-v1/hybrid_20260504_144659/
```

Fő kimeneti állományok:

- `metrics_summary.csv`
- `predictions.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `score_distribution.png`
- `roc_curve.png`
- `run_metadata.json`

A hibrid döntés az AE-Minimal és a rule_proxy predikciók unióját használja:

```text
hybrid_pred = ae_pred == 1 OR rule_pred == 1
```

A hibrid pontszám a normalizált AE-pontszám és a normalizált rule_proxy pontszám maximuma.

## Final comparison státuszösszefoglaló

| Konfiguráció | Státusz | Megjegyzés |
|---|---|---|
| `ae_minimal` | ok | Validált AE-Minimal futtatás alapján. |
| `ae_context` | ok | Validált AE-Context futtatás alapján. |
| `baseline_stat` | ok | Validált statisztikai baseline futtatás alapján. |
| `rule_proxy` | ok | Kontrollált flow-alapú szabályproxy alapján. |
| `hybrid` | ok | Offline, rule_proxy-alapú hibrid kiértékelés alapján. |
| `baseline_wazuh_real` | missing | Nincs megfelelő címkézett natív Wazuh export. |

## Dolgozatbeli értelmezés

A rule_proxy nem natív Wazuh teljesítménymérés. A dolgozatban szabályalapú proxy baseline-ként kell megnevezni, amely kontrollált offline környezetben Wazuh-szerű exportot állít elő a flow-alapú adatokból.

A hibrid eredmény offline, kontrollált teszthalmaz-sorrenden alapul. Nem éles eseménykorrelációt mér, hanem azt mutatja be, hogyan kombinálható az AE-Minimal és a szabályproxy predikciója egységes kiértékelési állományokban.

## Nyitva maradt feladatok május 7-re

- Az implementációs fejezet írása.
- Az eredményfejezet első táblázatainak beemelése.
- A dashboard és reporting rész előkészítése, ha a diplomamunka szerkezete igényli.
- Natív Wazuh export előkészítése csak akkor, ha rendelkezésre áll címkézett, kiértékelhető Wazuh adat.
