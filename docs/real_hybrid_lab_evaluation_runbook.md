# Valódi lab-alapú Wazuh és AE hibrid kiértékelés

Ez az útmutató azt a mérési láncot írja le, amely ugyanazon címkézett lab eseményeken hasonlítja össze a natív Wazuh riasztásokat, az AE-Minimal modell offline pontozását és a Wazuh+AE hibrid döntéseket. A cél az, hogy a dolgozatban külön kezelhető legyen a CIC-IDS2017 alapú kontrollált összehasonlítás és a lab ground truth eseményekhez kötött Wazuh+AE mérés.

## Szükséges input fájlok

| Fájl | Szerep |
|---|---|
| `data/wazuh/alerts.jsonl` vagy `data/wazuh/alerts.json` | Natív Wazuh alert export a lab futás időintervallumából. |
| `data/lab/lab_ground_truth.csv` | Címkézett lab események időablakkal, IP-címekkel és benign/attack címkével. |
| `data/lab/lab_features.csv` | Ugyanezen események AE-Minimal feature vektora. |

A `lab_features.csv` kötelező oszlopai:

| Oszlop | Leírás |
|---|---|
| `event_id` | A ground truth eseménnyel egyező egyedi azonosító. |
| `destination_port` | Célport. |
| `flow_duration` | Flow időtartama. |
| `total_fwd_packets` | Előre irányú csomagok száma. |
| `total_backward_packets` | Visszairányú csomagok száma. |
| `flow_bytes_per_sec` | Byte/másodperc arány. |
| `flow_packets_per_sec` | Csomag/másodperc arány. |
| `protocol` | Protokoll értéke. |

Opcionális oszlopok:

| Oszlop | Leírás |
|---|---|
| `timestamp` | Eseményidő, ha rendelkezésre áll. |
| `source_ip` | Forrás IP-cím. |
| `target_ip` | Cél IP-cím. |
| `scenario` | Lab szcenárió neve. |

## Teljes futtatási parancs

A teljes lab-alapú Wazuh+AE kiértékelés egy Makefile célból futtatható:

```bash
make final-real-hybrid \
  WAZUH_ALERTS=data/wazuh/alerts.jsonl \
  WAZUH_GROUND_TRUTH=data/lab/lab_ground_truth.csv \
  LAB_FEATURES=data/lab/lab_features.csv
```

A cél sorrendben lefuttatja:

1. Wazuh alert export normalizálása.
2. Ground truth validálása és Wazuh alert korreláció.
3. Wazuh-only baseline kiértékelés.
4. Lab feature CSV validálása.
5. AE-Minimal offline pontozás a lab eseményeken.
6. AE-only lab metrikák számítása.
7. Hibrid Wazuh+AE stratégiák kiértékelése.
8. Egységes összehasonlító táblázat és ábrák előállítása.

## Kimeneti állományok

| Mérés | Kimeneti könyvtár | Fő fájl |
|---|---|---|
| Wazuh-only | `results/wazuh_real/` | `metrics_summary.csv`, `predictions.csv` |
| AE-Minimal lab | `results/ae_lab/` | `metrics_summary.csv`, `predictions.csv` |
| Hybrid real | `results/hybrid_real/` | `metrics_summary.csv`, `predictions.csv` |
| Összehasonlítás | `results/real_comparison/` | `metrics_comparison.csv`, `metrics_comparison.md`, PNG ábrák |

## Értelmezett konfigurációk

### Wazuh-only

A Wazuh-only baseline azt méri, hogy a natív Wazuh alert exportból hány címkézett lab eseményhez rendelhető releváns riasztás a megadott időablakon belül. Ez szabályalapú mérés, amelynek eredménye a Wazuh szabályrendszerétől, az export teljességétől és az időszinkrontól függ.

### AE-Minimal lab

Az AE-Minimal lab kiértékelés az előre betanított végleges AE-Minimal modellt használja. A `lab_features.csv` eseményeit a mentett `preprocess.pkl`, modellfájl és küszöbfájl alapján pontozza. A modell nem tanul újra, és nem használ kézzel megadott pontszámot.

### Hybrid OR

A `hybrid_or` stratégia akkor jelez pozitívat, ha a Wazuh vagy az AE-Minimal közül legalább az egyik pozitív jelzést ad:

```text
hybrid_pred = wazuh_pred OR ae_pred
```

Ez a stratégia jellemzően a recall növelésére alkalmas, de a false positive rate növekedhet.

### Hybrid weighted

A `hybrid_weighted` stratégia min-max normalizált AE anomáliapontszámot és Wazuh rule level értéket kombinál:

```text
hybrid_score = 0.6 * normalized_ae_score + 0.4 * normalized_wazuh_level
```

Az alapértelmezett döntési küszöb 0.5, amely `HYBRID_WEIGHTED_THRESHOLD` Makefile változóval módosítható.

### Hybrid priority

A `hybrid_priority` stratégia szakértői prioritási szintet rendel az eseményhez:

| Wazuh | AE | Prioritás |
|---|---|---|
| pozitív | pozitív | `critical` |
| pozitív | negatív | `high` |
| negatív | pozitív | `medium` |
| negatív | negatív | `normal` |

A bináris hibrid predikció pozitív, ha a prioritás nem `normal`.

## Korlátok

- Az AE lab scoring eredménye a `lab_features.csv` feature mapping pontosságától függ.
- Az AE-only mérés offline pontozás, ezért nem ad natív eseményidő-alapú time-to-detection mutatót.
- A Wazuh-only és hibrid TTD csak akkor értelmezhető, ha a Wazuh alert export időbélyegei pontosak.
- A hibrid mérés címkézett lab ground truth eseményeken történik, nem hosszú idejű éles SOC-validáció.
- Ha nincs valódi Wazuh alert export, a Wazuh-only eredmény nem tekinthető natív Wazuh baseline mérésnek.
