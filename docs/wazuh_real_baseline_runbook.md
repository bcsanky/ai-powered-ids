# Natív Wazuh baseline futtatási útmutató

Ez az útmutató a natív Wazuh riasztásokon alapuló szabályalapú baseline mérési láncot írja le. A cél az, hogy a Wazuh-only eredmény ugyanazon címkézett lab eseményekhez legyen köthető, mint az autoencoder-alapú és hibrid kiértékelések. A mérés csak akkor tekinthető natív Wazuh baseline-nak, ha a bemenet valódi Wazuh alert exportból és külön karbantartott ground truth állományból származik.

## Bemeneti állományok

A mérési lánc két bemenetet használ:

| Állomány | Szerep |
|---|---|
| `alerts.json` vagy `alerts.jsonl` | Wazuh alert export, amely a lab futás alatt keletkezett riasztásokat tartalmazza. |
| `lab_ground_truth.csv` | Címkézett lab eseménylista, amely az események időablakát, típusát és IP-címeit rögzíti. |

A ground truth CSV kötelező oszlopai:

| Oszlop | Leírás |
|---|---|
| `event_id` | Egyedi eseményazonosító. |
| `timestamp_start` | Az esemény kezdete UTC időbélyeggel. |
| `timestamp_end` | Az esemény vége UTC időbélyeggel. |
| `scenario` | A lab szcenárió neve, például `benign_activity`, `port_scan` vagy `ssh_bruteforce`. |
| `label` | `benign` vagy `attack`. |
| `attack_type` | Támadási típus, benign eseménynél üresen hagyható. |
| `source_ip` | Forrás IP-cím. |
| `target_ip` | Cél IP-cím. |

## Wazuh alert export

A Wazuh Dashboard vagy az indexer felületén a lab futás időintervallumára kell szűrni az alert indexeket. Az exportálásnál JSON vagy JSONL formátum használható. A parser több gyakori mezőelnevezést kezel, de a mért riasztások értelmezéséhez az alábbi mezők ajánlottak:

| Wazuh mező | Normalizált mező |
|---|---|
| `timestamp` vagy `@timestamp` | `timestamp` |
| `rule.id` | `rule_id` |
| `rule.level` | `rule_level` |
| `rule.description` | `rule_description` |
| `agent.name` | `agent_name` |
| `data.srcip`, `source.ip` vagy hasonló | `source_ip` |
| `data.dstip`, `destination.ip` vagy hasonló | `target_ip` |
| `full_log` vagy `message` | `full_log` |

## Futtatási parancsok

Az alapértelmezett útvonalak felülírhatók Makefile változókkal.

```bash
make wazuh-parse-alerts \
  WAZUH_ALERTS=data/wazuh/alerts.jsonl \
  WAZUH_WORK_DIR=data/processed/wazuh_real
```

Ez létrehozza:

```text
data/processed/wazuh_real/alerts_parsed.csv
```

A ground truth validálása és a riasztások korrelációja:

```bash
make wazuh-correlate \
  WAZUH_GROUND_TRUTH=data/lab/lab_ground_truth.csv \
  WAZUH_WORK_DIR=data/processed/wazuh_real \
  WAZUH_WINDOW_SECONDS=60
```

Ez létrehozza:

```text
data/processed/wazuh_real/lab_ground_truth_validated.csv
data/processed/wazuh_real/wazuh_correlated.csv
```

A Wazuh-only baseline kiértékelése:

```bash
make wazuh-eval-real \
  WAZUH_WORK_DIR=data/processed/wazuh_real \
  WAZUH_RESULTS_DIR=results/wazuh_real
```

Ez létrehozza:

```text
results/wazuh_real/metrics_summary.csv
results/wazuh_real/predictions.csv
results/wazuh_real/confusion_matrix.csv
results/wazuh_real/run_metadata.json
```

## Korrelációs szabály

Egy ground truth esemény Wazuh-pozitívnak számít, ha a Wazuh alert időbélyege az esemény `timestamp_start` és `timestamp_end + WAZUH_WINDOW_SECONDS` intervallumába esik, és az alert IP-mezői relevánsan illeszkednek az esemény forrás- vagy célcíméhez. Az alapértelmezett többletidőablak 60 másodperc.

A kimeneti predikciós fájl tartalmazza az első illeszkedő alert időpontját, a detektálásig eltelt időt, az illeszkedő rule azonosítókat, a legmagasabb rule szintet és az illeszkedő alert darabszámot.

## Metrikák értelmezése

A `metrics_summary.csv` a következő fő mutatókat tartalmazza:

| Metrika | Jelentés |
|---|---|
| `TP`, `FP`, `TN`, `FN` | A címkézett események és a Wazuh predikciók alapján számolt konfúziós mátrix elemei. |
| `precision` | A Wazuh által jelzett események közül mennyi volt támadás. |
| `recall` | A támadásként címkézett események közül mennyit jelzett a Wazuh. |
| `f1` | A precision és recall harmonikus átlaga. |
| `false_positive_rate` | Benign eseményekre adott téves riasztások aránya. |
| `false_negative_rate` | Támadások elmulasztásának aránya. |
| `alert_count` | Wazuh-pozitív események darabszáma. |
| `mean_ttd`, `median_ttd` | Detektálásig eltelt idő a valódi pozitív eseményeknél. |

## Korlátok

A mérés helyes értelmezéséhez az alábbi korlátokat külön rögzíteni kell:

- a Wazuh alert export teljessége és időszinkronja közvetlenül befolyásolja az eredményt;
- a ground truth eseményablakok kézi vagy lab-alapú pontossága meghatározza a korreláció megbízhatóságát;
- az IP-cím alapú illesztés nem helyettesíti a teljes eseménykorrelációt;
- a mérés Wazuh-only baseline-t ad a címkézett lab eseményekre, nem általános éles üzemi SOC-teljesítménymérést;
- ha nincs valódi Wazuh export, ezt a láncot nem szabad natív Wazuh eredményként bemutatni.
