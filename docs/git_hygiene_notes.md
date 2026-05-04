# Git hygiene jegyzet

## Cél

Ez a jegyzet a május 8-i lab/replay és riportkimenetek verziókezelési állapotát rögzíti. A cél annak ellenőrzése, hogy ne kerüljenek a repositoryba túl nagy vagy feleslegesen tárolt automatikusan előállított eredményfájlok.

## Ellenőrzött fájlok mérete

| Fájl | Méret byte-ban | Körülbelüli méret | Javaslat |
|---|---:|---:|---|
| `reports/scored_events.jsonl` | 2161 | 2.1 KiB | Kicsi demonstrációs kimenet, megtartható. |
| `reports/final/security_report.html` | 3708 | 3.6 KiB | Kicsi, de Markdown riportból származtatott állomány; opcionálisan kizárható. |
| `reports/lab/lab_scored_events.csv` | 6262 | 6.1 KiB | Kicsi demonstrációs táblázat, megtartható. |
| `reports/lab/lab_scored_events.jsonl` | 11220 | 11.0 KiB | Kicsi demonstrációs eseménysor, megtartható. |
| `reports/lab/risk_level_distribution.png` | 28854 | 28.2 KiB | Kicsi dolgozati ábra, megtartható. |
| `reports/final/thesis_figures/lab_risk_level_distribution.png` | 28854 | 28.2 KiB | Kicsi dolgozati ábra, megtartható. |
| `reports/final/thesis_figures/ae_context_confusion_matrix.png` | 42030 | 41.0 KiB | Kicsi dolgozati ábra, megtartható. |
| `reports/final/thesis_figures/ae_minimal_confusion_matrix.png` | 42031 | 41.0 KiB | Kicsi dolgozati ábra, megtartható. |
| `reports/final/thesis_figures/comparison_false_positive_rate.png` | 42599 | 41.6 KiB | Kicsi dolgozati ábra, megtartható. |
| `reports/final/thesis_figures/comparison_alert_count.png` | 44544 | 43.5 KiB | Kicsi dolgozati ábra, megtartható. |
| `reports/final/thesis_figures/comparison_precision_recall_f1.png` | 51134 | 49.9 KiB | Kicsi dolgozati ábra, megtartható. |
| `reports/lab/lab_timeline.png` | 58810 | 57.4 KiB | Kicsi dolgozati ábra, megtartható. |
| `reports/final/thesis_figures/lab_timeline.png` | 58810 | 57.4 KiB | Kicsi dolgozati ábra, megtartható. |

## Értékelés

Az ellenőrzött fájlok közül egyik sem nagy méretű. A legnagyobb állomány körülbelül 57.4 KiB, ezért a jelenlegi állapotban nem jelent repository méretkockázatot.

Verziókezelésben megtartandó fájlok:

- dolgozatba beemelhető kis PNG ábrák;
- `figure_manifest.csv`;
- kisebb Markdown, CSV és README jellegű dokumentációs állományok;
- kis elemszámú demonstrációs eseménykimenetek, ha a bemutató reprodukálhatóságát támogatják.

Verziókezelésből kizárandó fájlok nagyobb jövőbeli futások esetén:

- nagy méretű pontozott eseménylisták;
- nagy elemszámú JSONL vagy CSV scoring kimenetek;
- tömegesen előállított PNG ábrák;
- ismételten előállítható HTML riportok, ha a Markdown változat megmarad.

## Javasolt gyakorlat

A jelenlegi kis méretű május 8-i demonstrációs kimenetek maradhatnak a repositoryban, mert közvetlenül támogatják a diplomamunka bemutatását. Ha később nagyobb lab/replay vagy benchmark futás készül, akkor az ilyen futási eredményeket célszerű `.gitignore` szabályokkal kizárni, és a beadási melléklet részeként átadni.

Lehetséges későbbi kizárási minták:

```text
reports/lab/*.png
reports/lab/lab_scored_events*.csv
reports/lab/lab_scored_events*.jsonl
reports/final/security_report.html
```

Ezeket csak akkor érdemes aktiválni, ha a fájlok mérete vagy száma már zavarja a repository áttekinthetőségét. A dolgozatba szánt végleges ábrák és manifest fájlok külön dokumentált kivételként továbbra is megtarthatók.
