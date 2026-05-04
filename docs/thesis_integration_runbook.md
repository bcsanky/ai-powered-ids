# Dolgozati fejezetintegrációs runbook

## Cél

Ez a runbook azt írja le, hogyan készülnek el a Wordbe másolható dolgozati fejezetrészek a tényleges, provenance-szel igazolt real-lab mérés után. A folyamat nem indít új mérést, nem állít elő Wazuh exportot, és nem számol új metrikát: kizárólag a már meglévő, ellenőrzött CSV és JSON kimenetekből készít szöveges segédanyagot.

## Előfeltételek

A `final-thesis-integration` cél csak akkor használható végleges dolgozati anyag előállítására, ha rendelkezésre állnak az alábbi fájlok:

- `reports/real_measurement/measurement_provenance.json`
- `results/real_comparison/metrics_comparison.csv`
- `results/wazuh_real/metrics_summary.csv`
- `results/ae_lab/metrics_summary.csv`
- `results/hybrid_real/metrics_summary.csv`
- `reports/real_measurement_qa/research_question_answer.json`, ha a post-run QA lefutott
- `reports/live_integration/enrichment_summary.csv`, ha az integrációs demonstráció is elkészült

Ha nincs verified real_lab provenance, a kimenetek csak szerkezeti vázlatként használhatók. Ilyenkor a szöveg nem tekinthető végleges real-lab eredménynek.

## Futtatás

A teljes folyamat:

```bash
make final-thesis-integration
```

Részenként is futtatható:

```bash
make thesis-check-inputs
make thesis-generate-chapter5
make thesis-generate-chapter6
make thesis-generate-summary-hu
make thesis-generate-summary-en
make thesis-generate-abstracts
make thesis-generate-figures-tables-plan
make thesis-generate-appendix-plan
make thesis-generate-update-package
make thesis-generate-defense-questions
```

## Kimenetek

A kimenetek a `reports/thesis_integration/` könyvtárba kerülnek. Ez futási kimeneti könyvtár, Gitbe nem kerül.

- `chapter5_implementation_generated.md`: 5. fejezetbe illeszthető implementációs rész.
- `chapter6_results_generated.md`: 6. fejezetbe illeszthető eredmény- és értékelési rész.
- `chapter6_tables.md`: Wordbe másolható eredménytáblák.
- `chapter6_limitations.md`: korlátok és értelmezési feltételek.
- `chapter6_research_question_answer.md`: kutatási kérdésre adott mérnöki válasz.
- `chapter7_osszegzes_generated.md`: magyar összegzés vázlata.
- `chapter8_summary_generated.md`: angol Summary vázlata.
- `abstract_hu_generated.md` és `abstract_en_generated.md`: absztraktok.
- `figures_plan.md` és `tables_plan.md`: beemelési terv.
- `appendix_plan.md`: mellékletterv.
- `thesis_update_package.md`: Word frissítési csomag.
- `defense_questions_generated.md`: védési kérdés-válasz segédlet.

## Wordbe illesztés

A Markdown szövegeket beillesztés után kézzel ellenőrizni kell:

- fejezetszámozás;
- ábra- és táblázatszámozás;
- kereszthivatkozások;
- IEEE hivatkozási sorrend;
- rövidítések jegyzéke;
- absztrakt és Summary terjedelme;
- minden metrika egyezése a mérési CSV fájlokkal.

## Értelmezési szabályok

- A fejezetrész nem állíthat javulást, ha azt a metrikák nem támasztják alá.
- A real-lab eredmények csak a vizsgált lab mérésben értelmezhetők.
- Az end-to-end live integration kimenet integrációs demonstráció, nem benchmark.
- A dolgozatban külön kell kezelni az offline CIC-IDS2017 eredményeket és a verified real-lab eredményeket.
- Provenance nélküli eredmény nem használható végleges dolgozati bizonyítékként.

