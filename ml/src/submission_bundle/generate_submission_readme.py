from __future__ import annotations

import argparse
from pathlib import Path

from ml.src.submission_bundle.common import ensure_dir


README_TEXT = """# AI-powered IDS beadási melléklet

Ez a csomag az MSc diplomamunka mérnöki prototípusának forráskódját, konfigurációit, dokumentációját, sablonjait és engedélyezett demonstrációs példáit tartalmazza.

## Csomag felépítése

- `ml/src/`: a feldolgozási, értékelési, guard, QA és csomagoló modulok.
- `ml/tests/`: unit tesztek és ellenőrző fixture-logika.
- `experiments/final/`: végleges konfigurációk.
- `templates/lab/`: kitölthető lab mérési sablonok.
- `docs/`: runbookok, követelmény- és provenance-dokumentáció.
- `examples/`: kizárólag demonstrációs bemenetek.

## Validáció

A csomag ellenőrzéséhez a következő parancsok használhatók:

```bash
make final-validate
make repo-hygiene-check
make final-acceptance
make final-submission-check
```

Valódi mérés után a mérési és dolgozati kimenetek előállítása:

```bash
make final-real-measurement-package-with-provenance
make final-live-integration
make final-measurement-quality
make final-thesis-integration
```

## Mit nem tartalmaz

A csomag nem tartalmaz raw Wazuh alert exportot, PCAP/PCAPNG állományt, érzékeny kulcsot, tanított modellbinárist vagy titkos konfigurációt. Ezeket külön, anonimizálva és intézményi adatkezelési szabályok szerint kell kezelni.

Fontos: a demo examples nem mérési eredmények. Az `examples/lab` és `examples/scoring` fájlok csak fejlesztési és demonstrációs célra szerepelnek, nem real-lab benchmark bizonyítékok.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/submission_bundle")
    parser.add_argument("--dist-dir", default="dist/submission")
    return parser.parse_args()


def generate_readme(output_dir: Path, dist_dir: Path) -> None:
    ensure_dir(output_dir)
    ensure_dir(dist_dir)
    (output_dir / "SUBMISSION_README.md").write_text(README_TEXT, encoding="utf-8")
    (dist_dir / "SUBMISSION_README.md").write_text(README_TEXT, encoding="utf-8")


def main() -> None:
    args = parse_args()
    generate_readme(Path(args.output_dir), Path(args.dist_dir))
    print(f"[OK] SUBMISSION_README: {args.dist_dir}/SUBMISSION_README.md")


if __name__ == "__main__":
    main()

