from __future__ import annotations

import argparse
from pathlib import Path

from ml.src.final_acceptance.common import ensure_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/final_acceptance")
    return parser.parse_args()


def generate_brief(output_dir: Path) -> Path:
    output_dir = ensure_output_dir(output_dir)
    body = """# Real-lab mérési execution brief

## Előfeltételek

- Izolált lab környezet rendelkezésre áll.
- Wazuh manager és agent működik.
- OpenSearch export hozzáférés vagy manuális Wazuh export elérhető.
- Az AE-Minimal modellfájl és preprocess állomány rendelkezésre áll.
- Lefutott a `make final-acceptance`.

## Tilos inputok

- `examples/lab/` alatti fájlok.
- `examples/scoring/` alatti fájlok.
- `templates/` alatti fájlok.
- Teszt fixture vagy sample nevű bemenetek.

## Kötelező valós inputok

- `data/lab/lab_ground_truth.csv`
- `data/lab/lab_features.csv`
- `data/wazuh/alerts.jsonl`

## Futtatási sorrend

1. `make lab-session-prep`
2. A mérés közben: `python -m ml.src.lab_session.scenario_marker_helper start ...`
3. A mérés közben: `python -m ml.src.lab_session.scenario_marker_helper end ...`
4. A mérés után: `python -m ml.src.lab_session.scenario_marker_helper export --output data/lab/lab_ground_truth.csv`
5. Wazuh alert export valós rendszerből.
6. Zeek conn.log vagy flow CSV előállítása valós lab forgalomból.
7. `make lab-session-after-capture`
8. `make final-real-measurement-package-with-provenance`
9. `make final-live-integration`
10. `make final-real-measurement-thesis-ready`
11. `make final-thesis-integration`
12. `make repo-hygiene-check`
13. `make final-validate`

## Ellenőrizendő eredményfájlok

- `reports/real_measurement/measurement_provenance.json`
- `results/real_comparison/metrics_comparison.md`
- `reports/real_measurement_qa/thesis_readiness.md`
- `reports/live_integration/thesis_live_integration_section.md`
- `reports/thesis_integration/chapter6_results_generated.md`

## Anonimizálás

Raw Wazuh exportot, hostneveket, IP-címeket és esetleges felhasználóneveket beadás előtt ellenőrizni kell. Érzékeny mezők esetén futtasd a redaction lépést, és a beadási mellékletbe az anonimizált változat kerüljön.

## Beemelhetőség

Az eredmények csak akkor kerülhetnek a dolgozatba real-lab eredményként, ha a provenance verified real_lab státuszú, a no-demo guard sikeres, és a post-run QA READY vagy READY_WITH_LIMITATIONS státuszt ad.
"""
    output = output_dir / "real_lab_execution_brief.md"
    output.write_text(body, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    output = generate_brief(Path(args.output_dir))
    print(f"[OK] Kimenet: {output}")


if __name__ == "__main__":
    main()

