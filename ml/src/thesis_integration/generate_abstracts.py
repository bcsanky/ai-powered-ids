from __future__ import annotations

import argparse
from pathlib import Path

from ml.src.thesis_integration.common import (
    DEFAULT_PROVENANCE,
    cautious_improvement_sentence,
    compute_research_answer,
    ensure_output_dir,
    provenance_status,
    read_csv_optional,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", default="results/real_comparison/metrics_comparison.csv")
    parser.add_argument("--provenance", default=str(DEFAULT_PROVENANCE))
    parser.add_argument("--output-dir", default="reports/thesis_integration")
    return parser.parse_args()


def generate_abstracts(comparison_path: Path, provenance_path: Path, output_dir: Path) -> dict[str, Path]:
    output_dir = ensure_output_dir(output_dir)
    comparison = read_csv_optional(comparison_path)
    answer = compute_research_answer(comparison)
    improvement_hu = cautious_improvement_sentence(answer)
    verified = provenance_status(provenance_path) == "verified_real_lab"
    provenance_hu = "A real-lab mérési bemenetek provenance fájllal igazoltak." if verified else "A real-lab provenance még nem véglegesített, ezért a szöveg csak vázlatként használható."
    provenance_en = "The real-lab measurement inputs are verified by provenance metadata." if verified else "The real-lab provenance is not finalized yet, so this text can only be used as a draft aid."
    hu = f"""# Absztrakt

A dolgozat egy AI-alapú kiberfenyegetés-felderítő és -elemző laboratóriumi prototípust mutat be, amely a Wazuh/SIEM jellegű szabályalapú riasztásokat autoencoder-alapú anomáliadetektálással kapcsolja össze. A munka célja egy reprodukálható feldolgozási lánc kialakítása volt, amelyben ugyanazon címkézett lab eseményeken értékelhető a Wazuh-only baseline, az AE-Minimal pontozás és több hibrid döntési stratégia.

A prototípus Python-alapú komponensekből, Wazuh alert feldolgozásból, lab feature validálásból, autoencoder scoringból, hibrid kockázati döntésből, provenance ellenőrzésből és dolgozati riportkészítésből áll. {provenance_hu}

Az értékelés fő metrikái a precision, recall, F1, hamis pozitív arány, hamis negatív arány és riasztásszám. {improvement_hu} Az eredmény a vizsgált lab mérésre vonatkozik, és nem tekinthető éles SOC-rendszerre általánosítható teljesítménygaranciának.

A dolgozat mérnöki hozzájárulása a szabályalapú és gépi tanulási detektálás összekapcsolása, valamint az a guard és provenance réteg, amely megakadályozza a demonstrációs bemenetek kutatási eredményként történő kezelését.
"""
    delta = answer.get("f1_delta_vs_wazuh")
    if delta is None:
        improvement_en = "The F1 change compared to the Wazuh-only baseline cannot be determined from the available metrics."
    elif delta > 0:
        improvement_en = "In the controlled real-lab measurement, the best hybrid strategy showed an F1 improvement over the Wazuh-only baseline."
    else:
        improvement_en = "In the controlled real-lab measurement, the metrics do not show clear F1 improvement over the Wazuh-only baseline."
    en = f"""# Abstract

This thesis presents a laboratory prototype for AI-based cyber threat detection and analysis. The prototype combines Wazuh-style rule-based alert processing with autoencoder-based anomaly detection and evaluates Wazuh-only, AE-Minimal, and hybrid strategies on the same labeled lab events.

The implemented pipeline includes Wazuh alert parsing, lab ground truth and feature validation, AE-Minimal scoring, hybrid risk decisions, provenance checks, and thesis-oriented reporting. {provenance_en}

The evaluation uses precision, recall, F1, false positive rate, false negative rate, and alert count as the main metrics. {improvement_en} The findings are limited to the measured laboratory setting and do not imply a production-ready SOC system or a general performance guarantee.

The main engineering contribution is a traceable hybrid detection pipeline that separates verified real-lab results from development examples and offline outputs.
"""
    outputs = {
        "hu": output_dir / "abstract_hu_generated.md",
        "en": output_dir / "abstract_en_generated.md",
    }
    outputs["hu"].write_text(hu, encoding="utf-8")
    outputs["en"].write_text(en, encoding="utf-8")
    return outputs


def main() -> None:
    args = parse_args()
    outputs = generate_abstracts(Path(args.comparison), Path(args.provenance), Path(args.output_dir))
    for path in outputs.values():
        print(f"[OK] Kimenet: {path}")


if __name__ == "__main__":
    main()

