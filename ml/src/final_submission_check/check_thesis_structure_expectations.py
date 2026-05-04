from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ml.src.final_submission_check.common import check_row, has_fail, read_text_optional, write_outputs
from ml.src.repo_hygiene.common import load_provenance, validate_provenance_payload


TOPICS = {
    "architektúra": ["architektúra", "prototípus"],
    "technológiák": ["Python", "FastAPI", "Wazuh", "Docker"],
    "adathalmaz": ["CIC-IDS2017", "adathalmaz", "ground truth"],
    "feature engineering": ["feature", "jellemző"],
    "autoencoder": ["autoencoder", "rekonstrukciós"],
    "Wazuh baseline": ["Wazuh-only", "Wazuh baseline"],
    "hibrid döntés": ["hibrid", "Hybrid"],
    "live integration": ["live integration", "dashboard-ready"],
    "mérési módszertan": ["módszertan", "mérés", "metrika"],
    "eredmények": ["eredmény", "precision", "recall", "F1"],
    "korlátok": ["korlát"],
    "etikai/adatvédelmi megfontolások": ["adatvédelmi", "etikai", "érzékeny"],
    "jövőbeli munka": ["továbbfejleszt", "jövőbeli"],
    "összegzés": ["összegzés"],
    "summary": ["Summary", "summary"],
}
DOCUMENTS = [
    "docs/thesis_chapter_5_implementation_final.md",
    "docs/thesis_chapter_6_results_final.md",
    "reports/thesis_integration/chapter5_implementation_generated.md",
    "reports/thesis_integration/chapter6_results_generated.md",
    "reports/thesis_integration/chapter7_osszegzes_generated.md",
    "reports/thesis_integration/chapter8_summary_generated.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/final_submission_check")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def combined_text(root: Path) -> tuple[str, list[str]]:
    texts = []
    existing = []
    for rel_path in DOCUMENTS:
        text = read_text_optional(root / rel_path)
        if text:
            texts.append(text)
            existing.append(rel_path)
    return "\n".join(texts), existing


def run_check(output_dir: Path, root: Path = Path(".")) -> dict:
    text, existing = combined_text(root)
    provenance_valid, _ = validate_provenance_payload(load_provenance(root / "reports/real_measurement/measurement_provenance.json"))
    rows = []
    for topic, keywords in TOPICS.items():
        found = any(keyword.lower() in text.lower() for keyword in keywords)
        if found:
            status = "PASS"
            message = f"téma lefedve: {topic}"
            recommendation = ""
        elif topic in {"eredmények", "summary", "összegzés"} and not provenance_valid:
            status = "WARN"
            message = f"{topic}: valós mérés vagy thesis integration után véglegesíthető"
            recommendation = "Verified provenance és thesis integration futtatás után ellenőrizd újra."
        else:
            status = "WARN"
            message = f"hiányzó vagy gyenge lefedés: {topic}"
            recommendation = "Pótold vagy ellenőrizd a kapcsolódó fejezetrészt a Word dokumentumban."
        rows.append(check_row(topic.replace("/", "_").replace(" ", "_"), "Fejezetszerkezet", status, message, recommendation, "; ".join(existing)))
    if not existing:
        rows.append(check_row("source_documents", "Fejezetszerkezet", "FAIL", "nem található ellenőrizhető fejezetszöveg"))
    return write_outputs(
        output_dir=output_dir,
        basename="thesis_structure_check",
        title="Dolgozati fejezetszerkezet ellenőrzés",
        rows=rows,
    )


def main() -> None:
    args = parse_args()
    result = run_check(Path(args.output_dir), Path(args.root))
    print(f"[OK] Kimenet: {result['markdown']}")
    if has_fail(result["rows"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
