from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ml.src.measurement_quality.common import ensure_output_dir, read_json_optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/measurement_quality")
    return parser.parse_args()


def sentence_for_claim(claim_category: str) -> str:
    if claim_category == "CLAIM_SUPPORTED_WITH_LIMITATIONS":
        return "A vizsgált lab mérés alapján a hibrid megközelítésnél F1 szerint korlátozottan értelmezhető javulás figyelhető meg."
    if claim_category == "TRADEOFF_ONLY":
        return "A vizsgált lab mérés alapján elsősorban kompromisszum látszik: a recall javulása magasabb FPR-rel vagy nagyobb riasztási terheléssel járhat."
    if claim_category == "CLAIM_NOT_SUPPORTED":
        return "A vizsgált lab mérés alapján nem igazolható egyértelmű hibrid javulás a Wazuh-only viszonyítási alaphoz képest."
    return "A mérés minősége vagy a metrikák teljessége alapján nem tehető megalapozott kutatási állítás."


def run_notes(output_dir: Path) -> dict[str, Any]:
    ensure_output_dir(output_dir)
    summary = read_json_optional(output_dir / "measurement_quality_summary.json") or {}
    claim = read_json_optional(output_dir / "research_claim_strength.json") or {}
    status = summary.get("measurement_quality_status", "MEASUREMENT_NOT_READY")
    claim_category = claim.get("claim_category", "INSUFFICIENT_MEASUREMENT")
    lines = [
        "# Dolgozati mérési minőségi megjegyzések",
        "",
        f"A mérési minőségi státusz: **{status}**.",
        "",
        "A mérési lefedettséget a ground truth eseményszám, a benign és attack események aránya, valamint a különböző scenario-k száma alapján kell értelmezni. Ha az elemszám vagy a scenario-lefedettség alacsony, az eredmény csak korlátozott lab megfigyelésként használható.",
        "",
        "A Wazuh/AE/hibrid összehasonlítás megbízhatósága azon múlik, hogy minden komponens ugyanarra az event_id készletre futott-e, és hogy az AE scoring minden szükséges eseményhez tényleges anomáliapontszámot adott-e.",
        "",
        sentence_for_claim(str(claim_category)),
        "",
        "A dolgozatban ezért külön kell jelezni, hogy a mérés laboratóriumi környezetben készült, nem hosszú idejű éles SOC-validáció, és a TTD csak ott értelmezhető, ahol Wazuh alert matching ténylegesen rendelkezésre áll.",
        "",
        "Nem állítható általános ipari érvényesség vagy production teljesítménygarancia. A minőségi riport célja a mérési eredmények óvatos, metrikaalapú értelmezése.",
    ]
    output_path = output_dir / "thesis_measurement_quality_notes.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"markdown": output_path, "status": status, "claim_category": claim_category}


def main() -> None:
    args = parse_args()
    result = run_notes(Path(args.output_dir))
    print(f"[OK] Kimenet: {result['markdown']}")


if __name__ == "__main__":
    main()
