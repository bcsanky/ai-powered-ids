from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.measurement_quality.common import (
    check_row,
    has_fail,
    is_forbidden_real_path,
    provenance_row,
    read_yaml,
    validate_verified_provenance,
    write_outputs,
)


REQUIRED_COLUMNS = ["event_id", "timestamp_start", "timestamp_end", "scenario", "label"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", default="data/lab/lab_ground_truth.csv")
    parser.add_argument("--thresholds", default="docs/measurement_quality_thresholds.yaml")
    parser.add_argument("--provenance", default="reports/real_measurement/measurement_provenance.json")
    parser.add_argument("--output-dir", default="reports/measurement_quality")
    return parser.parse_args()


def count_check(check_id: str, category: str, actual: int, minimum: int, message: str) -> dict[str, Any]:
    status = "PASS" if actual >= minimum else ("FAIL" if actual == 0 else "WARN")
    return check_row(
        check_id,
        category,
        status,
        f"{message}: {actual}, elvárt minimum: {minimum}",
        "" if status == "PASS" else "A mérési lefedettséget további valós lab eseményekkel kell erősíteni.",
        actual,
    )


def run_check(
    *,
    ground_truth: Path,
    thresholds_path: Path,
    provenance_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    thresholds = read_yaml(thresholds_path)
    minimums = thresholds.get("minimums", {})
    expected = thresholds.get("expected_scenarios", {})
    rows: list[dict[str, Any]] = [provenance_row(provenance_path)]
    provenance_valid, _, provenance = validate_verified_provenance(provenance_path)

    if is_forbidden_real_path(ground_truth):
        rows.append(
            check_row(
                "ground_truth_path_guard",
                "Adateredet",
                "FAIL",
                "ground truth útvonal tiltott demo/sablon/teszt eredetű",
                "Valós lab futásból származó data/lab/lab_ground_truth.csv szükséges.",
                ground_truth.as_posix(),
            )
        )
    else:
        rows.append(check_row("ground_truth_path_guard", "Adateredet", "PASS", "ground truth útvonal elfogadható", value=ground_truth.as_posix()))

    if provenance:
        expected_gt = str(provenance.get("ground_truth_path", ""))
        rows.append(
            check_row(
                "ground_truth_provenance_path",
                "Adateredet",
                "PASS" if expected_gt == ground_truth.as_posix() else "WARN",
                "ground truth útvonal egyezik a provenance-szel"
                if expected_gt == ground_truth.as_posix()
                else f"ground truth útvonal eltér a provenance-től: {expected_gt}",
                "Ellenőrizd, hogy a quality gate ugyanazt az inputot olvassa-e, mint a mérési csomag.",
                ground_truth.as_posix(),
            )
        )

    if not ground_truth.exists():
        rows.append(check_row("ground_truth_exists", "Bemenet", "FAIL", f"hiányzó ground truth: {ground_truth}", "Futtasd a lab session exportot tényleges mérés után."))
        return write_outputs(
            output_dir=output_dir,
            basename="scenario_coverage",
            title="Scenario coverage ellenőrzés",
            rows=rows,
            intro="A scenario coverage csak verified real_lab provenance és tényleges ground truth alapján értelmezhető.",
        )

    df = pd.read_csv(ground_truth)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    rows.append(
        check_row(
            "ground_truth_columns",
            "Ground truth",
            "PASS" if not missing_columns else "FAIL",
            "kötelező oszlopok rendben" if not missing_columns else "hiányzó oszlopok: " + ", ".join(missing_columns),
        )
    )
    if missing_columns:
        return write_outputs(output_dir=output_dir, basename="scenario_coverage", title="Scenario coverage ellenőrzés", rows=rows)

    df = df.copy()
    df["label_norm"] = df["label"].astype(str).str.strip().str.lower()
    df["scenario_norm"] = df["scenario"].astype(str).str.strip()
    start = pd.to_datetime(df["timestamp_start"], utc=True, errors="coerce")
    end = pd.to_datetime(df["timestamp_end"], utc=True, errors="coerce")
    duration = (end - start).dt.total_seconds()
    attack_scenarios = set(df.loc[df["label_norm"] == "attack", "scenario_norm"])
    scenario_count = int(df["scenario_norm"].nunique())

    rows.extend(
        [
            check_row(
                "event_id_unique",
                "Ground truth",
                "PASS" if df["event_id"].astype(str).is_unique else "FAIL",
                "event_id értékek egyediek" if df["event_id"].astype(str).is_unique else "ismétlődő event_id található",
            ),
            check_row(
                "timestamps_valid",
                "Ground truth",
                "PASS" if start.notna().all() and end.notna().all() and (start < end).all() else "FAIL",
                "timestamp_start és timestamp_end értékek validak"
                if start.notna().all() and end.notna().all() and (start < end).all()
                else "hibás vagy nem növekvő időablak található",
            ),
            count_check("min_total_events", "Mérési elemszám", len(df), int(minimums.get("min_total_events", 20)), "összes esemény"),
            count_check(
                "min_benign_events",
                "Mérési elemszám",
                int((df["label_norm"] == "benign").sum()),
                int(minimums.get("min_benign_events", 5)),
                "benign események száma",
            ),
            count_check(
                "min_attack_events",
                "Mérési elemszám",
                int((df["label_norm"] == "attack").sum()),
                int(minimums.get("min_attack_events", 5)),
                "attack események száma",
            ),
            count_check("min_scenarios", "Scenario lefedettség", scenario_count, int(minimums.get("min_scenarios", 4)), "különböző scenario-k száma"),
            count_check(
                "min_attack_scenarios",
                "Scenario lefedettség",
                len(attack_scenarios),
                int(minimums.get("min_attack_scenarios", 2)),
                "attack scenario-k száma",
            ),
            check_row(
                "short_or_zero_duration",
                "Ground truth",
                "WARN" if (duration <= 0).any() else "PASS",
                "túl rövid vagy nulla időtartamú esemény található" if (duration <= 0).any() else "nincs nulla vagy negatív duration",
                "Ellenőrizd az event marker start/end használatát.",
                int((duration <= 0).sum()),
            ),
        ]
    )

    for label, expected_scenarios in expected.items():
        present = set(df.loc[df["label_norm"] == label, "scenario_norm"])
        missing = [scenario for scenario in expected_scenarios if scenario not in present]
        rows.append(
            check_row(
                f"expected_{label}_scenarios",
                "Scenario lefedettség",
                "WARN" if missing else "PASS",
                "hiányzó várt scenario-k: " + ", ".join(missing) if missing else f"várt {label} scenario-k lefedve",
                "A hiányzó scenario-kat csak akkor kell pótolni, ha a mérési terv részei.",
                len(expected_scenarios) - len(missing),
            )
        )

    return write_outputs(
        output_dir=output_dir,
        basename="scenario_coverage",
        title="Scenario coverage ellenőrzés",
        rows=rows,
        intro="A riport a lab ground truth eseményszámát, címkeeloszlását és scenario-lefedettségét vizsgálja.",
        extra_payload={"provenance_valid": provenance_valid, "scenario_count": scenario_count},
    )


def main() -> None:
    args = parse_args()
    result = run_check(
        ground_truth=Path(args.ground_truth),
        thresholds_path=Path(args.thresholds),
        provenance_path=Path(args.provenance),
        output_dir=Path(args.output_dir),
    )
    print(f"[OK] Kimenet: {result['markdown']}")
    if has_fail(result["rows"]):
        sys.exit(1)


if __name__ == "__main__":
    main()

