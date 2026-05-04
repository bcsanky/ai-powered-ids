from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ml.src.measurement_quality.common import check_row, format_value, has_fail, read_csv_optional, to_numeric_series, write_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wazuh-predictions", default="results/wazuh_real/predictions.csv")
    parser.add_argument("--hybrid-predictions", default="results/hybrid_real/predictions.csv")
    parser.add_argument("--ground-truth", default="data/lab/lab_ground_truth.csv")
    parser.add_argument("--large-ttd-threshold-sec", type=float, default=300.0)
    parser.add_argument("--output-dir", default="reports/measurement_quality")
    return parser.parse_args()


def run_check(
    *,
    wazuh_predictions: Path,
    hybrid_predictions: Path,
    ground_truth: Path,
    large_ttd_threshold_sec: float,
    output_dir: Path,
) -> dict[str, Any]:
    wazuh = read_csv_optional(wazuh_predictions)
    hybrid = read_csv_optional(hybrid_predictions)
    gt = read_csv_optional(ground_truth)
    rows: list[dict[str, Any]] = []
    for name, df, path in [("wazuh_predictions", wazuh, wazuh_predictions), ("hybrid_predictions", hybrid, hybrid_predictions), ("ground_truth", gt, ground_truth)]:
        rows.append(
            check_row(
                f"{name}_readable",
                "Bemenet",
                "PASS" if df is not None and not df.empty else "FAIL",
                f"{name} olvasható" if df is not None and not df.empty else f"{name} hiányzik vagy üres",
                "Futtasd a real-lab pipeline megfelelő lépését.",
                path.as_posix(),
            )
        )

    if wazuh is not None and not wazuh.empty and {"time_to_detection_sec", "wazuh_pred", "y_true"}.issubset(wazuh.columns):
        ttd_all = to_numeric_series(wazuh, "time_to_detection_sec")
        detected = wazuh[(to_numeric_series(wazuh, "wazuh_pred").fillna(0).astype(int) == 1)]
        ttd_detected = to_numeric_series(detected, "time_to_detection_sec").dropna()
        negative_count = int((ttd_detected < 0).sum())
        large_count = int((ttd_detected > large_ttd_threshold_sec).sum())
        attack_count = int((to_numeric_series(wazuh, "y_true").fillna(0).astype(int) == 1).sum())
        attack_with_ttd = int(((to_numeric_series(wazuh, "y_true").fillna(0).astype(int) == 1) & ttd_all.notna()).sum())
        coverage = float(attack_with_ttd / attack_count) if attack_count else 0.0
        rows.extend(
            [
                check_row(
                    "ttd_numeric",
                    "TTD",
                    "PASS" if ttd_detected.notna().all() else "WARN",
                    "TTD értékek numerikusak, ahol alert volt" if ttd_detected.notna().all() else "nem minden TTD érték numerikus",
                ),
                check_row(
                    "ttd_negative",
                    "TTD",
                    "FAIL" if negative_count else "PASS",
                    "negatív TTD érték található" if negative_count else "nincs negatív TTD",
                    "Ellenőrizd az időszinkront és a korrelációs időablakot.",
                    negative_count,
                ),
                check_row(
                    "ttd_large_values",
                    "TTD",
                    "WARN" if large_count else "PASS",
                    f"{large_count} darab {large_ttd_threshold_sec:.0f} másodpercnél nagyobb TTD",
                    "Túl nagy TTD esetén az időablak és az alert matching ellenőrizendő.",
                    large_count,
                ),
                check_row(
                    "ttd_attack_coverage",
                    "TTD",
                    "PASS" if coverage > 0 else "WARN",
                    f"attack TTD coverage: {coverage:.4f}",
                    "Ha nincs TTD adat, a TTD nem használható eredményértelmezésre.",
                    coverage,
                ),
            ]
        )
        if ttd_detected.empty:
            stats = {"mean": "nincs adat", "median": "nincs adat", "p95": "nincs adat"}
        else:
            stats = {
                "mean": format_value(float(ttd_detected.mean())),
                "median": format_value(float(ttd_detected.median())),
                "p95": format_value(float(ttd_detected.quantile(0.95))),
            }
        rows.append(check_row("ttd_summary_stats", "TTD", "PASS" if ttd_detected.notna().any() else "WARN", "TTD összefoglaló statisztikák", value=str(stats)))
    else:
        rows.append(check_row("ttd_columns", "TTD", "WARN", "TTD oszlopok nem ellenőrizhetők", "Nincs TTD adat."))

    if hybrid is not None and not hybrid.empty and "time_to_detection_sec" in hybrid.columns:
        rows.append(
            check_row(
                "hybrid_ttd_available",
                "TTD",
                "PASS" if to_numeric_series(hybrid, "time_to_detection_sec").notna().any() else "WARN",
                "hibrid predikciókban van TTD adat" if to_numeric_series(hybrid, "time_to_detection_sec").notna().any() else "hibrid predikciókban nincs TTD adat",
            )
        )

    return write_outputs(
        output_dir=output_dir,
        basename="ttd_quality",
        title="Time-to-detection minőségi ellenőrzés",
        rows=rows,
        intro="A riport csak meglévő TTD mezőkből dolgozik. Hiányzó TTD esetén nem számol ki helyettesítő értéket.",
    )


def main() -> None:
    args = parse_args()
    result = run_check(
        wazuh_predictions=Path(args.wazuh_predictions),
        hybrid_predictions=Path(args.hybrid_predictions),
        ground_truth=Path(args.ground_truth),
        large_ttd_threshold_sec=args.large_ttd_threshold_sec,
        output_dir=Path(args.output_dir),
    )
    print(f"[OK] Kimenet: {result['markdown']}")
    if has_fail(result["rows"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
