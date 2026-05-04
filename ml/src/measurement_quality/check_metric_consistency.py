from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.measurement_quality.common import (
    check_row,
    has_fail,
    read_csv_optional,
    to_numeric_series,
    write_outputs,
)


EXPECTED_CONFIGURATIONS = ["Wazuh-only", "AE-Minimal lab", "Hybrid OR", "Hybrid weighted", "Hybrid priority"]
RANGE_METRICS = ["precision", "recall", "f1", "false_positive_rate", "false_negative_rate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wazuh-metrics", default="results/wazuh_real/metrics_summary.csv")
    parser.add_argument("--ae-metrics", default="results/ae_lab/metrics_summary.csv")
    parser.add_argument("--hybrid-metrics", default="results/hybrid_real/metrics_summary.csv")
    parser.add_argument("--comparison", default="results/real_comparison/metrics_comparison.csv")
    parser.add_argument("--wazuh-predictions", default="results/wazuh_real/predictions.csv")
    parser.add_argument("--ae-predictions", default="results/ae_lab/predictions.csv")
    parser.add_argument("--hybrid-predictions", default="results/hybrid_real/predictions.csv")
    parser.add_argument("--output-dir", default="reports/measurement_quality")
    return parser.parse_args()


def csv_row_status(name: str, df: pd.DataFrame | None) -> dict[str, Any]:
    return check_row(
        f"{name}_readable",
        "Metrika CSV",
        "PASS" if df is not None and not df.empty else "FAIL",
        f"{name} olvasható" if df is not None and not df.empty else f"{name} hiányzik vagy üres",
        "Futtasd a real-lab metrika pipeline megfelelő lépését.",
        0 if df is None else len(df),
    )


def check_confusion_sum(name: str, df: pd.DataFrame | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if df is None or df.empty:
        return rows
    required = ["TP", "FP", "TN", "FN", "n_samples"]
    lower_required = ["tp", "fp", "tn", "fn", "n_samples"]
    columns = required if set(required).issubset(df.columns) else lower_required
    if not set(columns).issubset(df.columns):
        rows.append(check_row(f"{name}_confusion_sum", "Konfúziós mátrix", "WARN", "TP/FP/TN/FN vagy n_samples oszlop hiányzik"))
        return rows
    for index, row in df.iterrows():
        values = [pd.to_numeric(row[col], errors="coerce") for col in columns[:4]]
        n_samples_value = pd.to_numeric(row["n_samples"], errors="coerce")
        if any(pd.isna(value) for value in values) or pd.isna(n_samples_value):
            rows.append(
                check_row(
                    f"{name}_confusion_sum_{index}",
                    "Konfúziós mátrix",
                    "WARN",
                    "konfúziós mátrix vagy n_samples érték hiányos",
                    "A hiányos metrikát értelmezési korlátként kell kezelni.",
                )
            )
            continue
        total = sum(int(value) for value in values)
        n_samples = int(n_samples_value)
        rows.append(
            check_row(
                f"{name}_confusion_sum_{index}",
                "Konfúziós mátrix",
                "PASS" if total == n_samples else "FAIL",
                f"TP+FP+TN+FN={total}, n_samples={n_samples}",
                "" if total == n_samples else "A metrikaszámítást ellenőrizni kell.",
            )
        )
    return rows


def run_check(
    *,
    wazuh_metrics: Path,
    ae_metrics: Path,
    hybrid_metrics: Path,
    comparison_path: Path,
    wazuh_predictions: Path,
    ae_predictions: Path,
    hybrid_predictions: Path,
    output_dir: Path,
) -> dict[str, Any]:
    wazuh = read_csv_optional(wazuh_metrics)
    ae = read_csv_optional(ae_metrics)
    hybrid = read_csv_optional(hybrid_metrics)
    comparison = read_csv_optional(comparison_path)
    rows: list[dict[str, Any]] = [
        csv_row_status("wazuh_metrics", wazuh),
        csv_row_status("ae_metrics", ae),
        csv_row_status("hybrid_metrics", hybrid),
        csv_row_status("metrics_comparison", comparison),
    ]

    if comparison is not None and not comparison.empty:
        configs = set(comparison.get("configuration", pd.Series(dtype=str)).astype(str))
        missing = [config for config in EXPECTED_CONFIGURATIONS if config not in configs]
        rows.append(
            check_row(
                "comparison_configurations",
                "Comparison",
                "PASS" if not missing else "FAIL",
                "minden kötelező konfiguráció szerepel" if not missing else "hiányzó konfigurációk: " + ", ".join(missing),
            )
        )
        for metric in RANGE_METRICS:
            if metric not in comparison.columns:
                rows.append(check_row(f"range_{metric}", "Metrikatartomány", "FAIL", f"hiányzó metrika: {metric}"))
                continue
            values = pd.to_numeric(comparison[metric], errors="coerce")
            ok = values.notna().all() and ((values >= 0.0) & (values <= 1.0)).all()
            rows.append(check_row(f"range_{metric}", "Metrikatartomány", "PASS" if ok else "FAIL", f"{metric} 0 és 1 közötti"))
        if "alert_count" in comparison.columns:
            alerts = pd.to_numeric(comparison["alert_count"], errors="coerce")
            rows.append(
                check_row(
                    "alert_count_non_negative",
                    "Comparison",
                    "PASS" if alerts.notna().all() and (alerts >= 0).all() else "FAIL",
                    "alert_count nem negatív",
                )
            )
        else:
            rows.append(check_row("alert_count_non_negative", "Comparison", "FAIL", "alert_count oszlop hiányzik"))
        if "n_samples" in comparison.columns:
            values = pd.to_numeric(comparison["n_samples"], errors="coerce").dropna().astype(int).tolist()
            rows.append(
                check_row(
                    "n_samples_consistency",
                    "Comparison",
                    "PASS" if len(values) == len(comparison) and len(set(values)) == 1 else "FAIL",
                    f"n_samples értékek: {values}",
                )
            )
        else:
            rows.append(check_row("n_samples_consistency", "Comparison", "WARN", "n_samples oszlop hiányzik"))
        nan_metrics = int(comparison[RANGE_METRICS].isna().sum().sum()) if set(RANGE_METRICS).issubset(comparison.columns) else 0
        rows.append(
            check_row(
                "nan_metric_values",
                "Comparison",
                "WARN" if nan_metrics else "PASS",
                "üres vagy NaN metrika található" if nan_metrics else "nincs NaN a fő metrikákban",
                "A hiányos metrikát értelmezési korlátként kell kezelni.",
                nan_metrics,
            )
        )

    rows.extend(check_confusion_sum("wazuh", wazuh))
    rows.extend(check_confusion_sum("ae", ae))
    rows.extend(check_confusion_sum("hybrid", hybrid))

    wazuh_pred = read_csv_optional(wazuh_predictions)
    ae_pred = read_csv_optional(ae_predictions)
    hybrid_pred = read_csv_optional(hybrid_predictions)
    if hybrid_pred is not None and not hybrid_pred.empty and {"hybrid_or_pred", "wazuh_pred", "ae_pred"}.issubset(hybrid_pred.columns):
        expected_or = (to_numeric_series(hybrid_pred, "wazuh_pred").fillna(0).astype(int) | to_numeric_series(hybrid_pred, "ae_pred").fillna(0).astype(int))
        actual_or = to_numeric_series(hybrid_pred, "hybrid_or_pred").fillna(0).astype(int)
        rows.append(
            check_row(
                "hybrid_or_logic",
                "Hibrid logika",
                "PASS" if (expected_or == actual_or).all() else "FAIL",
                "Hybrid OR predikció megfelel a Wazuh OR AE logikának"
                if (expected_or == actual_or).all()
                else "Hybrid OR predikció eltér a Wazuh OR AE logikától",
            )
        )
    else:
        rows.append(check_row("hybrid_or_logic", "Hibrid logika", "WARN", "Hybrid OR predikciós oszlopok nem ellenőrizhetők"))

    if wazuh_pred is not None and not wazuh_pred.empty and "wazuh_pred" in wazuh_pred.columns:
        has_wazuh_match = bool((to_numeric_series(wazuh_pred, "wazuh_pred").fillna(0).astype(int) == 1).any())
        for df_name, df in [("wazuh", wazuh), ("hybrid", hybrid)]:
            if has_wazuh_match and df is not None and not df.empty:
                for col in ["mean_ttd", "median_ttd"]:
                    values = to_numeric_series(df, col)
                    rows.append(
                        check_row(
                            f"{df_name}_{col}_available",
                            "TTD metrika",
                            "PASS" if values.notna().any() else "WARN",
                            f"{df_name} {col} {'elérhető' if values.notna().any() else 'nincs adat'}",
                        )
                    )
    _ = ae_pred  # kept for schema symmetry in tests and future checks
    return write_outputs(
        output_dir=output_dir,
        basename="metric_consistency",
        title="Metrikai konzisztencia ellenőrzés",
        rows=rows,
        intro="A riport a Wazuh-only, AE-Minimal lab és hibrid metrikák önkonzisztenciáját vizsgálja.",
    )


def main() -> None:
    args = parse_args()
    result = run_check(
        wazuh_metrics=Path(args.wazuh_metrics),
        ae_metrics=Path(args.ae_metrics),
        hybrid_metrics=Path(args.hybrid_metrics),
        comparison_path=Path(args.comparison),
        wazuh_predictions=Path(args.wazuh_predictions),
        ae_predictions=Path(args.ae_predictions),
        hybrid_predictions=Path(args.hybrid_predictions),
        output_dir=Path(args.output_dir),
    )
    print(f"[OK] Kimenet: {result['markdown']}")
    if has_fail(result["rows"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
