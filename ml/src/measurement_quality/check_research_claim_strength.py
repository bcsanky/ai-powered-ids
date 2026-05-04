from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.measurement_quality.common import (
    check_row,
    format_value,
    json_safe,
    metric_for,
    read_csv_optional,
    read_yaml,
    safe_float,
    write_json,
    write_markdown_report,
    write_rows_csv,
)


HYBRID_CONFIGS = ["Hybrid OR", "Hybrid weighted", "Hybrid priority"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", default="results/real_comparison/metrics_comparison.csv")
    parser.add_argument("--thresholds", default="docs/measurement_quality_thresholds.yaml")
    parser.add_argument("--output-dir", default="reports/measurement_quality")
    return parser.parse_args()


def best_hybrid_by_f1(comparison: pd.DataFrame | None) -> tuple[str | None, float | None]:
    if comparison is None or comparison.empty or "configuration" not in comparison.columns or "f1" not in comparison.columns:
        return None, None
    hybrids = comparison[comparison["configuration"].astype(str).isin(HYBRID_CONFIGS)].copy()
    if hybrids.empty:
        return None, None
    hybrids["f1_numeric"] = pd.to_numeric(hybrids["f1"], errors="coerce")
    hybrids = hybrids.dropna(subset=["f1_numeric"])
    if hybrids.empty:
        return None, None
    best = hybrids.sort_values("f1_numeric", ascending=False, kind="mergesort").iloc[0]
    return str(best["configuration"]), float(best["f1_numeric"])


def compute_claim(comparison: pd.DataFrame | None, thresholds: dict[str, Any]) -> dict[str, Any]:
    claim_thresholds = thresholds.get("claim_thresholds", {})
    min_f1_delta = float(claim_thresholds.get("min_f1_delta_for_improvement", 0.01))
    max_fpr_increase = float(claim_thresholds.get("max_fpr_increase_without_warning", 0.10))
    max_alert_multiplier = float(claim_thresholds.get("max_alert_count_multiplier_without_warning", 2.0))

    best_name, best_f1 = best_hybrid_by_f1(comparison)
    wazuh_f1 = metric_for(comparison, "Wazuh-only", "f1")
    ae_f1 = metric_for(comparison, "AE-Minimal lab", "f1")
    wazuh_recall = metric_for(comparison, "Wazuh-only", "recall")
    best_recall = metric_for(comparison, best_name or "", "recall")
    wazuh_fpr = metric_for(comparison, "Wazuh-only", "false_positive_rate")
    best_fpr = metric_for(comparison, best_name or "", "false_positive_rate")
    wazuh_alert_count = metric_for(comparison, "Wazuh-only", "alert_count")
    best_alert_count = metric_for(comparison, best_name or "", "alert_count")

    f1_delta = None if wazuh_f1 is None or best_f1 is None else best_f1 - wazuh_f1
    recall_delta = None if wazuh_recall is None or best_recall is None else best_recall - wazuh_recall
    fpr_delta = None if wazuh_fpr is None or best_fpr is None else best_fpr - wazuh_fpr
    alert_delta = None if wazuh_alert_count is None or best_alert_count is None else best_alert_count - wazuh_alert_count
    alert_multiplier = None
    if wazuh_alert_count is not None and best_alert_count is not None:
        alert_multiplier = float("inf") if wazuh_alert_count == 0 and best_alert_count > 0 else safe_float(best_alert_count / wazuh_alert_count) if wazuh_alert_count else 1.0

    missing = [value is None for value in [best_name, wazuh_f1, best_f1, wazuh_recall, best_recall, wazuh_fpr, best_fpr]]
    if any(missing):
        category = "INSUFFICIENT_MEASUREMENT"
        interpretation = "a vizsgált lab mérés alapján a kutatási állítás nem minősíthető, mert hiányosak a metrikák"
    elif f1_delta is not None and f1_delta >= min_f1_delta and (
        fpr_delta is None or fpr_delta <= max_fpr_increase
    ) and (alert_multiplier is None or alert_multiplier <= max_alert_multiplier):
        category = "CLAIM_SUPPORTED_WITH_LIMITATIONS"
        interpretation = "a vizsgált lab mérés alapján F1 szerint óvatos, korlátokkal kezelt javulás figyelhető meg"
    elif recall_delta is not None and recall_delta > 0 and (
        (fpr_delta is not None and fpr_delta > max_fpr_increase)
        or (alert_multiplier is not None and alert_multiplier > max_alert_multiplier)
    ):
        category = "TRADEOFF_ONLY"
        interpretation = "a vizsgált lab mérés alapján inkább recall és riasztási terhelés közötti kompromisszum látszik"
    elif f1_delta is not None and f1_delta > 0 and f1_delta < min_f1_delta:
        category = "CLAIM_NOT_SUPPORTED"
        interpretation = "a vizsgált lab mérés alapján az F1-különbség nem éri el a javulási küszöböt"
    else:
        category = "CLAIM_NOT_SUPPORTED"
        interpretation = "a vizsgált lab mérés alapján F1 szerint nem igazolható egyértelmű hibrid javulás"

    return {
        "claim_category": category,
        "best_hybrid_by_f1": best_name,
        "wazuh_f1": json_safe(wazuh_f1),
        "ae_f1": json_safe(ae_f1),
        "best_hybrid_f1": json_safe(best_f1),
        "f1_delta_vs_wazuh": json_safe(f1_delta),
        "recall_delta_vs_wazuh": json_safe(recall_delta),
        "fpr_delta_vs_wazuh": json_safe(fpr_delta),
        "alert_count_delta_vs_wazuh": json_safe(alert_delta),
        "alert_count_multiplier_vs_wazuh": json_safe(alert_multiplier),
        "interpretation": interpretation,
    }


def run_check(comparison_path: Path, thresholds_path: Path, output_dir: Path) -> dict[str, Any]:
    comparison = read_csv_optional(comparison_path)
    thresholds = read_yaml(thresholds_path)
    claim = compute_claim(comparison, thresholds)
    rows = [
        check_row(
            "metrics_available",
            "Kutatási állítás",
            "PASS" if comparison is not None and not comparison.empty else "FAIL",
            "comparison metrikák elérhetők" if comparison is not None and not comparison.empty else "comparison metrikák hiányoznak",
            "Futtasd a real comparison pipeline-t tényleges mérés után.",
        ),
        check_row("best_hybrid_by_f1", "Kutatási állítás", "PASS" if claim["best_hybrid_by_f1"] else "WARN", "legjobb hibrid F1 alapján", value=claim["best_hybrid_by_f1"] or "nincs adat"),
        check_row("f1_delta_vs_wazuh", "Kutatási állítás", "PASS" if claim["f1_delta_vs_wazuh"] is not None else "WARN", "F1 delta Wazuh-onlyhoz képest", value=format_value(claim["f1_delta_vs_wazuh"])),
        check_row("claim_category", "Kutatási állítás", "PASS" if claim["claim_category"] != "INSUFFICIENT_MEASUREMENT" else "WARN", claim["interpretation"], value=claim["claim_category"]),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "research_claim_strength.md"
    csv_path = output_dir / "research_claim_strength.csv"
    json_path = output_dir / "research_claim_strength.json"
    write_markdown_report(md_path, "Kutatási állítás erősségének ellenőrzése", rows, "A minősítés kizárólag a meglévő real-lab comparison metrikákból következik.")
    write_rows_csv(csv_path, rows)
    write_json(json_path, claim)
    return {"claim": claim, "rows": rows, "markdown": md_path, "csv": csv_path, "json": json_path}


def main() -> None:
    args = parse_args()
    result = run_check(Path(args.comparison), Path(args.thresholds), Path(args.output_dir))
    print(f"[OK] Kimenet: {result['markdown']}")


if __name__ == "__main__":
    main()

