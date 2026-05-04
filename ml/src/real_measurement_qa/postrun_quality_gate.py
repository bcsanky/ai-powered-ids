from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.repo_hygiene.common import load_provenance, validate_provenance_payload


EXPECTED_CONFIGURATIONS = [
    "Wazuh-only",
    "AE-Minimal lab",
    "Hybrid OR",
    "Hybrid weighted",
    "Hybrid priority",
]
REQUIRED_FILES = [
    "results/wazuh_real/metrics_summary.csv",
    "results/ae_lab/metrics_summary.csv",
    "results/hybrid_real/metrics_summary.csv",
    "results/real_comparison/metrics_comparison.csv",
    "results/real_comparison/metrics_comparison.md",
    "reports/real_measurement/real_lab_results_report.md",
    "reports/real_measurement/thesis_real_lab_section.md",
    "reports/real_measurement/measurement_manifest.csv",
]
RANGE_METRICS = ["precision", "recall", "f1", "false_positive_rate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/real_measurement_qa")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def qa_row(check_id: str, category: str, status: str, message: str, recommendation: str = "") -> dict[str, str]:
    return {
        "check_id": check_id,
        "category": category,
        "status": status,
        "message": message,
        "recommendation": recommendation,
    }


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    return df


def numeric(value: Any) -> float | None:
    converted = pd.to_numeric(value, errors="coerce")
    if pd.isna(converted):
        return None
    return float(converted)


def metric_for(df: pd.DataFrame | None, configuration: str, metric: str) -> float | None:
    if df is None or df.empty or "configuration" not in df.columns or metric not in df.columns:
        return None
    matched = df[df["configuration"].astype(str) == configuration]
    if matched.empty:
        return None
    return numeric(matched.iloc[0][metric])


def best_hybrid_by_f1(comparison: pd.DataFrame | None) -> dict[str, Any]:
    if comparison is None or comparison.empty or "configuration" not in comparison.columns or "f1" not in comparison.columns:
        return {"configuration": None, "f1": None}
    hybrids = comparison[comparison["configuration"].astype(str).str.startswith("Hybrid")].copy()
    if hybrids.empty:
        return {"configuration": None, "f1": None}
    hybrids["f1_numeric"] = pd.to_numeric(hybrids["f1"], errors="coerce")
    hybrids = hybrids.dropna(subset=["f1_numeric"])
    if hybrids.empty:
        return {"configuration": None, "f1": None}
    best = hybrids.sort_values("f1_numeric", ascending=False, kind="mergesort").iloc[0]
    return {"configuration": str(best["configuration"]), "f1": float(best["f1_numeric"])}


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def compute_research_answer(comparison: pd.DataFrame | None) -> dict[str, Any]:
    best = best_hybrid_by_f1(comparison)
    best_name = best["configuration"]
    wazuh_f1 = metric_for(comparison, "Wazuh-only", "f1")
    ae_f1 = metric_for(comparison, "AE-Minimal lab", "f1")
    best_hybrid_f1 = best["f1"]
    wazuh_recall = metric_for(comparison, "Wazuh-only", "recall")
    wazuh_fpr = metric_for(comparison, "Wazuh-only", "false_positive_rate")
    wazuh_alert_count = metric_for(comparison, "Wazuh-only", "alert_count")
    hybrid_recall = metric_for(comparison, best_name, "recall") if best_name else None
    hybrid_fpr = metric_for(comparison, best_name, "false_positive_rate") if best_name else None
    hybrid_alert_count = metric_for(comparison, best_name, "alert_count") if best_name else None
    answer = {
        "best_hybrid_by_f1": best_name,
        "wazuh_f1": wazuh_f1,
        "ae_f1": ae_f1,
        "best_hybrid_f1": best_hybrid_f1,
        "f1_delta_vs_wazuh": None if wazuh_f1 is None or best_hybrid_f1 is None else best_hybrid_f1 - wazuh_f1,
        "recall_delta_vs_wazuh": None if wazuh_recall is None or hybrid_recall is None else hybrid_recall - wazuh_recall,
        "fpr_delta_vs_wazuh": None if wazuh_fpr is None or hybrid_fpr is None else hybrid_fpr - wazuh_fpr,
        "alert_count_delta_vs_wazuh": None
        if wazuh_alert_count is None or hybrid_alert_count is None
        else hybrid_alert_count - wazuh_alert_count,
    }
    return {key: json_safe(value) for key, value in answer.items()}


def check_required_files(root: Path) -> list[dict[str, str]]:
    rows = []
    for rel_path in REQUIRED_FILES:
        path = root / rel_path
        if not path.exists():
            rows.append(qa_row(f"file_{Path(rel_path).stem}", "Kötelező eredményfájlok", "FAIL", f"hiányzik: {rel_path}"))
        elif path.suffix == ".csv" and (read_csv(path) is None or read_csv(path).empty):
            rows.append(qa_row(f"file_{Path(rel_path).stem}", "Kötelező eredményfájlok", "FAIL", f"üres CSV: {rel_path}"))
        else:
            rows.append(qa_row(f"file_{Path(rel_path).stem}", "Kötelező eredményfájlok", "PASS", f"rendben: {rel_path}"))
    return rows


def check_comparison(comparison: pd.DataFrame | None) -> list[dict[str, str]]:
    rows = []
    if comparison is None or comparison.empty:
        return [qa_row("comparison_readable", "Metrikai konzisztencia", "FAIL", "metrics_comparison.csv nem olvasható")]
    configs = set(comparison.get("configuration", pd.Series(dtype=str)).astype(str))
    missing = [name for name in EXPECTED_CONFIGURATIONS if name not in configs]
    rows.append(
        qa_row(
            "comparison_configurations",
            "Metrikai konzisztencia",
            "FAIL" if missing else "PASS",
            "hiányzó konfigurációk: " + ", ".join(missing) if missing else "minden konfiguráció szerepel",
        )
    )
    if "n_samples" in comparison.columns:
        values = pd.to_numeric(comparison["n_samples"], errors="coerce").dropna().astype(int).tolist()
        consistent = len(values) == len(comparison) and len(set(values)) == 1
        rows.append(
            qa_row(
                "n_samples_consistency",
                "Metrikai konzisztencia",
                "PASS" if consistent else "FAIL",
                f"n_samples értékek: {values}",
            )
        )
    else:
        rows.append(qa_row("n_samples_consistency", "Metrikai konzisztencia", "WARN", "n_samples oszlop nincs a comparison táblában"))

    for metric in RANGE_METRICS:
        if metric not in comparison.columns:
            rows.append(qa_row(f"range_{metric}", "Metrikai konzisztencia", "FAIL", f"hiányzó metrika: {metric}"))
            continue
        values = pd.to_numeric(comparison[metric], errors="coerce")
        valid = values.notna().all() and ((0.0 <= values) & (values <= 1.0)).all()
        rows.append(qa_row(f"range_{metric}", "Metrikai konzisztencia", "PASS" if valid else "FAIL", f"{metric} tartományellenőrzés"))

    if "alert_count" in comparison.columns:
        alerts = pd.to_numeric(comparison["alert_count"], errors="coerce")
        valid_alerts = alerts.notna().all() and (alerts >= 0).all()
        rows.append(
            qa_row(
                "alert_count_non_negative",
                "Metrikai konzisztencia",
                "PASS" if valid_alerts else "FAIL",
                "alert_count nem negatív",
            )
        )
    else:
        rows.append(qa_row("alert_count_non_negative", "Metrikai konzisztencia", "FAIL", "hiányzó alert_count oszlop"))

    for config in ["Wazuh-only", "AE-Minimal lab"]:
        f1 = metric_for(comparison, config, "f1")
        rows.append(
            qa_row(
                f"{config.lower().replace(' ', '_')}_valid",
                "Metrikai konzisztencia",
                "PASS" if f1 is not None else "FAIL",
                f"{config} sor {'érvényes' if f1 is not None else 'nem érvényes'}",
            )
        )
    best = best_hybrid_by_f1(comparison)
    rows.append(
        qa_row(
            "hybrid_row_valid",
            "Metrikai konzisztencia",
            "PASS" if best["configuration"] is not None else "FAIL",
            f"legjobb hibrid: {best['configuration']}" if best["configuration"] else "nincs érvényes hibrid sor",
        )
    )
    return rows


def check_label_coverage(metric_frames: list[pd.DataFrame | None]) -> dict[str, str]:
    n_attack_values = []
    n_benign_values = []
    for df in metric_frames:
        if df is None or df.empty or "n_attack" not in df.columns or "n_benign" not in df.columns:
            continue
        n_attack_values.extend(pd.to_numeric(df["n_attack"], errors="coerce").dropna().astype(int).tolist())
        n_benign_values.extend(pd.to_numeric(df["n_benign"], errors="coerce").dropna().astype(int).tolist())
    has_attack = any(value > 0 for value in n_attack_values)
    has_benign = any(value > 0 for value in n_benign_values)
    if has_attack and has_benign:
        return qa_row("label_coverage", "Metrikai konzisztencia", "PASS", "van legalább 1 benign és 1 attack esemény")
    return qa_row("label_coverage", "Metrikai konzisztencia", "FAIL", "nem igazolható benign és attack esemény is")


def check_provenance(root: Path) -> dict[str, str]:
    provenance = load_provenance(root / "reports/real_measurement/measurement_provenance.json")
    valid, errors = validate_provenance_payload(provenance)
    if valid:
        return qa_row("measurement_provenance", "Adateredet", "PASS", "verified_real_lab provenance rendelkezésre áll")
    return qa_row(
        "measurement_provenance",
        "Adateredet",
        "FAIL",
        "Az eredmények metrikailag értelmezhetők lehetnek, de provenance hiányában nem tekinthetők végleges real-lab dolgozati eredménynek.",
        "; ".join(errors),
    )


def readiness_status(rows: list[dict[str, str]]) -> str:
    if any(row["status"] == "FAIL" for row in rows):
        return "NOT_READY"
    if any(row["status"] == "WARN" for row in rows):
        return "READY_WITH_LIMITATIONS"
    return "READY"


def write_quality_report(rows: list[dict[str, str]], output_path: Path, status: str) -> None:
    lines = [
        "# Real-lab post-run QA riport",
        "",
        f"Dolgozati beemelhetőségi státusz: **{status}**",
        "",
        "| Ellenőrzés | Kategória | Státusz | Üzenet | Javaslat |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['check_id']} | {row['category']} | {row['status']} | {row['message']} | {row['recommendation']} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def bool_text(value: bool | None) -> str:
    if value is None:
        return "nem dönthető el"
    return "igen" if value else "nem"


def write_thesis_readiness(answer: dict[str, Any], output_path: Path, status: str) -> None:
    f1_delta = answer.get("f1_delta_vs_wazuh")
    fpr_delta = answer.get("fpr_delta_vs_wazuh")
    alert_delta = answer.get("alert_count_delta_vs_wazuh")
    improved = None if f1_delta is None else f1_delta > 0
    fpr_worse = None if fpr_delta is None else fpr_delta > 0
    alert_increased = None if alert_delta is None else alert_delta > 0
    usable = status in {"READY", "READY_WITH_LIMITATIONS"}
    lines = [
        "# Real-lab dolgozati beemelhetőség",
        "",
        f"Státusz: **{status}**",
        "",
        f"Használható a 6. fejezetben: **{bool_text(usable)}**",
        "",
        f"Legjobb F1 szerinti hibrid konfiguráció: `{answer.get('best_hybrid_by_f1') or 'nincs adat'}`.",
        f"Javult-e a Wazuh-only eredményhez képest: **{bool_text(improved)}**.",
        f"Romlott-e a false positive rate: **{bool_text(fpr_worse)}**.",
        f"Nőtt-e a riasztásszám: **{bool_text(alert_increased)}**.",
        "",
        "## Értelmezés",
    ]
    if status == "NOT_READY":
        lines.append("Az eredmények jelen állapotban nem emelhetők be végleges real-lab mérési eredményként.")
    elif improved:
        lines.append("A vizsgált lab mérés alapján a legjobb hibrid konfigurációnál F1 javulás figyelhető meg a Wazuh-only baseline-hoz képest.")
    else:
        lines.append("A vizsgált lab mérés alapján nem igazolható egyértelmű F1 javulás a Wazuh-only baseline-hoz képest.")
    lines.extend(
        [
            "",
            "## Korlátok",
            "- A mérés lab környezetben készült, nem hosszú idejű éles SOC-validáció.",
            "- A hibrid eredmény az event_id alapú illesztés és az időszinkron pontosságától függ.",
            "- Az AE-only ág offline scoring, ezért natív detektálási idő csak a Wazuh riasztásoknál értelmezhető.",
            "- A metrikák csak az adott lab eseménykészletre vonatkoznak.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_quality_gate(root: Path, output_dir: Path) -> dict[str, Any]:
    rows = check_required_files(root)
    comparison_path = root / "results/real_comparison/metrics_comparison.csv"
    comparison = read_csv(comparison_path)
    rows.extend(check_comparison(comparison))
    metric_frames = [
        read_csv(root / "results/wazuh_real/metrics_summary.csv"),
        read_csv(root / "results/ae_lab/metrics_summary.csv"),
        read_csv(root / "results/hybrid_real/metrics_summary.csv"),
    ]
    rows.append(check_label_coverage(metric_frames))
    rows.append(check_provenance(root))
    status = readiness_status(rows)
    answer = compute_research_answer(comparison)
    answer["thesis_readiness_status"] = status

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "postrun_quality_summary.csv"
    report_path = output_dir / "postrun_quality_report.md"
    answer_path = output_dir / "research_question_answer.json"
    readiness_path = output_dir / "thesis_readiness.md"
    metadata_path = output_dir / "postrun_metadata.json"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    write_quality_report(rows, report_path, status)
    answer_path.write_text(json.dumps(answer, indent=2, ensure_ascii=False), encoding="utf-8")
    write_thesis_readiness(answer, readiness_path, status)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "output_dir": str(output_dir),
        "thesis_readiness_status": status,
        "fail_count": sum(row["status"] == "FAIL" for row in rows),
        "warn_count": sum(row["status"] == "WARN" for row in rows),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "status": status,
        "summary": summary_path,
        "report": report_path,
        "answer": answer_path,
        "readiness": readiness_path,
        "metadata": metadata_path,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    result = run_quality_gate(Path(args.root), Path(args.output_dir))
    for key in ["summary", "report", "answer", "readiness", "metadata"]:
        print(f"[OK] Kimenet: {result[key]}")
    if result["status"] == "NOT_READY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
