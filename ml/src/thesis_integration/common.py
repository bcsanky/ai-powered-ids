from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.repo_hygiene.common import is_demo_or_fixture_path, load_provenance, sha256_file, validate_provenance_payload


EXPECTED_CONFIGURATIONS = [
    "Wazuh-only",
    "AE-Minimal lab",
    "Hybrid OR",
    "Hybrid weighted",
    "Hybrid priority",
]
HYBRID_CONFIGURATIONS = ["Hybrid OR", "Hybrid weighted", "Hybrid priority"]
RANGE_METRICS = ["precision", "recall", "f1", "false_positive_rate"]
METRIC_COLUMNS = [
    "configuration",
    "TP",
    "FP",
    "TN",
    "FN",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "false_negative_rate",
    "alert_count",
    "mean_ttd",
    "median_ttd",
    "n_samples",
    "n_attack",
    "n_benign",
]
METRIC_LABELS = {
    "configuration": "Konfiguráció",
    "strategy": "Stratégia",
    "TP": "TP",
    "FP": "FP",
    "TN": "TN",
    "FN": "FN",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "false_positive_rate": "Hamis pozitív arány",
    "false_negative_rate": "Hamis negatív arány",
    "alert_count": "Riasztásszám",
    "mean_ttd": "Átlagos TTD (s)",
    "median_ttd": "Medián TTD (s)",
    "n_samples": "Mintaszám",
    "n_attack": "Támadó esemény",
    "n_benign": "Benign esemény",
}
COUNT_COLUMNS = {"TP", "FP", "TN", "FN", "alert_count", "n_samples", "n_attack", "n_benign", "failed_events"}
RATIO_COLUMNS = {
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "false_negative_rate",
    "roc_auc",
    "f1_delta_vs_wazuh",
    "recall_delta_vs_wazuh",
    "fpr_delta_vs_wazuh",
}
DEFAULT_PROVENANCE = Path("reports/real_measurement/measurement_provenance.json")


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text_optional(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_csv_required(path: Path) -> pd.DataFrame:
    df = read_csv_optional(path)
    if df is None:
        raise FileNotFoundError(f"Hiányzó CSV: {path}")
    if df.empty:
        raise ValueError(f"Üres CSV: {path}")
    return df


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def numeric(value: Any) -> float | None:
    if value is None:
        return None
    converted = pd.to_numeric(value, errors="coerce")
    if pd.isna(converted):
        return None
    return float(converted)


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def format_metric(value: Any, column: str) -> str:
    if value is None:
        return "nincs adat"
    try:
        if pd.isna(value):
            return "nincs adat"
    except TypeError:
        pass
    if column in COUNT_COLUMNS:
        converted = numeric(value)
        return "nincs adat" if converted is None else str(int(converted))
    if column in RATIO_COLUMNS or column.endswith("_rate") or column.endswith("_f1"):
        converted = numeric(value)
        return "nincs adat" if converted is None else f"{converted:.4f}"
    if column in {"mean_ttd", "median_ttd", "time_to_detection_sec"}:
        converted = numeric(value)
        return "nincs adat" if converted is None else f"{converted:.4f}"
    return str(value)


def escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def markdown_table(rows: list[dict[str, Any]], columns: list[str], labels: dict[str, str] | None = None) -> str:
    labels = labels or {}
    header = [labels.get(column, METRIC_LABELS.get(column, column)) for column in columns]
    lines = [
        "| " + " | ".join(escape_markdown_cell(cell) for cell in header) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        cells = [escape_markdown_cell(format_metric(row.get(column), column)) for column in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def dataframe_markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    present = [column for column in columns if column in df.columns]
    return markdown_table(df[present].to_dict("records"), present)


def metric_for(df: pd.DataFrame | None, configuration: str | None, metric: str) -> float | None:
    if df is None or df.empty or configuration is None:
        return None
    if "configuration" not in df.columns or metric not in df.columns:
        return None
    matched = df[df["configuration"].astype(str) == configuration]
    if matched.empty:
        return None
    return numeric(matched.iloc[0][metric])


def get_config_row(df: pd.DataFrame | None, configuration: str) -> dict[str, Any] | None:
    if df is None or df.empty or "configuration" not in df.columns:
        return None
    matched = df[df["configuration"].astype(str) == configuration]
    if matched.empty:
        return None
    return matched.iloc[0].to_dict()


def best_hybrid_by_f1(comparison: pd.DataFrame | None) -> dict[str, Any]:
    if comparison is None or comparison.empty or "configuration" not in comparison.columns or "f1" not in comparison.columns:
        return {"configuration": None, "f1": None}
    hybrids = comparison[comparison["configuration"].astype(str).isin(HYBRID_CONFIGURATIONS)].copy()
    hybrids["f1_numeric"] = pd.to_numeric(hybrids["f1"], errors="coerce")
    hybrids = hybrids.dropna(subset=["f1_numeric"])
    if hybrids.empty:
        return {"configuration": None, "f1": None}
    best = hybrids.sort_values("f1_numeric", ascending=False, kind="mergesort").iloc[0]
    return {"configuration": str(best["configuration"]), "f1": float(best["f1_numeric"])}


def compute_research_answer(comparison: pd.DataFrame | None) -> dict[str, Any]:
    best = best_hybrid_by_f1(comparison)
    best_name = best["configuration"]
    wazuh_f1 = metric_for(comparison, "Wazuh-only", "f1")
    ae_f1 = metric_for(comparison, "AE-Minimal lab", "f1")
    best_hybrid_f1 = best["f1"]
    wazuh_recall = metric_for(comparison, "Wazuh-only", "recall")
    wazuh_fpr = metric_for(comparison, "Wazuh-only", "false_positive_rate")
    wazuh_alert_count = metric_for(comparison, "Wazuh-only", "alert_count")
    hybrid_recall = metric_for(comparison, best_name, "recall")
    hybrid_fpr = metric_for(comparison, best_name, "false_positive_rate")
    hybrid_alert_count = metric_for(comparison, best_name, "alert_count")
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


def load_research_answer(path: Path, comparison: pd.DataFrame | None) -> dict[str, Any]:
    loaded = read_json_optional(path)
    if loaded:
        return loaded
    return compute_research_answer(comparison)


def provenance_validation(path: Path = DEFAULT_PROVENANCE) -> tuple[dict[str, Any] | None, bool, list[str]]:
    payload = load_provenance(path)
    valid, errors = validate_provenance_payload(payload)
    return payload, valid, errors


def sha256_for_path(path: Path) -> str:
    return sha256_file(path)


def is_forbidden_real_lab_path(path: str | Path) -> bool:
    return is_demo_or_fixture_path(path)


def reject_forbidden_real_lab_path(path: str | Path) -> None:
    if is_forbidden_real_lab_path(path):
        raise ValueError(f"Nem használható real-lab dolgozati bemenetként: {path}")


def provenance_status(path: Path = DEFAULT_PROVENANCE) -> str:
    _, valid, errors = provenance_validation(path)
    if valid:
        return "verified_real_lab"
    if errors == ["measurement_provenance.json hiányzik"]:
        return "missing_provenance"
    return "invalid_provenance"


def can_use_real_lab_phrase(path: Path = DEFAULT_PROVENANCE) -> bool:
    return provenance_status(path) == "verified_real_lab"


def comparison_ready_checks(comparison: pd.DataFrame | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if comparison is None or comparison.empty:
        return [
            {
                "check_id": "comparison_readable",
                "category": "Metrikák",
                "status": "FAIL",
                "message": "metrics_comparison.csv nem olvasható vagy üres",
                "recommendation": "Futtasd a real comparison lépést valós lab bemenetekkel.",
            }
        ]

    configs = set(comparison.get("configuration", pd.Series(dtype=str)).astype(str))
    missing = [config for config in EXPECTED_CONFIGURATIONS if config not in configs]
    rows.append(
        {
            "check_id": "expected_configurations",
            "category": "Metrikák",
            "status": "PASS" if not missing else "FAIL",
            "message": "minden elvárt konfiguráció szerepel" if not missing else "hiányzó konfigurációk: " + ", ".join(missing),
            "recommendation": "" if not missing else "Ellenőrizd a final-real-hybrid pipeline eredményeit.",
        }
    )

    for metric in RANGE_METRICS:
        if metric not in comparison.columns:
            rows.append(
                {
                    "check_id": f"metric_{metric}",
                    "category": "Metrikák",
                    "status": "FAIL",
                    "message": f"hiányzó metrika: {metric}",
                    "recommendation": "A comparison táblát újra kell előállítani.",
                }
            )
            continue
        values = pd.to_numeric(comparison[metric], errors="coerce")
        valid = values.notna().all() and ((0.0 <= values) & (values <= 1.0)).all()
        rows.append(
            {
                "check_id": f"range_{metric}",
                "category": "Metrikák",
                "status": "PASS" if valid else "FAIL",
                "message": f"{metric} tartományellenőrzés",
                "recommendation": "" if valid else "A metrika értékeinek 0 és 1 között kell lenniük.",
            }
        )

    if "n_samples" in comparison.columns:
        values = pd.to_numeric(comparison["n_samples"], errors="coerce").dropna().astype(int).tolist()
        consistent = len(values) == len(comparison) and len(set(values)) == 1
        rows.append(
            {
                "check_id": "n_samples_consistency",
                "category": "Metrikák",
                "status": "PASS" if consistent else "FAIL",
                "message": f"n_samples értékek: {values}",
                "recommendation": "" if consistent else "Az összehasonlított soroknak ugyanazon eseményszámra kell vonatkozniuk.",
            }
        )
    else:
        rows.append(
            {
                "check_id": "n_samples_consistency",
                "category": "Metrikák",
                "status": "WARN",
                "message": "n_samples oszlop nincs a comparison táblában",
                "recommendation": "A dolgozatban jelezni kell, ha az eseményszám külön táblából kerül ellenőrzésre.",
            }
        )

    for config in ["Wazuh-only", "AE-Minimal lab"]:
        rows.append(
            {
                "check_id": f"{config.lower().replace(' ', '_')}_row",
                "category": "Metrikák",
                "status": "PASS" if metric_for(comparison, config, "f1") is not None else "FAIL",
                "message": f"{config} sor ellenőrzése",
                "recommendation": "",
            }
        )
    rows.append(
        {
            "check_id": "hybrid_row",
            "category": "Metrikák",
            "status": "PASS" if best_hybrid_by_f1(comparison)["configuration"] else "FAIL",
            "message": "legalább egy hibrid sor értelmezhető",
            "recommendation": "",
        }
    )
    return rows


def label_coverage_from_frames(frames: list[pd.DataFrame | None]) -> dict[str, str]:
    attacks: list[int] = []
    benigns: list[int] = []
    for df in frames:
        if df is None or df.empty:
            continue
        if "n_attack" in df.columns:
            attacks.extend(pd.to_numeric(df["n_attack"], errors="coerce").dropna().astype(int).tolist())
        if "n_benign" in df.columns:
            benigns.extend(pd.to_numeric(df["n_benign"], errors="coerce").dropna().astype(int).tolist())
    has_attack = any(value > 0 for value in attacks)
    has_benign = any(value > 0 for value in benigns)
    return {
        "check_id": "label_coverage",
        "category": "Metrikák",
        "status": "PASS" if has_attack and has_benign else "FAIL",
        "message": "van benign és attack esemény" if has_attack and has_benign else "nem igazolható benign és attack esemény is",
        "recommendation": "" if has_attack and has_benign else "A lab mérési készletnek mindkét címketípust tartalmaznia kell.",
    }


def status_from_rows(rows: list[dict[str, str]]) -> str:
    if any(row.get("status") == "FAIL" for row in rows):
        return "NOT_READY"
    if any(row.get("status") == "WARN" for row in rows):
        return "READY_WITH_LIMITATIONS"
    return "READY"


def cautious_improvement_sentence(answer: dict[str, Any]) -> str:
    delta = answer.get("f1_delta_vs_wazuh")
    if delta is None:
        return "A Wazuh-only baseline-hoz viszonyított F1 változás a rendelkezésre álló adatokból nem dönthető el."
    if delta > 0:
        return "a vizsgált lab mérésben F1 alapján javulás figyelhető meg a Wazuh-only baseline-hoz képest."
    return "a vizsgált lab mérésben F1 alapján nem igazolható egyértelmű javulás a Wazuh-only baseline-hoz képest."


def tradeoff_sentences(answer: dict[str, Any]) -> list[str]:
    sentences = [cautious_improvement_sentence(answer)]
    recall_delta = answer.get("recall_delta_vs_wazuh")
    fpr_delta = answer.get("fpr_delta_vs_wazuh")
    alert_delta = answer.get("alert_count_delta_vs_wazuh")
    if recall_delta is not None and recall_delta > 0 and fpr_delta is not None and fpr_delta > 0:
        sentences.append(
            "A recall növekedése magasabb hamis pozitív aránnyal járt, ezért az eredmény kompromisszumként értelmezendő."
        )
    elif recall_delta is not None and recall_delta > 0:
        sentences.append("A legjobb hibrid stratégia recall értéke a Wazuh-only sorhoz képest magasabb volt.")
    if fpr_delta is not None and fpr_delta > 0:
        sentences.append("A hamis pozitív arány emelkedése üzemeltetési kockázatként jelenik meg.")
    if alert_delta is not None and alert_delta > 0:
        sentences.append("A riasztásszám növekedése nagyobb elemzői terhelést okozhat.")
    elif alert_delta is not None and alert_delta <= 0:
        sentences.append("A legjobb hibrid sor nem növelte a riasztásszámot a Wazuh-only sorhoz képest.")
    return sentences


def provenance_warning_text(provenance_path: Path = DEFAULT_PROVENANCE) -> str:
    _, valid, errors = provenance_validation(provenance_path)
    if valid:
        return "A mérési bemenetek provenance fájllal igazolt real-lab eredetűek."
    return (
        "Figyelmeztetés: nincs verified real_lab provenance. A szöveg csak szerkezeti vázlatként használható, "
        "végleges dolgozati eredményként nem."
        + (f" Ellenőrzési megjegyzés: {'; '.join(errors)}." if errors else "")
    )
