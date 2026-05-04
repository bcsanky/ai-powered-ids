from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "alert_count",
]

DISPLAY_NAMES = {
    "ae_minimal": "AE-Minimal",
    "ae_context": "AE-Context",
    "baseline_stat": "Statisztikai baseline",
    "rule_proxy": "Szabályalapú proxy baseline",
    "hybrid": "Offline hibrid",
    "baseline_wazuh_real": "Natív Wazuh baseline",
}

RISK_ORDER = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "normal": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", default="results/final/comparison/metrics_comparison.csv")
    parser.add_argument("--scored-events", default=None)
    parser.add_argument("--case-summary", default=None)
    parser.add_argument("--output-dir", default="reports/final")
    return parser.parse_args()


def load_comparison(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó összehasonlító CSV: {path}")
    df = pd.read_csv(path)
    if "config_name" not in df.columns or "status" not in df.columns:
        raise ValueError("Az összehasonlító CSV-ben szükséges a config_name és status oszlop.")
    return df


def load_scored_events(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("A pontozott események támogatott formátumai: .jsonl, .csv")


def load_case_summary(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def format_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def comparison_subset(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["config_name", "status", *[c for c in METRIC_COLUMNS if c in df.columns]]
    out = df[columns].copy()
    out["configuration"] = out["config_name"].map(DISPLAY_NAMES).fillna(out["config_name"])
    return out[["configuration", "status", *[c for c in METRIC_COLUMNS if c in out.columns]]]


def top_risk_events(scored_events: pd.DataFrame | None, limit: int = 10) -> pd.DataFrame:
    if scored_events is None or scored_events.empty:
        return pd.DataFrame()

    df = scored_events.copy()
    if "risk_level" not in df.columns:
        return pd.DataFrame()

    df["_risk_rank"] = df["risk_level"].map(RISK_ORDER).fillna(0)
    score_col = "anomaly_score" if "anomaly_score" in df.columns else None
    if score_col:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
        df = df.sort_values(["_risk_rank", score_col], ascending=[False, False])
    else:
        df = df.sort_values("_risk_rank", ascending=False)

    columns = [
        col
        for col in [
            "event_id",
            "risk_level",
            "anomaly_score",
            "threshold_name",
            "threshold_value",
            "ml_alert",
            "rule_flag",
            "rule_level",
            "reason",
        ]
        if col in df.columns
    ]
    return df[columns].head(limit)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Nincs megjeleníthető adat._"
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [format_number(row[col]) for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def status_text(comparison: pd.DataFrame, config_name: str) -> str:
    row = comparison.loc[comparison["config_name"] == config_name]
    if row.empty:
        return "nem szerepel az összehasonlító táblázatban"
    return str(row.iloc[0]["status"])


def ae_comparison_sentence(comparison: pd.DataFrame) -> str:
    ae_min = comparison.loc[comparison["config_name"] == "ae_minimal"]
    ae_ctx = comparison.loc[comparison["config_name"] == "ae_context"]
    if ae_min.empty or ae_ctx.empty:
        return "Az AE-Minimal és AE-Context közvetlen összehasonlítása nem teljes, mert az egyik eredménysor hiányzik."
    if ae_min.iloc[0]["status"] != "ok" or ae_ctx.iloc[0]["status"] != "ok":
        return "Az AE-Minimal és AE-Context közvetlen összehasonlítása csak mindkét validált eredménysor mellett értelmezhető."

    min_f1 = ae_min.iloc[0].get("f1")
    ctx_f1 = ae_ctx.iloc[0].get("f1")
    if pd.isna(min_f1) or pd.isna(ctx_f1):
        return "Az AE-Minimal és AE-Context metrikái rendelkezésre állnak, de az F1 mező nem teljes."
    diff = float(ctx_f1) - float(min_f1)
    if abs(diff) < 0.00005:
        return (
            "Az AE-Context F1 értéke az AE-Minimal eredményéhez képest "
            "gyakorlatilag azonos."
        )
    direction = "magasabb" if diff > 0 else "alacsonyabb" if diff < 0 else "azonos"
    return (
        "Az AE-Context F1 értéke az AE-Minimal eredményéhez képest "
        f"{direction}; az eltérés abszolút értéke {abs(diff):.4f}."
    )


def build_markdown_report(
    *,
    comparison: pd.DataFrame,
    scored_events: pd.DataFrame | None,
    case_summary: pd.DataFrame | None,
    created_at: datetime,
) -> str:
    dashboard = comparison_subset(comparison)
    top_events = top_risk_events(scored_events)
    validated = dashboard.loc[dashboard["status"] == "ok", "configuration"].tolist()

    lines = [
        "# Szakértői biztonsági összefoglaló",
        "",
        f"Futtatás dátuma: {created_at.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Validált konfigurációk",
        "",
        ", ".join(validated) if validated else "Nincs validált konfiguráció az összehasonlító táblázatban.",
        "",
        "## Fő összehasonlító metrikák",
        "",
        markdown_table(dashboard),
        "",
        "## AE-Minimal és AE-Context összevetése",
        "",
        ae_comparison_sentence(comparison),
        "",
        "## Baseline és hibrid státusz",
        "",
        f"- Statisztikai baseline: {status_text(comparison, 'baseline_stat')}",
        f"- Szabályalapú proxy baseline: {status_text(comparison, 'rule_proxy')}",
        f"- Offline hibrid: {status_text(comparison, 'hybrid')}",
        f"- Natív Wazuh baseline: {status_text(comparison, 'baseline_wazuh_real')}",
        "",
    ]

    if top_events.empty:
        lines.extend(
            [
                "## Legmagasabb kockázatú pontozott események",
                "",
                "Nem áll rendelkezésre pontozott eseménylista, vagy nem tartalmaz risk_level mezőt.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Legmagasabb kockázatú pontozott események",
                "",
                markdown_table(top_events),
                "",
            ]
        )

    if case_summary is not None and not case_summary.empty:
        lab_columns = [
            col
            for col in [
                "scenario",
                "event_count",
                "medium_count",
                "high_count",
                "critical_count",
                "max_anomaly_score",
            ]
            if col in case_summary.columns
        ]
        lines.extend(
            [
                "## Lab/replay demonstráció",
                "",
                markdown_table(case_summary[lab_columns]),
                "",
                "A lab/replay demonstráció kontrollált eseménysoron mutatja be a scoring és priorizálási folyamatot. Ez nem éles üzemű SOC eseményfolyam és önmagában nem bizonyít éles üzemi teljesítményt.",
                "",
            ]
        )

    lines.extend(
        [
            "## Korlátok",
            "",
            "- A mérés CIC-IDS2017 flow-alapú adatokon történt.",
            "- A Wazuh logorientált adatmodellje és a CIC flow jellemzői között szerkezeti eltérés van.",
            "- A rule_proxy kontrollált flow-alapú szabályproxy, nem natív Wazuh teljesítménymérés.",
            "- A hibrid eredmény offline, azonos teszthalmaz-sorrenden alapuló kiértékelés.",
            "- A riport szakdolgozati demonstrációs összefoglaló, nem éles SOC incidensjelentés.",
            "",
        ]
    )

    return "\n".join(lines)


def markdown_to_html(markdown_text: str) -> str:
    escaped = html.escape(markdown_text)
    body = escaped.replace("\n", "<br>\n")
    return (
        "<!doctype html>\n"
        "<html lang=\"hu\">\n"
        "<head><meta charset=\"utf-8\"><title>Szakértői biztonsági összefoglaló</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:2rem auto;line-height:1.5;}"
        "code{background:#f2f2f2;padding:0.1rem 0.25rem;}br{line-height:1.35;}</style></head>\n"
        f"<body>{body}</body>\n"
        "</html>\n"
    )


def write_report(
    *,
    comparison_path: Path,
    scored_events_path: Path | None,
    case_summary_path: Path | None = None,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = load_comparison(comparison_path)
    scored_events = load_scored_events(scored_events_path)
    case_summary = load_case_summary(case_summary_path)
    created_at = datetime.now()

    dashboard = comparison_subset(comparison)
    dashboard_path = output_dir / "dashboard_summary.csv"
    dashboard.to_csv(dashboard_path, index=False)

    markdown = build_markdown_report(
        comparison=comparison,
        scored_events=scored_events,
        case_summary=case_summary,
        created_at=created_at,
    )
    md_path = output_dir / "security_report.md"
    html_path = output_dir / "security_report.html"
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(markdown_to_html(markdown), encoding="utf-8")

    return {
        "markdown": md_path,
        "html": html_path,
        "dashboard": dashboard_path,
    }


def main() -> None:
    args = parse_args()
    scored_path = Path(args.scored_events) if args.scored_events else None
    case_summary_path = Path(args.case_summary) if args.case_summary else None
    outputs = write_report(
        comparison_path=Path(args.comparison),
        scored_events_path=scored_path,
        case_summary_path=case_summary_path,
        output_dir=Path(args.output_dir),
    )
    print(f"[OK] Szakértői jelentés: {outputs['markdown']}")
    print(f"[OK] HTML összefoglaló: {outputs['html']}")
    print(f"[OK] Dashboard CSV: {outputs['dashboard']}")


if __name__ == "__main__":
    main()
