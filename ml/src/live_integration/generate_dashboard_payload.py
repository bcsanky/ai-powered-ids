from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--output-dir", default="reports/live_integration")
    return parser.parse_args()


def read_enriched_alerts(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó enriched alert CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Üres enriched alert CSV: {path}")
    return df


def bool_sum(series: pd.Series) -> int:
    values = series.astype(str).str.lower().isin({"1", "true", "yes", "igen"})
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return int((values | (numeric > 0)).sum())


def count_status(df: pd.DataFrame, value: str) -> int:
    if "risk_level" not in df.columns:
        return 0
    return int((df["risk_level"].astype(str) == value).sum())


def top_values(df: pd.DataFrame, column: str, limit: int = 10) -> list[dict[str, Any]]:
    if column not in df.columns:
        return []
    series = df[column].fillna("").astype(str).str.strip()
    series = series[series.ne("")]
    if series.empty:
        return []
    counts = series.value_counts().head(limit)
    return [{"value": str(index), "count": int(count)} for index, count in counts.items()]


def build_cards(df: pd.DataFrame) -> list[dict[str, Any]]:
    cards = [
        ("total_alerts", len(df), "Összes feldolgozott Wazuh alert"),
        ("scored_alerts", int((df["top_level_status"].astype(str) == "scored").sum()), "ML pontszámmal ellátott alert"),
        (
            "unmatched_alerts",
            int((df["top_level_status"].astype(str) == "unmatched").sum()),
            "Feature mapping nélkül maradt alert",
        ),
        ("ml_positive", bool_sum(df["ml_alert"]) if "ml_alert" in df.columns else 0, "ML pozitív döntések"),
        ("wazuh_positive", bool_sum(df["wazuh_pred"]) if "wazuh_pred" in df.columns else 0, "Wazuh pozitív jelzések"),
        (
            "hybrid_positive",
            bool_sum(df["hybrid_or_pred"]) if "hybrid_or_pred" in df.columns else 0,
            "Hibrid pozitív döntések",
        ),
        ("critical_count", count_status(df, "critical"), "Kritikus kockázati szint"),
        ("high_count", count_status(df, "high"), "Magas kockázati szint"),
        ("medium_count", count_status(df, "medium"), "Közepes kockázati szint"),
        ("normal_count", count_status(df, "normal"), "Normál kockázati szint"),
    ]
    return [{"metric": metric, "value": int(value), "description": description} for metric, value, description in cards]


def markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Live integration dashboard összefoglaló",
        "",
        "Az összefoglaló az enriched alert CSV kimenet mezőiből készült. Nem önálló benchmark, hanem integrációs áttekintés.",
        "",
        "## Fő kártyák",
        "",
        "| Mutató | Érték | Leírás |",
        "| --- | ---: | --- |",
    ]
    for card in payload["cards"]:
        lines.append(f"| {card['metric']} | {card['value']} | {card['description']} |")
    lines.extend(["", "## Top rule_id értékek", ""])
    lines.extend(markdown_top_table(payload["top_rule_ids"], "rule_id"))
    lines.extend(["", "## Top source_ip értékek", ""])
    lines.extend(markdown_top_table(payload["top_source_ips"], "source_ip"))
    lines.extend(["", "## Top scenario értékek", ""])
    lines.extend(markdown_top_table(payload["top_scenarios"], "scenario"))
    return "\n".join(lines) + "\n"


def markdown_top_table(rows: list[dict[str, Any]], label: str) -> list[str]:
    if not rows:
        return ["Nincs adat."]
    lines = ["| Érték | Darabszám |", "| --- | ---: |"]
    for row in rows:
        lines.append(f"| {row['value']} | {row['count']} |")
    return lines


def generate_dashboard_payload(input_path: Path, output_dir: Path, summary_path: Path | None = None) -> dict[str, Path]:
    df = read_enriched_alerts(input_path)
    summary_exists = bool(summary_path and summary_path.exists())
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(input_path),
        "summary_input": "" if summary_path is None else str(summary_path),
        "summary_input_exists": summary_exists,
        "cards": build_cards(df),
        "top_rule_ids": top_values(df, "rule_id"),
        "top_source_ips": top_values(df, "source_ip"),
        "top_scenarios": top_values(df, "scenario"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "dashboard_payload.json"
    summary_md_path = output_dir / "dashboard_summary.md"
    cards_path = output_dir / "dashboard_cards.csv"
    payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(payload["cards"]).to_csv(cards_path, index=False)
    summary_md_path.write_text(markdown_summary(payload), encoding="utf-8")
    return {"payload": payload_path, "summary": summary_md_path, "cards": cards_path}


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary) if args.summary else Path(args.output_dir) / "enrichment_summary.csv"
    outputs = generate_dashboard_payload(Path(args.input), Path(args.output_dir), summary_path)
    for path in outputs.values():
        print(f"[OK] {path}")


if __name__ == "__main__":
    main()

