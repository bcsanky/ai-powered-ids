from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from ml.src.wazuh_baseline.parse_wazuh_alerts import load_alert_objects, normalize_alert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/wazuh/alerts.jsonl")
    parser.add_argument("--output-dir", default="reports/wazuh_export")
    return parser.parse_args()


def top_lines(series: pd.Series, limit: int = 10) -> list[str]:
    values = series.fillna("").astype(str)
    values = values[values.str.len() > 0]
    if values.empty:
        return ["Nincs elérhető adat."]
    return [f"- `{idx}`: {count}" for idx, count in values.value_counts().head(limit).items()]


def build_timeline(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(columns=["time_bucket", "alert_count"])
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return pd.DataFrame(columns=["time_bucket", "alert_count"])
    span = ts.max() - ts.min()
    freq = "min" if span <= pd.Timedelta(hours=6) else "h"
    timeline = ts.dt.floor(freq).value_counts().sort_index()
    return pd.DataFrame(
        {
            "time_bucket": [idx.strftime("%Y-%m-%dT%H:%M:%SZ") for idx in timeline.index],
            "alert_count": timeline.values,
        }
    )


def summarize_wazuh_export(input_path: Path, output_dir: Path) -> dict[str, Path]:
    alerts = load_alert_objects(input_path)
    rows = [normalize_alert(alert) for alert in alerts if isinstance(alert, dict)]
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["timestamp", "rule_id", "rule_level", "rule_description", "agent_name", "source_ip"])

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "wazuh_export_summary.md"
    rule_summary_path = output_dir / "wazuh_export_rule_summary.csv"
    timeline_path = output_dir / "wazuh_export_timeline.csv"
    metadata_path = output_dir / "run_metadata.json"

    timestamps = pd.to_datetime(df.get("timestamp", pd.Series(dtype=str)), utc=True, errors="coerce").dropna()
    first_ts = timestamps.min().strftime("%Y-%m-%dT%H:%M:%SZ") if not timestamps.empty else ""
    last_ts = timestamps.max().strftime("%Y-%m-%dT%H:%M:%SZ") if not timestamps.empty else ""

    if "rule_id" in df.columns:
        rule_summary = (
            df.assign(
                rule_id=df["rule_id"].fillna("").astype(str),
                rule_description=df.get("rule_description", "").fillna("").astype(str),
            )
            .groupby(["rule_id", "rule_description"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "rule_id"], ascending=[False, True], kind="mergesort")
        )
    else:
        rule_summary = pd.DataFrame(columns=["rule_id", "rule_description", "count"])
    rule_summary.to_csv(rule_summary_path, index=False)

    timeline = build_timeline(df)
    timeline.to_csv(timeline_path, index=False)

    lines = [
        "# Wazuh alert export összefoglaló",
        "",
        f"- Alert count: {len(df)}",
        f"- Első timestamp: {first_ts or 'nem elérhető'}",
        f"- Utolsó timestamp: {last_ts or 'nem elérhető'}",
        "",
        "## Top rule_id-k",
        *top_lines(df.get("rule_id", pd.Series(dtype=str))),
        "",
        "## Top rule level értékek",
        *top_lines(df.get("rule_level", pd.Series(dtype=str))),
        "",
        "## Top agentek",
        *top_lines(df.get("agent_name", pd.Series(dtype=str))),
        "",
        "## Top source_ip értékek",
        *top_lines(df.get("source_ip", pd.Series(dtype=str))),
        "",
        "## Megjegyzés",
        "Az összefoglaló a megadott Wazuh alert exportból készült, nem hoz létre új mérési eseményt.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "event_count": len(df),
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "summary": summary_path,
        "rule_summary": rule_summary_path,
        "timeline": timeline_path,
        "metadata": metadata_path,
    }


def main() -> None:
    args = parse_args()
    outputs = summarize_wazuh_export(Path(args.input), Path(args.output_dir))
    for path in outputs.values():
        print(f"[OK] Kimenet: {path}")


if __name__ == "__main__":
    main()
