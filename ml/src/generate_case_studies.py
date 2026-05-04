from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


RISK_LEVELS = ["normal", "medium", "high", "critical"]
SCENARIO_DESCRIPTIONS = {
    "benign_activity": "Normál jellegű, mérsékelt forgalmi intenzitású események.",
    "port_scan": "Több célportot érintő, rövid időtartamú és magasabb csomagrátájú események.",
    "ssh_bruteforce": "Ismétlődő SSH kapcsolati kísérleteket leíró események.",
    "combined_suspicious": "Szabályalapú jelzéssel és magasabb forgalmi intenzitással jellemzett események.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-events", default="reports/lab/lab_scored_events.jsonl")
    parser.add_argument("--output-dir", default="reports/lab")
    return parser.parse_args()


def read_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó pontozott eseményfájl: {path}")
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("A pontozott események támogatott formátumai: .jsonl, .csv")


def validate_events(df: pd.DataFrame) -> None:
    required = {"event_id", "scenario", "risk_level", "anomaly_score", "ml_alert", "rule_flag"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Hiányzó kötelező oszlopok a case study bemenetben: {missing}")


def normalize_events(df: pd.DataFrame) -> pd.DataFrame:
    validate_events(df)
    out = df.copy()
    out["anomaly_score"] = pd.to_numeric(out["anomaly_score"], errors="coerce").fillna(0.0)
    out["ml_alert"] = out["ml_alert"].astype(str).str.lower().isin({"true", "1", "yes", "igen"})
    out["rule_flag"] = out["rule_flag"].astype(str).str.lower().isin({"true", "1", "yes", "igen"})
    out["risk_level"] = out["risk_level"].fillna("normal").astype(str)
    out["scenario"] = out["scenario"].fillna("unknown").astype(str)
    if "timestamp" in out.columns:
        out["timestamp"] = out["timestamp"].astype(str)
    else:
        out["timestamp"] = [str(i) for i in range(len(out))]
    return out


def build_scenario_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, group in df.groupby("scenario", sort=True):
        risk_counts = group["risk_level"].value_counts().to_dict()
        rows.append(
            {
                "scenario": scenario,
                "event_count": int(len(group)),
                "normal_count": int(risk_counts.get("normal", 0)),
                "medium_count": int(risk_counts.get("medium", 0)),
                "high_count": int(risk_counts.get("high", 0)),
                "critical_count": int(risk_counts.get("critical", 0)),
                "avg_anomaly_score": float(group["anomaly_score"].mean()),
                "max_anomaly_score": float(group["anomaly_score"].max()),
                "ml_alert_count": int(group["ml_alert"].sum()),
                "rule_alert_count": int(group["rule_flag"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_risk_matrix(df: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.crosstab(df["scenario"], df["risk_level"])
    for level in RISK_LEVELS:
        if level not in matrix.columns:
            matrix[level] = 0
    return matrix[RISK_LEVELS].reset_index()


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Nincs megjeleníthető adat._"
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def risk_count_table(risk_counts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "risk_level": list(risk_counts.index),
            "event_count": [int(value) for value in risk_counts.values],
        }
    )


def write_summary(df: pd.DataFrame, scenario_summary: pd.DataFrame, risk_matrix: pd.DataFrame, path: Path) -> None:
    risk_counts = df["risk_level"].value_counts().reindex(RISK_LEVELS, fill_value=0)
    top_events = df.sort_values("anomaly_score", ascending=False).head(5)
    prioritized = int(df["risk_level"].isin(["medium", "high", "critical"]).sum())

    lines = [
        "# Lab/replay esettanulmány összefoglaló",
        "",
        f"Összes esemény száma: {len(df)}",
        "",
        "## Események szcenáriónként",
        "",
        markdown_table(scenario_summary),
        "",
        "## Kockázati szintek eloszlása",
        "",
        markdown_table(risk_count_table(risk_counts)),
        "",
        "## Legmagasabb anomáliapontszámú események",
        "",
        markdown_table(top_events[[c for c in ["event_id", "scenario", "risk_level", "anomaly_score", "reason"] if c in top_events.columns]]),
        "",
        f"Közepes, magas vagy kritikus kockázati szintet kapott események száma: {prioritized}",
        "",
        "## Szcenárió-kockázat mátrix",
        "",
        markdown_table(risk_matrix),
        "",
        "## Szakmai értelmezés",
        "",
        "A replay-alapú demonstráció célja annak bemutatása, hogy a pontozási lánc eseményszinten képes kockázati prioritást rendelni normál és gyanúsabb mintázatokhoz. Az eredmények a validált AE-Minimal modell és az egyszerű szabályjelzések kombinált értelmezését szemléltetik.",
        "",
        "## Korlát",
        "",
        "Ez kontrollált replay-alapú demonstráció, nem éles SOC mérés és nem natív Wazuh teljesítménymérés. A kis elemszámú eseménysor esettanulmányos bemutatásra alkalmas, általános teljesítménykövetkeztetésre nem.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_scenario_case_study(df: pd.DataFrame, scenario: str, path: Path) -> None:
    group = df.loc[df["scenario"] == scenario].copy()
    risk_counts = group["risk_level"].value_counts().reindex(RISK_LEVELS, fill_value=0)
    top = group.sort_values("anomaly_score", ascending=False).head(5)

    expected_notes = ""
    if "expected_behavior" in group.columns:
        expected_notes = (
            "\n## Elvárt viselkedés és megfigyelések\n\n"
            + markdown_table(
                group[
                    [
                        c
                        for c in ["event_id", "expected_behavior", "risk_level", "ml_alert", "rule_flag"]
                        if c in group.columns
                    ]
                ]
            )
            + "\n"
        )

    if scenario == "benign_activity":
        elevated = int(group["risk_level"].isin(["medium", "high", "critical"]).sum())
        fp_fn_note = (
            f"A benign_activity szcenárióban {elevated} esemény kapott közepes vagy magasabb "
            "prioritást. Ez replay elváráshoz viszonyított false positive jellegű megfigyelésként "
            "kezelhető, nem formális mérési címkeként."
        )
    else:
        low = int(group["risk_level"].isin(["normal"]).sum())
        fp_fn_note = (
            f"A {scenario} szcenárióban {low} esemény kapott normál prioritást. Ezek replay "
            "elváráshoz viszonyítva false negative jellegű megfigyelésként lennének értelmezhetők. "
            "A jelen kimenet nem formális ground truth mérés."
        )

    lines = [
        f"# Esettanulmány: {scenario}",
        "",
        "## Szcenárió leírása",
        "",
        SCENARIO_DESCRIPTIONS.get(scenario, "Kontrollált replay-alapú demonstrációs szcenárió."),
        "",
        "## Bemeneti események jellemzői",
        "",
        markdown_table(group[[c for c in ["event_id", "timestamp", "description", "destination_port", "flow_packets_per_sec", "rule_flag", "rule_level"] if c in group.columns]]),
        "",
        "## Scoring eredmények",
        "",
        markdown_table(group[[c for c in ["event_id", "anomaly_score", "threshold_name", "ml_alert", "rule_flag", "risk_level", "reason"] if c in group.columns]]),
        "",
        "## Kockázati szintek",
        "",
        markdown_table(risk_count_table(risk_counts)),
        "",
        "## False positive / false negative jellegű megfigyelések",
        "",
        fp_fn_note,
        "",
        "## Legfontosabb események",
        "",
        markdown_table(top[[c for c in ["event_id", "description", "anomaly_score", "risk_level"] if c in top.columns]]),
        expected_notes,
        "## Dolgozatba emelhető rövid összefoglaló",
        "",
        "A szcenárió azt szemlélteti, hogy a prototípus a bemeneti flow jellemzők és a szabályalapú jelzések alapján eseményszintű prioritást rendel a kontrollált mintákhoz. A megfigyelések demonstrációs jellegűek, és nem helyettesítik a nagy elemszámú mérési benchmarkot.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def save_timeline(df: pd.DataFrame, path: Path) -> None:
    ordered = df.copy()
    ordered["_time"] = pd.to_datetime(ordered["timestamp"], errors="coerce")
    if ordered["_time"].notna().any():
        ordered = ordered.sort_values("_time")
        xlabel = "Időpont"
    else:
        ordered = ordered.reset_index(drop=True)
        xlabel = "Eseménysorrend"

    scenarios = sorted(ordered["scenario"].unique())
    scenario_to_y = {name: idx for idx, name in enumerate(scenarios)}
    colors = {"normal": "#4c78a8", "medium": "#f2c14e", "high": "#f58518", "critical": "#c73e1d"}

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for level in RISK_LEVELS:
        subset = ordered.loc[ordered["risk_level"] == level]
        if subset.empty:
            continue
        xs = subset["_time"] if ordered["_time"].notna().any() else subset.index
        ys = subset["scenario"].map(scenario_to_y)
        ax.scatter(xs, ys, label=level, color=colors[level], s=70, alpha=0.85)

    ax.set_title("Lab/replay események idővonala")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Szcenárió")
    ax.set_yticks(list(scenario_to_y.values()))
    ax.set_yticklabels(list(scenario_to_y.keys()))
    ax.legend(title="Kockázati szint")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_risk_distribution(df: pd.DataFrame, path: Path) -> None:
    counts = df["risk_level"].value_counts().reindex(RISK_LEVELS, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(counts.index, counts.values, color=["#4c78a8", "#f2c14e", "#f58518", "#c73e1d"])
    ax.set_title("Kockázati szintek eloszlása")
    ax.set_xlabel("Kockázati szint")
    ax.set_ylabel("Darabszám")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_case_studies(scored_events_path: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = normalize_events(read_events(scored_events_path))

    scenario_summary = build_scenario_summary(df)
    risk_matrix = build_risk_matrix(df)
    scenario_summary_path = output_dir / "scenario_summary.csv"
    risk_matrix_path = output_dir / "scenario_risk_matrix.csv"
    scenario_summary.to_csv(scenario_summary_path, index=False)
    risk_matrix.to_csv(risk_matrix_path, index=False)

    summary_path = output_dir / "case_study_summary.md"
    write_summary(df, scenario_summary, risk_matrix, summary_path)

    case_paths = {}
    for scenario in sorted(df["scenario"].unique()):
        path = output_dir / f"case_study_{scenario}.md"
        write_scenario_case_study(df, scenario, path)
        case_paths[scenario] = path

    timeline_path = output_dir / "lab_timeline.png"
    risk_distribution_path = output_dir / "risk_level_distribution.png"
    save_timeline(df, timeline_path)
    save_risk_distribution(df, risk_distribution_path)

    outputs = {
        "summary": summary_path,
        "scenario_summary": scenario_summary_path,
        "risk_matrix": risk_matrix_path,
        "timeline": timeline_path,
        "risk_distribution": risk_distribution_path,
    }
    outputs.update({f"case_{scenario}": path for scenario, path in case_paths.items()})
    return outputs


def main() -> None:
    args = parse_args()
    outputs = generate_case_studies(
        scored_events_path=Path(args.scored_events),
        output_dir=Path(args.output_dir),
    )
    print(f"[OK] Case study összefoglaló: {outputs['summary']}")
    print(f"[OK] Scenario summary: {outputs['scenario_summary']}")
    print(f"[OK] Timeline ábra: {outputs['timeline']}")
    print(f"[OK] Kockázati eloszlás ábra: {outputs['risk_distribution']}")


if __name__ == "__main__":
    main()
