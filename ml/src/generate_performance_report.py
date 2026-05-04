from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="reports/performance/benchmark_results.csv")
    parser.add_argument("--system-info", default="reports/performance/system_info.json")
    parser.add_argument("--output-dir", default="reports/performance")
    return parser.parse_args()


def load_benchmark(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó benchmark CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("A benchmark CSV üres.")
    return df


def load_system_info(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, f"A system_info.json nem található: {path}"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f), None


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


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["total_events", "batch_size"], as_index=False)
        .agg(
            events_per_second=("events_per_second", "mean"),
            avg_latency_ms=("avg_latency_ms", "mean"),
            p50_latency_ms=("p50_latency_ms", "mean"),
            p95_latency_ms=("p95_latency_ms", "mean"),
            p99_latency_ms=("p99_latency_ms", "mean"),
            failed_events=("failed_events", "sum"),
        )
        .sort_values(["total_events", "batch_size"])
    )
    return grouped


def build_report(df: pd.DataFrame, system_info: dict[str, Any], warning: str | None) -> str:
    summary = aggregate_results(df)
    best_throughput = summary.sort_values("events_per_second", ascending=False).iloc[0]
    best_latency = summary.sort_values("p95_latency_ms", ascending=True).iloc[0]
    failed_total = int(df["failed_events"].sum())
    event_counts = ", ".join(str(int(v)) for v in sorted(df["total_events"].unique()))
    batch_sizes = ", ".join(str(int(v)) for v in sorted(df["batch_size"].unique()))

    system_lines = []
    if warning:
        system_lines.append(f"- Figyelmeztetés: {warning}")
    for key in ["timestamp", "python_version", "platform", "processor", "cpu_count", "input_file", "preprocess_file"]:
        if key in system_info:
            system_lines.append(f"- {key}: `{system_info[key]}`")
    if not system_lines:
        system_lines.append("- A rendszerinformáció nem áll rendelkezésre.")

    lines = [
        "# Batch scoring teljesítményriport",
        "",
        "## Mérés célja",
        "",
        "A mérés célja a meglévő AE-Minimal batch scoring feldolgozási lánc lokális/labor sebességének, késleltetésének és áteresztőképességének dokumentált vizsgálata.",
        "",
        "## Bemenet és modell",
        "",
        f"- Bemeneti eseménykészlet: `{system_info.get('input_file', 'nincs rögzítve')}`",
        "- Modell: AE-Minimal végleges modellkimenet.",
        f"- Preprocess fájl: `{system_info.get('preprocess_file', 'nincs rögzítve')}`",
        f"- Eseményszámok: {event_counts}",
        f"- Batch size értékek: {batch_sizes}",
        "",
        "## Rendszerinformáció",
        "",
        "\n".join(system_lines),
        "",
        "## Fő eredmények",
        "",
        f"- Legjobb áteresztőképesség: {best_throughput['events_per_second']:.2f} esemény/másodperc, batch size {int(best_throughput['batch_size'])}, eseményszám {int(best_throughput['total_events'])}.",
        f"- Legalacsonyabb p95 késleltetés: {best_latency['p95_latency_ms']:.4f} ms, batch size {int(best_latency['batch_size'])}, eseményszám {int(best_latency['total_events'])}.",
        f"- Hibás események összesen: {failed_total}.",
        "",
        "## Összesített táblázat",
        "",
        markdown_table(summary),
        "",
        "## Rövid értelmezés",
        "",
        "A batch size hatása a lokális mérésben az áteresztőképesség és a késleltetés változásán keresztül értelmezhető. Az eseményszám növelése determinisztikus ismétléssel történt, ezért a mérés a scoring feldolgozási költségét, nem pedig új adatminták detektálási minőségét vizsgálja.",
        "",
        "## Dolgozatba emelhető összefoglaló",
        "",
        "A prototípus batch scoring komponensén lokális/labor teljesítménymérés készült több eseményszám és batch size beállítás mellett. A mérés az esemény/másodperc alapú áteresztőképességet, az átlagos és percentilis késleltetést, valamint a hibás események számát rögzíti.",
        "",
        "## Korlátok",
        "",
        "- lokális/labor mérés.",
        "- Hardver- és környezetfüggő eredmények.",
        "- Nem hosszú idejű éles üzemi terhelés.",
        "- Nem natív Wazuh indexelési teljesítmény.",
        "- Batch scoring benchmark, nem teljes SIEM end-to-end benchmark.",
        "",
    ]
    return "\n".join(lines)


def markdown_to_html(markdown_text: str) -> str:
    escaped = html.escape(markdown_text)
    body = escaped.replace("\n", "<br>\n")
    return (
        "<!doctype html>\n"
        "<html lang=\"hu\">\n"
        "<head><meta charset=\"utf-8\"><title>Batch scoring teljesítményriport</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:2rem auto;line-height:1.5;}"
        "code{background:#f2f2f2;padding:0.1rem 0.25rem;}</style></head>\n"
        f"<body>{body}</body>\n"
        "</html>\n"
    )


def write_performance_report(benchmark_path: Path, system_info_path: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_benchmark(benchmark_path)
    system_info, warning = load_system_info(system_info_path)
    markdown = build_report(df, system_info, warning)

    md_path = output_dir / "performance_report.md"
    html_path = output_dir / "performance_report.html"
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(markdown_to_html(markdown), encoding="utf-8")
    return {"markdown": md_path, "html": html_path}


def main() -> None:
    args = parse_args()
    outputs = write_performance_report(
        benchmark_path=Path(args.benchmark),
        system_info_path=Path(args.system_info),
        output_dir=Path(args.output_dir),
    )
    print(f"[OK] Teljesítményriport: {outputs['markdown']}")
    print(f"[OK] HTML riport: {outputs['html']}")


if __name__ == "__main__":
    main()
