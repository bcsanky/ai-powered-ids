from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/final/thesis_figures")
    parser.add_argument("--ae-minimal-root", default="results/final/final-ae-minimal-v1")
    parser.add_argument("--ae-context-root", default="results/final/final-ae-context-v1")
    parser.add_argument("--comparison-dir", default="results/final/comparison")
    parser.add_argument("--lab-dir", default="reports/lab")
    parser.add_argument("--performance-dir", default="reports/performance")
    return parser.parse_args()


def latest_run_with_file(root: Path, filename: str) -> Path | None:
    if not root.exists():
        return None
    candidates = [path for path in root.iterdir() if path.is_dir() and (path / filename).exists()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.name)[-1]


def copy_or_mark_missing(source: Path | None, target: Path, thesis_section: str, note: str) -> dict:
    if source is not None and source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        status = "copied"
        source_text = str(source)
        final_note = note
    else:
        status = "missing"
        source_text = "" if source is None else str(source)
        final_note = note or "A forrásábra nem található."

    return {
        "figure_file": str(target),
        "source_file": source_text,
        "thesis_section": thesis_section,
        "status": status,
        "note": final_note,
    }


def collect_figures(
    *,
    output_dir: Path,
    ae_minimal_root: Path,
    ae_context_root: Path,
    comparison_dir: Path,
    lab_dir: Path,
    performance_dir: Path | None = None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    performance_dir = performance_dir or Path("reports/performance")
    ae_minimal_run = latest_run_with_file(ae_minimal_root, "confusion_matrix.png")
    ae_context_run = latest_run_with_file(ae_context_root, "confusion_matrix.png")

    requested = [
        (
            ae_minimal_run / "confusion_matrix.png" if ae_minimal_run else None,
            output_dir / "ae_minimal_confusion_matrix.png",
            "6. fejezet",
            "AE-Minimal konfúziós mátrix.",
        ),
        (
            ae_context_run / "confusion_matrix.png" if ae_context_run else None,
            output_dir / "ae_context_confusion_matrix.png",
            "6. fejezet",
            "AE-Context konfúziós mátrix.",
        ),
        (
            comparison_dir / "fig_comparison_precision_recall_f1.png",
            output_dir / "comparison_precision_recall_f1.png",
            "6. fejezet",
            "Precision, recall és F1 összehasonlítása.",
        ),
        (
            comparison_dir / "fig_comparison_false_positive_rate.png",
            output_dir / "comparison_false_positive_rate.png",
            "6. fejezet",
            "Hamis pozitív arány összehasonlítása.",
        ),
        (
            comparison_dir / "fig_comparison_alert_count.png",
            output_dir / "comparison_alert_count.png",
            "6. fejezet",
            "Riasztásszám összehasonlítása.",
        ),
        (
            lab_dir / "lab_timeline.png",
            output_dir / "lab_timeline.png",
            "5. fejezet",
            "Lab/replay eseménysor idővonala.",
        ),
        (
            lab_dir / "risk_level_distribution.png",
            output_dir / "lab_risk_level_distribution.png",
            "5. fejezet",
            "Lab/replay kockázati szintek eloszlása.",
        ),
        (
            performance_dir / "latency_by_batch_size.png",
            output_dir / "performance_latency_by_batch_size.png",
            "6. fejezet: Teljesítménymérés",
            "Batch scoring késleltetés batch size szerint.",
        ),
        (
            performance_dir / "throughput_by_batch_size.png",
            output_dir / "performance_throughput_by_batch_size.png",
            "6. fejezet: Teljesítménymérés",
            "Batch scoring áteresztőképesség batch size szerint.",
        ),
        (
            performance_dir / "scoring_time_distribution.png",
            output_dir / "performance_scoring_time_distribution.png",
            "6. fejezet: Teljesítménymérés",
            "Batch scoring időeloszlás.",
        ),
    ]

    rows = [copy_or_mark_missing(source, target, section, note) for source, target, section, note in requested]
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "figure_manifest.csv", index=False)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = collect_figures(
        output_dir=Path(args.output_dir),
        ae_minimal_root=Path(args.ae_minimal_root),
        ae_context_root=Path(args.ae_context_root),
        comparison_dir=Path(args.comparison_dir),
        lab_dir=Path(args.lab_dir),
        performance_dir=Path(args.performance_dir),
    )
    copied = int((manifest["status"] == "copied").sum())
    missing = int((manifest["status"] == "missing").sum())
    print(f"[OK] Ábrajegyzék: {Path(args.output_dir) / 'figure_manifest.csv'}")
    print(f"[OK] Másolt ábrák: {copied}, hiányzó ábrák: {missing}")


if __name__ == "__main__":
    main()
