from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ml.src.lab_session.common import now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/lab_session")
    parser.add_argument("--session-plan", default="reports/lab_session/session_plan.yaml")
    parser.add_argument("--post-input-check", default="reports/lab_session/post_session_input_check.csv")
    parser.add_argument("--provenance", default="reports/real_measurement/measurement_provenance.json")
    parser.add_argument("--thesis-readiness", default="reports/real_measurement_qa/thesis_readiness.md")
    parser.add_argument("--metrics-comparison", default="results/real_comparison/metrics_comparison.csv")
    return parser.parse_args()


def read_session_id(path: Path) -> str:
    if not path.exists():
        return "nincs session_plan.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return str(data.get("session_id", "nincs session_id"))


def input_check_status(path: Path) -> str:
    if not path.exists():
        return "hiányzik"
    df = pd.read_csv(path)
    if df.empty:
        return "üres"
    if "status" in df.columns and (df["status"] == "FAIL").any():
        return "FAIL"
    if "status" in df.columns and (df["status"] == "WARN").any():
        return "WARN"
    return "PASS"


def metrics_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "row_count": 0, "configurations": []}
    df = pd.read_csv(path)
    configs = []
    if "configuration" in df.columns:
        configs = [str(value) for value in df["configuration"].dropna().tolist()]
    return {"exists": True, "row_count": int(len(df)), "configurations": configs}


def read_text_excerpt(path: Path) -> str:
    if not path.exists():
        return "hiányzik"
    text = path.read_text(encoding="utf-8").strip()
    return text.splitlines()[0] if text else "üres"


def build_summary_payload(
    *,
    session_plan_path: Path,
    post_input_check_path: Path,
    provenance_path: Path,
    thesis_readiness_path: Path,
    metrics_comparison_path: Path,
) -> dict[str, Any]:
    metrics = metrics_status(metrics_comparison_path)
    return {
        "created_at": now_iso(),
        "session_id": read_session_id(session_plan_path),
        "session_plan_exists": session_plan_path.exists(),
        "post_input_check_status": input_check_status(post_input_check_path),
        "provenance_exists": provenance_path.exists(),
        "thesis_readiness": read_text_excerpt(thesis_readiness_path),
        "metrics_comparison_exists": metrics["exists"],
        "metrics_comparison_rows": metrics["row_count"],
        "metrics_configurations": metrics["configurations"],
        "pipeline_has_results": bool(metrics["exists"] and metrics["row_count"] > 0),
    }


def markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Lab session összefoglaló",
        "",
        "Ez az összefoglaló csak meglévő inputokat és kimeneteket listáz. Nem hoz létre metrikát és nem talál ki eredményt.",
        "",
        f"- Session ID: `{payload['session_id']}`",
        f"- Session plan létezik: `{payload['session_plan_exists']}`",
        f"- Post-session input check státusz: `{payload['post_input_check_status']}`",
        f"- Provenance létezik: `{payload['provenance_exists']}`",
        f"- Thesis readiness: `{payload['thesis_readiness']}`",
        f"- Metrics comparison létezik: `{payload['metrics_comparison_exists']}`",
        f"- Metrics comparison sorok: `{payload['metrics_comparison_rows']}`",
        "",
        "## Konfigurációk",
        "",
    ]
    if payload["metrics_configurations"]:
        lines.extend(f"- `{config}`" for config in payload["metrics_configurations"])
    else:
        lines.append("Nincs metrics_comparison.csv, ezért konfigurációs eredmény nem listázható.")
    lines.extend(
        [
            "",
            "## Következő teendők",
            "",
        ]
    )
    if not payload["provenance_exists"]:
        lines.append("- Futtasd a `make final-real-measurement-package-with-provenance` célt tényleges inputokkal.")
    if not payload["pipeline_has_results"]:
        lines.append("- Futtasd a real-lab mérési pipeline-t, majd ellenőrizd a `results/real_comparison/metrics_comparison.csv` fájlt.")
    if payload["provenance_exists"] and payload["pipeline_has_results"]:
        lines.append("- A mérési csomag rendelkezésre áll; ellenőrizd a QA és anonimizálási kimeneteket.")
    return "\n".join(lines) + "\n"


def generate_session_summary(
    *,
    output_dir: Path,
    session_plan_path: Path,
    post_input_check_path: Path,
    provenance_path: Path,
    thesis_readiness_path: Path,
    metrics_comparison_path: Path,
) -> dict[str, Path]:
    payload = build_summary_payload(
        session_plan_path=session_plan_path,
        post_input_check_path=post_input_check_path,
        provenance_path=provenance_path,
        thesis_readiness_path=thesis_readiness_path,
        metrics_comparison_path=metrics_comparison_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "session_summary.md"
    json_path = output_dir / "session_summary.json"
    md_path.write_text(markdown_summary(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"markdown": md_path, "json": json_path}


def main() -> None:
    args = parse_args()
    outputs = generate_session_summary(
        output_dir=Path(args.output_dir),
        session_plan_path=Path(args.session_plan),
        post_input_check_path=Path(args.post_input_check),
        provenance_path=Path(args.provenance),
        thesis_readiness_path=Path(args.thesis_readiness),
        metrics_comparison_path=Path(args.metrics_comparison),
    )
    for path in outputs.values():
        print(f"[OK] Kimenet: {path}")


if __name__ == "__main__":
    main()

