from __future__ import annotations

import argparse
from pathlib import Path

from ml.src.thesis_integration.common import (
    DEFAULT_PROVENANCE,
    compute_research_answer,
    ensure_output_dir,
    provenance_status,
    read_csv_optional,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", default="results/real_comparison/metrics_comparison.csv")
    parser.add_argument("--provenance", default=str(DEFAULT_PROVENANCE))
    parser.add_argument("--output-dir", default="reports/thesis_integration")
    return parser.parse_args()


def english_improvement(answer: dict) -> str:
    delta = answer.get("f1_delta_vs_wazuh")
    if delta is None:
        return "The F1 change compared to the Wazuh-only baseline cannot be determined from the available files."
    if delta > 0:
        return "In the controlled real-lab measurement, the best hybrid strategy showed an F1 improvement over the Wazuh-only baseline."
    return "In the controlled real-lab measurement, the metrics do not provide clear evidence of an F1 improvement over the Wazuh-only baseline."


def generate_summary_en(comparison_path: Path, provenance_path: Path, output_dir: Path) -> Path:
    output_dir = ensure_output_dir(output_dir)
    comparison = read_csv_optional(comparison_path)
    answer = compute_research_answer(comparison)
    provenance_note = (
        "The measurement inputs are tied to verified real-lab provenance."
        if provenance_status(provenance_path) == "verified_real_lab"
        else "The real-lab provenance is not verified yet, therefore this text is a draft aid rather than final measurement evidence."
    )
    body = f"""# 8. Summary

This thesis presents a laboratory prototype for AI-based cyber threat detection and analysis. The work focuses on combining Wazuh-style rule-based alerting with autoencoder-based anomaly detection in a reproducible engineering pipeline.

The implemented system separates offline CIC-IDS2017 experiments from the controlled real-lab measurement path. The real-lab path includes ground truth event windows, Wazuh alert export, lab feature extraction, AE-Minimal scoring, hybrid detection strategies, provenance checks, and report preparation.

{provenance_note} This separation is important because demonstration inputs and development outputs must not be treated as research evidence. Only verified real-lab inputs and their derived metrics can support the final thesis conclusions.

{english_improvement(answer)} The result must be interpreted within the scope of the measured lab session, not as a production-ready SOC claim or as a general industrial performance guarantee.

The main engineering contribution is the end-to-end hybrid detection pipeline and its traceability layer. Further work should include larger and more diverse lab measurements, improved feature mapping between Wazuh alerts and flow-level data, longer observation windows, and evaluation in environments closer to operational SIEM deployments.
"""
    output = output_dir / "chapter8_summary_generated.md"
    output.write_text(body, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    output = generate_summary_en(Path(args.comparison), Path(args.provenance), Path(args.output_dir))
    print(f"[OK] Kimenet: {output}")


if __name__ == "__main__":
    main()

