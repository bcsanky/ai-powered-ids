from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.repo_hygiene.common import (
    TRACKED_GENERATED_OUTPUT_PREFIXES,
    load_provenance,
    validate_provenance_payload,
)


SCAN_PATTERNS = [
    "reports/final/**/*",
    "reports/lab/**/*",
    "reports/performance/**/*",
    "reports/scored_events.jsonl",
    "reports/real_measurement_qa/preflight_*",
    "reports/live_integration/**/*",
    "results/performance/**/*",
    "figures/final/**/*",
    "examples/lab/**/*",
    "examples/scoring/**/*",
    "data/lab/**/*",
    "data/wazuh/**/*",
    "results/real_comparison/**/*",
    "results/wazuh_real/**/*",
    "results/ae_lab/**/*",
    "results/hybrid_real/**/*",
    "reports/real_measurement/**/*",
    "reports/real_measurement_qa/**/*",
    "reports/wazuh_export/**/*",
    "reports/lab_input_validation/**/*",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/repo_hygiene")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def generated_output_classification(rel_path: Path, category: str, reason: str) -> dict[str, str]:
    text = rel_path.as_posix()
    risk_level = "high" if text.startswith(("reports/final/", "figures/final/")) else "medium"
    return {
        "category": category,
        "risk_level": risk_level,
        "tracked_policy": "should_remove_from_git",
        "reason": reason,
        "recommendation": "Távolítsd el a Gitből, és futási kimenetként vagy beadási mellékletben kezeld.",
    }


def classify_path(rel_path: Path, has_valid_provenance: bool) -> dict[str, str]:
    text = rel_path.as_posix()
    name = rel_path.name
    if text.startswith("examples/lab/") or text.startswith("examples/scoring/"):
        return {
            "category": "demo_input",
            "risk_level": "medium",
            "tracked_policy": "keep_tracked",
            "reason": "Demonstrációs bemenet, nem real-lab mérési forrás.",
            "recommendation": "Tartsd demonstrációként, de ne használd real-lab eredményként.",
        }
    if text.startswith("templates/"):
        return {"category": "template", "risk_level": "low", "tracked_policy": "keep_tracked", "reason": "Kitölthető sablon.", "recommendation": "Verziókezelésben tartható."}
    if text.startswith("reports/lab/"):
        return generated_output_classification(rel_path, "demo_output", "Replay/demo kimenet.")
    if any(text.startswith(prefix) for prefix in TRACKED_GENERATED_OUTPUT_PREFIXES) or text.startswith(
        "reports/performance/"
    ):
        return generated_output_classification(
            rel_path,
            "generated_offline_result",
            "Automatikusan előállított offline vagy futási eredmény.",
        )
    if text == "reports/scored_events.jsonl":
        return generated_output_classification(rel_path, "demo_output", "Demonstrációs batch scoring kimenet.")
    if text.startswith("data/wazuh/alerts") and rel_path.suffix.lower() in {".json", ".jsonl"}:
        return {"category": "sensitive_input", "risk_level": "high", "tracked_policy": "should_ignore", "reason": "Wazuh alert export érzékeny adatot tartalmazhat.", "recommendation": "Ne commitold; szükség esetén anonimizált mellékletbe kerüljön."}
    if text in {"data/lab/lab_ground_truth.csv", "data/lab/lab_features.csv"}:
        return {"category": "raw_real_lab_input", "risk_level": "high", "tracked_policy": "should_ignore", "reason": "Valós lab mérési input.", "recommendation": "Provenance hash-sel és beadási mellékletben kezeld."}
    if text.startswith("data/lab/") or text.startswith("data/wazuh/"):
        return {"category": "raw_real_lab_input", "risk_level": "high", "tracked_policy": "should_ignore", "reason": "Lab vagy Wazuh mérési input.", "recommendation": "Ne kerüljön automatikusan Gitbe."}
    if text.startswith("results/real_comparison/") or text.startswith("results/wazuh_real/") or text.startswith("results/ae_lab/") or text.startswith("results/hybrid_real/") or text.startswith("reports/real_measurement/") or text.startswith("reports/wazuh_export/") or text.startswith("reports/lab_input_validation/"):
        if has_valid_provenance:
            return {"category": "generated_real_lab_result", "risk_level": "medium", "tracked_policy": "real_lab_only", "reason": "Provenance alapján real-lab méréshez köthető eredmény.", "recommendation": "Csak ellenőrzött mellékletként kezeld."}
        return {"category": "unknown_generated", "risk_level": "high", "tracked_policy": "should_remove_from_git", "reason": "Real-lab jellegű kimenet provenance nélkül.", "recommendation": "Készíts measurement_provenance.json fájlt tényleges mérés után."}
    if text.startswith("reports/live_integration/"):
        if has_valid_provenance:
            return {
                "category": "generated_real_lab_result",
                "risk_level": "medium",
                "tracked_policy": "real_lab_only",
                "reason": "Live integration kimenet verified real-lab provenance mellett.",
                "recommendation": "Ne commitold, hanem mérési csomagként kezeld.",
            }
        return {
            "category": "unknown_generated",
            "risk_level": "high",
            "tracked_policy": "should_remove_from_git",
            "reason": "Live integration kimenet provenance nélkül.",
            "recommendation": "Ne commitold; csak verified real-lab provenance mellett használd dolgozati demonstrációként.",
        }
    if text.startswith("reports/real_measurement_qa/"):
        return {"category": "unknown_generated", "risk_level": "medium", "tracked_policy": "should_remove_from_git", "reason": "Futtatási QA kimenet.", "recommendation": "Ne commitold; futtasd újra a mérés után."}
    if "fixture" in name:
        return {"category": "test_fixture", "risk_level": "low", "tracked_policy": "keep_tracked", "reason": "Tesztfixture.", "recommendation": "Csak tests környezetben használható."}
    return {"category": "unknown_generated", "risk_level": "medium", "tracked_policy": "should_remove_from_git", "reason": "Ismeretlen eredetű vizsgált állomány.", "recommendation": "Ellenőrizd kézzel."}


def iter_scanned_files(root: Path) -> list[Path]:
    files = []
    for pattern in SCAN_PATTERNS:
        for path in root.glob(pattern):
            if path.is_file():
                files.append(path)
    return sorted(set(files))


def audit(root: Path, output_dir: Path) -> dict[str, Path]:
    provenance = load_provenance(root / "reports/real_measurement/measurement_provenance.json")
    valid_provenance, _ = validate_provenance_payload(provenance)
    rows: list[dict[str, Any]] = []
    for path in iter_scanned_files(root):
        rel_path = path.relative_to(root)
        classification = classify_path(rel_path, valid_provenance)
        rows.append({"relative_path": rel_path.as_posix(), **classification})
    df = pd.DataFrame(
        rows,
        columns=["relative_path", "category", "risk_level", "tracked_policy", "reason", "recommendation"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "generated_artifact_audit.csv"
    md_path = output_dir / "generated_artifact_audit.md"
    json_path = output_dir / "generated_artifact_audit.json"
    df.to_csv(csv_path, index=False)
    write_markdown(df, md_path)
    json_path.write_text(json.dumps(df.to_dict(orient="records"), indent=2, ensure_ascii=False), encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path, "json": json_path}


def write_markdown(df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Repository hygiene audit",
        "",
        "| relative_path | category | risk_level | tracked_policy | reason | recommendation |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['relative_path']} | {row['category']} | {row['risk_level']} | "
            f"{row['tracked_policy']} | {row['reason']} | {row['recommendation']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    outputs = audit(Path(args.root), Path(args.output_dir))
    for path in outputs.values():
        print(f"[OK] Kimenet: {path}")


if __name__ == "__main__":
    main()
