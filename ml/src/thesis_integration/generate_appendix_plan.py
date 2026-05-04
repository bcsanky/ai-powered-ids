from __future__ import annotations

import argparse
from pathlib import Path

from ml.src.thesis_integration.common import ensure_output_dir, provenance_status, write_rows_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/thesis_integration")
    parser.add_argument("--provenance", default="reports/real_measurement/measurement_provenance.json")
    return parser.parse_args()


def appendix_rows(provenance_path: Path) -> list[dict[str, str]]:
    verified = provenance_status(provenance_path) == "verified_real_lab"
    return [
        {
            "appendix_item": "Mérési manifest",
            "source_file": "reports/real_measurement/measurement_manifest.md",
            "include": "true" if verified else "with_limitations",
            "note": "Hash-ekkel ellátott beadási lista.",
        },
        {
            "appendix_item": "Provenance kivonat",
            "source_file": "reports/real_measurement/measurement_provenance.json",
            "include": "true" if verified else "false",
            "note": "Csak érzékeny mezők ellenőrzése után.",
        },
        {
            "appendix_item": "Makefile célok listája",
            "source_file": "Makefile",
            "include": "true",
            "note": "Reprodukálhatósági mellékletként hivatkozható.",
        },
        {
            "appendix_item": "Final konfigurációs YAML fájlok",
            "source_file": "experiments/final/",
            "include": "true",
            "note": "Konfigurációs melléklet.",
        },
        {
            "appendix_item": "Lab session runbook kivonata",
            "source_file": "docs/lab_session_orchestration_runbook.md",
            "include": "true",
            "note": "Mérési eljárás bemutatására.",
        },
        {
            "appendix_item": "Real measurement checklist",
            "source_file": "docs/final_real_measurement_checklist.md",
            "include": "true",
            "note": "Operátori ellenőrzőlista.",
        },
        {
            "appendix_item": "Raw Wazuh alert export",
            "source_file": "data/wazuh/alerts.jsonl",
            "include": "with_redaction",
            "note": "Érzékeny lehet, automatikus beadásra nem javasolt.",
        },
        {
            "appendix_item": "PCAP vagy Zeek raw állomány",
            "source_file": "data/lab/zeek/ vagy data/lab/*.pcap",
            "include": "with_redaction",
            "note": "Nagy és érzékeny lehet, külön mérlegelést igényel.",
        },
    ]


def generate_appendix_plan(output_dir: Path, provenance_path: Path) -> dict[str, Path]:
    output_dir = ensure_output_dir(output_dir)
    rows = appendix_rows(provenance_path)
    lines = [
        "# Mellékletterv",
        "",
        "| Melléklet | Forrás | Beemelés | Megjegyzés |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['appendix_item']} | {row['source_file']} | {row['include']} | {row['note']} |")
    outputs = {
        "md": output_dir / "appendix_plan.md",
        "csv": output_dir / "appendix_manifest.csv",
    }
    outputs["md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_rows_csv(outputs["csv"], rows, ["appendix_item", "source_file", "include", "note"])
    return outputs


def main() -> None:
    args = parse_args()
    outputs = generate_appendix_plan(Path(args.output_dir), Path(args.provenance))
    for path in outputs.values():
        print(f"[OK] Kimenet: {path}")


if __name__ == "__main__":
    main()

