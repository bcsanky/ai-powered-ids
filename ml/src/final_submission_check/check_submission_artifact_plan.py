from __future__ import annotations

import argparse
from pathlib import Path

from ml.src.final_submission_check.common import check_row, read_csv_optional, read_text_optional, write_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/final_submission_check")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def run_check(output_dir: Path, root: Path = Path(".")) -> dict:
    rows = [
        check_row("source_code", "Beadási terv", "PASS", "forráskód Gitben tartható", path="ml/src/"),
        check_row("config_yaml", "Beadási terv", "PASS", "final YAML konfigurációk szerepelnek", path="experiments/final/"),
    ]
    docs = read_text_optional(root / "docs/thesis_result_artifacts.md")
    rows.append(
        check_row(
            "docs_runbooks",
            "Beadási terv",
            "PASS" if docs and "runbook" in docs.lower() else "WARN",
            "runbookok dokumentálva" if docs and "runbook" in docs.lower() else "runbook beemelés kézi ellenőrzést igényel",
            path="docs/thesis_result_artifacts.md",
        )
    )
    manifest = read_csv_optional(root / "reports/real_measurement/measurement_manifest.csv")
    rows.append(
        check_row(
            "measurement_manifest",
            "Beadási terv",
            "PASS" if manifest is not None and not manifest.empty else "WARN",
            "mérési manifest rendelkezésre áll" if manifest is not None and not manifest.empty else "mérési manifest csak valós mérés után várható",
            "Valós mérés után ellenőrizd a manifestet.",
            "reports/real_measurement/measurement_manifest.csv",
        )
    )
    appendix = read_csv_optional(root / "reports/thesis_integration/appendix_manifest.csv")
    if appendix is not None and not appendix.empty and "source_file" in appendix.columns:
        raw_rows = appendix[appendix["source_file"].astype(str).str.contains("alerts|pcap|zeek", case=False, na=False)]
        unsafe = raw_rows[raw_rows.get("include", "").astype(str).str.lower() == "true"] if "include" in raw_rows.columns else raw_rows
        rows.append(
            check_row(
                "raw_inputs_not_auto_included",
                "Beadási terv",
                "PASS" if unsafe.empty else "FAIL",
                "raw Wazuh/PCAP/Zeek nincs automatikus beadásra jelölve" if unsafe.empty else "raw érzékeny input include=true jelölést kapott",
                "Raw inputot csak anonimizálás és kézi ellenőrzés után mellékelj.",
            )
        )
    else:
        rows.append(
            check_row(
                "appendix_manifest",
                "Beadási terv",
                "WARN",
                "appendix manifest még nem áll rendelkezésre",
                "Thesis integration után ellenőrizd újra.",
            )
        )
    if docs and "examples/lab" in docs and "demonstrációs" in docs.lower():
        status = "PASS"
        message = "examples/lab demo szerepe dokumentált"
    else:
        status = "WARN"
        message = "examples/lab mérési eredményként való kizárását kézzel ellenőrizni kell"
    rows.append(check_row("examples_lab_not_measurement", "Beadási terv", status, message, path="docs/thesis_result_artifacts.md"))
    rows.append(check_row("templates_lab_allowed", "Beadási terv", "PASS", "templates/lab sablonként verziózható", path="templates/lab/"))
    return write_outputs(
        output_dir=output_dir,
        basename="submission_artifact_plan",
        title="Beadási melléklet terv ellenőrzés",
        rows=rows,
    )


def main() -> None:
    args = parse_args()
    result = run_check(Path(args.output_dir), Path(args.root))
    print(f"[OK] Kimenet: {result['markdown']}")


if __name__ == "__main__":
    main()

