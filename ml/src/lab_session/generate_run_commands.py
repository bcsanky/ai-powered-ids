from __future__ import annotations

import argparse
from pathlib import Path


COMMANDS_BY_SOURCE = {
    "zeek": [
        ("make real-measurement-preflight", "Mérés előtti alapfeltételek ellenőrzése."),
        ("make lab-build-features-zeek", "Lab feature állomány előállítása Zeek conn.log alapján."),
        ("make lab-validate-real-inputs", "Ground truth, feature és Wazuh alert input konzisztencia-ellenőrzése."),
        (
            "make final-real-measurement-package-with-provenance",
            "Wazuh-only, AE-only, hibrid eredmények, provenance és manifest előállítása.",
        ),
        ("make final-live-integration", "Wazuh alert -> ML scoring -> hibrid prioritás integrációs kimenet."),
        ("make final-real-measurement-thesis-ready", "QA táblázatok és védési segédletek frissítése."),
        ("make repo-hygiene-check", "No-demo és tracked-output guard futtatása."),
        ("make final-validate", "Teljes validáció futtatása."),
    ],
    "flow-csv": [
        ("make real-measurement-preflight", "Mérés előtti alapfeltételek ellenőrzése."),
        ("make lab-build-features-flow-csv", "Lab feature állomány előállítása általános flow CSV alapján."),
        ("make lab-validate-real-inputs", "Ground truth, feature és Wazuh alert input konzisztencia-ellenőrzése."),
        (
            "make final-real-measurement-package-with-provenance",
            "Wazuh-only, AE-only, hibrid eredmények, provenance és manifest előállítása.",
        ),
        ("make final-live-integration", "Wazuh alert -> ML scoring -> hibrid prioritás integrációs kimenet."),
        ("make final-real-measurement-thesis-ready", "QA táblázatok és védési segédletek frissítése."),
        ("make repo-hygiene-check", "No-demo és tracked-output guard futtatása."),
        ("make final-validate", "Teljes validáció futtatása."),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-source", choices=sorted(COMMANDS_BY_SOURCE), required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def build_command_markdown(feature_source: str) -> str:
    lines = [
        "# Lab session futtatási parancslista",
        "",
        "Ez a dokumentum csak operátori parancslista. A parancsokat nem futtatja le, és nem hoz létre mérési inputot vagy metrikát.",
        "",
        f"Feature forrás: `{feature_source}`",
        "",
        "| Lépés | Parancs | Cél |",
        "| ---: | --- | --- |",
    ]
    for index, (command, description) in enumerate(COMMANDS_BY_SOURCE[feature_source], start=1):
        lines.append(f"| {index} | `{command}` | {description} |")
    lines.extend(
        [
            "",
            "A lista futtatása előtt ellenőrizni kell, hogy a ground truth, Wazuh alert export és Zeek/flow input tényleges lab futásból származik.",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_run_commands(feature_source: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_command_markdown(feature_source), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    output = generate_run_commands(args.feature_source, Path(args.output))
    print(f"[OK] Kimenet: {output}")


if __name__ == "__main__":
    main()

