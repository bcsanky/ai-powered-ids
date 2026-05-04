from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from ml.src.live_smoke.common import ensure_output_dir, read_json_optional


COMMANDS = [
    ("Mérés előtti session előkészítés", "make lab-session-prep"),
    ("Szcenáriók rögzítése", "python -m ml.src.lab_session.scenario_marker_helper start ..."),
    ("Szcenáriók lezárása", "python -m ml.src.lab_session.scenario_marker_helper end ..."),
    ("Ground truth export", "python -m ml.src.lab_session.scenario_marker_helper export --output data/lab/lab_ground_truth.csv"),
    ("Wazuh alert export", "make wazuh-export-opensearch WAZUH_EXPORT_START=... WAZUH_EXPORT_END=... OPENSEARCH_PASSWORD=..."),
    ("Zeek/flow input ellenőrzés", "make lab-session-after-capture"),
    ("Mérési csomag provenance-szel", "make final-real-measurement-package-with-provenance"),
    ("Live integráció", "make final-live-integration"),
    ("Dolgozati integráció", "make final-thesis-integration"),
    ("Beadási QA", "make final-submission-check"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/live_smoke")
    return parser.parse_args()


def read_check_statuses(output_dir: Path) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for path in sorted(output_dir.glob("*_check.csv")):
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if any(row.get("status") == "FAIL" for row in rows):
            statuses[path.stem] = "FAIL"
        elif any(row.get("status") == "WARN" for row in rows):
            statuses[path.stem] = "WARN"
        elif rows:
            statuses[path.stem] = "PASS"
        else:
            statuses[path.stem] = "SKIP"
    return statuses


def run_brief(output_dir: Path) -> dict[str, Any]:
    ensure_output_dir(output_dir)
    statuses = read_check_statuses(output_dir)
    readiness = read_json_optional(output_dir / "live_smoke_readiness.json") or {}
    lines = [
        "# Operátori live környezeti brief",
        "",
        f"Readiness státusz: **{readiness.get('readiness_status', 'nincs adat')}**",
        "",
        "Ez az összefoglaló a tényleges lab mérés előtti technikai ellenőrzések alapján készült. Nem tartalmaz mérési eredményt, és nem helyettesíti a provenance fájlt.",
        "",
        "## Ellenőrzési állapotok",
        "",
        "| Terület | Státusz |",
        "|---|---|",
    ]
    for key in [
        "docker_environment_check",
        "ml_service_health_check",
        "model_artifacts_check",
        "opensearch_connection_check",
        "real_input_paths_check",
        "make_workflow_dry_run",
    ]:
        lines.append(f"| {key} | {statuses.get(key, 'nincs adat')} |")

    lines.extend(
        [
            "",
            "## Következő operátori lépések",
            "",
        ]
    )
    for title, command in COMMANDS:
        lines.append(f"- {title}: `{command}`")

    lines.extend(
        [
            "",
            "## Figyelmeztetések",
            "",
            "- A Wazuh alert exportot és a Zeek/flow inputot tényleges lab futásból kell előállítani.",
            "- A demo, sablon vagy teszt könyvtárból származó input nem használható real-lab eredményként.",
            "- A mérési eredmények csak verified real_lab provenance mellett emelhetők be végleges dolgozati eredményként.",
        ]
    )
    output_path = output_dir / "operator_smoke_brief.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"markdown": output_path, "statuses": statuses}


def main() -> None:
    args = parse_args()
    result = run_brief(Path(args.output_dir))
    print(f"[OK] Kimenet: {result['markdown']}")


if __name__ == "__main__":
    main()
