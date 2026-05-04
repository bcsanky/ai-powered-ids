from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.lab_session.common import (
    assert_real_lab_path,
    check_row,
    has_fail,
    now_iso,
    overall_status,
    write_check_report,
    write_metadata,
    write_summary_csv,
)
from ml.src.wazuh_baseline.build_ground_truth import validate_ground_truth
from ml.src.wazuh_baseline.parse_wazuh_alerts import load_alert_objects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--wazuh-alerts", required=True)
    parser.add_argument("--zeek-conn", required=True)
    parser.add_argument("--flow-csv", required=True)
    parser.add_argument("--output-dir", default="reports/lab_session")
    return parser.parse_args()


def count_wazuh_alerts(path: Path) -> int:
    if path.suffix.lower() == ".csv":
        return len(pd.read_csv(path))
    return len(load_alert_objects(path))


def validate_real_path(rows: list[dict[str, str]], path: Path, check_id: str) -> bool:
    try:
        assert_real_lab_path(path, check_id)
        rows.append(check_row(check_id, "path", "PASS", f"Útvonal elfogadható: {path}"))
        return True
    except ValueError as exc:
        rows.append(check_row(check_id, "path", "FAIL", str(exc), "Használj tényleges data/lab vagy data/wazuh inputot."))
        return False


def run_post_session_input_check(
    *,
    ground_truth_path: Path,
    wazuh_alerts_path: Path,
    zeek_conn_path: Path,
    flow_csv_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    validate_real_path(rows, ground_truth_path, "ground_truth_path")
    validate_real_path(rows, wazuh_alerts_path, "wazuh_alerts_path")
    validate_real_path(rows, zeek_conn_path, "zeek_conn_path")
    validate_real_path(rows, flow_csv_path, "flow_csv_path")

    try:
        ground_truth = validate_ground_truth(ground_truth_path)
        rows.append(check_row("ground_truth_schema", "input", "PASS", f"{len(ground_truth)} ground truth esemény."))
        label_counts = ground_truth["label"].value_counts().to_dict()
        if label_counts.get("benign", 0) >= 1 and label_counts.get("attack", 0) >= 1:
            rows.append(check_row("label_coverage", "input", "PASS", json.dumps(label_counts, ensure_ascii=False)))
        else:
            rows.append(
                check_row(
                    "label_coverage",
                    "input",
                    "FAIL",
                    "Legalább 1 benign és 1 attack esemény szükséges.",
                    "Rögzíts mindkét típusból valós lab eseményt.",
                )
            )
        scenario_count = int(ground_truth["scenario"].nunique())
        if scenario_count >= 2:
            rows.append(check_row("scenario_coverage", "input", "PASS", f"{scenario_count} scenario."))
        else:
            rows.append(
                check_row(
                    "scenario_coverage",
                    "input",
                    "FAIL",
                    "Legalább 2 különböző scenario szükséges.",
                    "Bővítsd a mérési futást több szcenárióval.",
                )
            )
    except Exception as exc:  # noqa: BLE001
        rows.append(
            check_row(
                "ground_truth_schema",
                "input",
                "FAIL",
                f"Ground truth nem validálható: {exc}",
                "Ellenőrizd a scenario marker exportot.",
            )
        )
        ground_truth = pd.DataFrame()

    try:
        alert_count = count_wazuh_alerts(wazuh_alerts_path)
        if alert_count > 0:
            rows.append(check_row("wazuh_alerts", "input", "PASS", f"{alert_count} Wazuh alert található."))
        else:
            rows.append(
                check_row(
                    "wazuh_alerts",
                    "input",
                    "FAIL",
                    "A Wazuh alert export üres.",
                    "Exportáld újra a lab időablakra szűrt alert eseményeket.",
                )
            )
    except Exception as exc:  # noqa: BLE001
        rows.append(
            check_row(
                "wazuh_alerts",
                "input",
                "FAIL",
                f"Wazuh alert input nem olvasható: {exc}",
                "Ellenőrizd a Wazuh export útvonalát és formátumát.",
            )
        )

    zeek_exists = zeek_conn_path.exists()
    flow_exists = flow_csv_path.exists()
    if zeek_exists or flow_exists:
        rows.append(
            check_row(
                "flow_source",
                "input",
                "PASS",
                f"Flow forrás elérhető: {'Zeek conn.log' if zeek_exists else ''} {'flow CSV' if flow_exists else ''}".strip(),
            )
        )
    else:
        rows.append(
            check_row(
                "flow_source",
                "input",
                "FAIL",
                "Sem Zeek conn.log, sem flow CSV nem található.",
                "Állíts elő Zeek vagy flow inputot a feature buildhez.",
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "post_session_input_check.md"
    summary_path = output_dir / "post_session_input_check.csv"
    metadata_path = output_dir / "post_session_input_metadata.json"
    write_check_report(
        rows,
        report_path,
        title="Post-session input ellenőrzés",
        intro="Ez a riport a tényleges mérés után létrejött real-lab inputok meglétét és alapkonzisztenciáját ellenőrzi.",
    )
    write_summary_csv(rows, summary_path)
    write_metadata(
        metadata_path,
        {
            "created_at": now_iso(),
            "status": overall_status(rows),
            "ground_truth": str(ground_truth_path),
            "wazuh_alerts": str(wazuh_alerts_path),
            "zeek_conn": str(zeek_conn_path),
            "flow_csv": str(flow_csv_path),
            "n_ground_truth_events": len(ground_truth),
        },
    )
    return {"status": overall_status(rows), "rows": rows, "report": report_path, "summary": summary_path, "metadata": metadata_path}


def main() -> None:
    args = parse_args()
    result = run_post_session_input_check(
        ground_truth_path=Path(args.ground_truth),
        wazuh_alerts_path=Path(args.wazuh_alerts),
        zeek_conn_path=Path(args.zeek_conn),
        flow_csv_path=Path(args.flow_csv),
        output_dir=Path(args.output_dir),
    )
    print(f"[OK] Kimenet: {result['report']}")
    print(f"[OK] Kimenet: {result['summary']}")
    print(f"[OK] Kimenet: {result['metadata']}")
    if has_fail(result["rows"]):
        sys.exit(1)


if __name__ == "__main__":
    main()

