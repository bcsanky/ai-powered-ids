from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from ml.src.final_acceptance.common import check_row, has_fail, write_check_outputs
from ml.src.lab_features.validate_real_lab_inputs import validate_real_lab_inputs
from ml.src.lab_session.post_session_input_check import run_post_session_input_check
from ml.src.real_measurement_qa.postrun_quality_gate import run_quality_gate
from ml.src.repo_hygiene.create_measurement_provenance import validate_input_path
from ml.src.thesis_integration.check_thesis_inputs import run_check as run_thesis_input_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/final_acceptance")
    return parser.parse_args()


def write_minimal_ground_truth(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "event_id,timestamp_start,timestamp_end,scenario,label,attack_type,source_ip,target_ip\n"
        "e1,2026-05-04T10:00:00Z,2026-05-04T10:01:00Z,benign_ssh_login,benign,none,10.0.0.1,10.0.0.2\n"
        "e2,2026-05-04T10:02:00Z,2026-05-04T10:03:00Z,port_scan,attack,port_scan,10.0.0.3,10.0.0.2\n",
        encoding="utf-8",
    )


def write_minimal_features(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "event_id,destination_port,flow_duration,total_fwd_packets,total_backward_packets,flow_bytes_per_sec,flow_packets_per_sec,protocol\n"
        "e1,22,1.0,2,2,100.0,4.0,tcp\n"
        "e2,80,1.0,4,1,200.0,5.0,tcp\n",
        encoding="utf-8",
    )


def check_thesis_without_provenance(tmp_root: Path) -> dict[str, str]:
    result = run_thesis_input_check(tmp_root, tmp_root / "reports/final_acceptance/thesis_inputs")
    ok = result["status"] == "NOT_READY"
    return check_row(
        "missing_provenance_thesis_inputs",
        "failure mode",
        "PASS" if ok else "FAIL",
        f"thesis input státusz: {result['status']}",
        "" if ok else "Provenance nélkül nem lehet READY státusz.",
    )


def check_demo_path_rejection(tmp_root: Path) -> dict[str, str]:
    demo_path = tmp_root / "examples/lab/lab_events.jsonl"
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    demo_path.write_text("{}\n", encoding="utf-8")
    try:
        validate_input_path(demo_path, "lab input")
    except ValueError:
        return check_row("demo_path_rejected", "failure mode", "PASS", "demo path real-lab inputként elutasítva", path=demo_path.as_posix())
    return check_row(
        "demo_path_rejected",
        "failure mode",
        "FAIL",
        "demo path nem lett elutasítva",
        "A real-lab guardnak tiltania kell az examples/templates/tests eredetű inputokat.",
        demo_path.as_posix(),
    )


def check_missing_wazuh_alert(tmp_root: Path) -> dict[str, str]:
    ground_truth = tmp_root / "data/lab/lab_ground_truth.csv"
    write_minimal_ground_truth(ground_truth)
    result = run_post_session_input_check(
        ground_truth_path=ground_truth,
        wazuh_alerts_path=tmp_root / "data/wazuh/alerts.jsonl",
        zeek_conn_path=tmp_root / "data/lab/zeek/conn.log",
        flow_csv_path=tmp_root / "data/lab/flows.csv",
        output_dir=tmp_root / "reports/lab_session",
    )
    ok = result["status"] == "FAIL"
    return check_row(
        "missing_wazuh_alerts",
        "failure mode",
        "PASS" if ok else "FAIL",
        f"post-session input check státusz: {result['status']}",
        "" if ok else "Hiányzó Wazuh alert input mellett FAIL szükséges.",
    )


def check_missing_lab_feature(tmp_root: Path) -> dict[str, str]:
    ground_truth = tmp_root / "data/lab/lab_ground_truth.csv"
    alerts = tmp_root / "data/wazuh/alerts.jsonl"
    write_minimal_ground_truth(ground_truth)
    alerts.parent.mkdir(parents=True, exist_ok=True)
    alerts.write_text('{"timestamp":"2026-05-04T10:02:10Z"}\n', encoding="utf-8")
    try:
        validate_real_lab_inputs(
            ground_truth_path=ground_truth,
            features_path=tmp_root / "data/lab/lab_features.csv",
            wazuh_alerts_path=alerts,
            output_dir=tmp_root / "reports/lab_input_validation",
        )
    except FileNotFoundError:
        return check_row("missing_lab_features", "failure mode", "PASS", "hiányzó lab_features.csv elutasítva")
    return check_row(
        "missing_lab_features",
        "failure mode",
        "FAIL",
        "hiányzó lab_features.csv mellett nem történt hiba",
        "A real-lab input validációnak kötelezően kérnie kell a feature fájlt.",
    )


def check_postrun_without_provenance(tmp_root: Path) -> dict[str, str]:
    ground_truth = tmp_root / "data/lab/lab_ground_truth.csv"
    features = tmp_root / "data/lab/lab_features.csv"
    alerts = tmp_root / "data/wazuh/alerts.jsonl"
    write_minimal_ground_truth(ground_truth)
    write_minimal_features(features)
    alerts.parent.mkdir(parents=True, exist_ok=True)
    alerts.write_text('{"timestamp":"2026-05-04T10:02:10Z"}\n', encoding="utf-8")
    metrics = (
        "precision,recall,f1,false_positive_rate,false_negative_rate,alert_count,n_samples,n_attack,n_benign\n"
        "0.5,0.5,0.5,0.2,0.5,2,4,2,2\n"
    )
    for rel_path in [
        "results/wazuh_real/metrics_summary.csv",
        "results/ae_lab/metrics_summary.csv",
    ]:
        path = tmp_root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(metrics, encoding="utf-8")
    hybrid = tmp_root / "results/hybrid_real/metrics_summary.csv"
    hybrid.parent.mkdir(parents=True, exist_ok=True)
    hybrid.write_text("strategy," + metrics, encoding="utf-8")
    comparison = tmp_root / "results/real_comparison/metrics_comparison.csv"
    comparison.parent.mkdir(parents=True, exist_ok=True)
    comparison.write_text(
        "configuration,precision,recall,f1,false_positive_rate,false_negative_rate,alert_count,n_samples,n_attack,n_benign\n"
        "Wazuh-only,0.5,0.5,0.5,0.2,0.5,2,4,2,2\n"
        "AE-Minimal lab,0.5,0.5,0.4,0.2,0.5,2,4,2,2\n"
        "Hybrid OR,0.5,0.7,0.7,0.2,0.3,2,4,2,2\n"
        "Hybrid weighted,0.5,0.5,0.45,0.2,0.5,2,4,2,2\n"
        "Hybrid priority,0.5,0.5,0.45,0.2,0.5,2,4,2,2\n",
        encoding="utf-8",
    )
    (tmp_root / "results/real_comparison/metrics_comparison.md").write_text("| ok |\n", encoding="utf-8")
    for rel_path in [
        "reports/real_measurement/real_lab_results_report.md",
        "reports/real_measurement/thesis_real_lab_section.md",
    ]:
        path = tmp_root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n", encoding="utf-8")
    manifest = tmp_root / "reports/real_measurement/measurement_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("relative_path,sha256\nx,abc\n", encoding="utf-8")
    result = run_quality_gate(tmp_root, tmp_root / "reports/real_measurement_qa")
    ok = result["status"] != "READY"
    return check_row(
        "postrun_without_provenance_not_ready",
        "failure mode",
        "PASS" if ok else "FAIL",
        f"postrun QA státusz: {result['status']}",
        "" if ok else "Post-run QA provenance nélkül nem lehet READY.",
    )


def run_check(output_dir: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="final_acceptance_") as temp_dir:
        tmp_root = Path(temp_dir)
        rows = [
            check_thesis_without_provenance(tmp_root),
            check_demo_path_rejection(tmp_root),
            check_missing_wazuh_alert(tmp_root),
            check_missing_lab_feature(tmp_root),
            check_postrun_without_provenance(tmp_root),
        ]
    return write_check_outputs(
        output_dir=output_dir,
        basename="failure_modes_check",
        title="Elvárt hibamódok ellenőrzése",
        rows=rows,
        intro="Az ellenőrzések átmeneti könyvtárban futnak, és nem hoznak létre fő mérési eredményt.",
    )


def main() -> None:
    args = parse_args()
    result = run_check(Path(args.output_dir))
    print(f"[OK] Kimenet: {result['markdown']}")
    if has_fail(result["rows"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
