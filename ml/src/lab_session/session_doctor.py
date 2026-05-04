from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import requests

from ml.src.lab_session.common import (
    check_row,
    has_fail,
    now_iso,
    overall_status,
    write_check_report,
    write_metadata,
    write_summary_csv,
)


REQUIRED_TARGETS = [
    "lab-templates",
    "real-measurement-preflight",
    "final-real-measurement-package-with-provenance",
    "final-live-integration",
    "repo-hygiene-check",
]
DIRECTORIES = [
    Path("data/lab"),
    Path("data/wazuh"),
    Path("data/processed"),
    Path("reports"),
]
TEMPLATES = [
    Path("templates/lab/lab_ground_truth_template.csv"),
    Path("templates/lab/lab_features_template.csv"),
    Path("templates/lab/lab_scenarios_template.yaml"),
]
DOCUMENTS = [
    Path("docs/lab_attack_scenarios_runbook.md"),
    Path("docs/final_real_measurement_checklist.md"),
    Path("docs/real_measurement_export_and_packaging_runbook.md"),
    Path("docs/data_provenance_and_no_fake_measurements.md"),
    Path("docs/live_integration_runbook.md"),
]
IMPORTS = [
    "ml.src.lab_capture.event_marker",
    "ml.src.lab_features.validate_real_lab_inputs",
    "ml.src.repo_hygiene.check_no_demo_real_results",
    "ml.src.real_measurement.validate_measurement_bundle",
    "ml.src.live_integration.enrich_wazuh_alerts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/lab_session")
    parser.add_argument("--check-opensearch", action="store_true")
    parser.add_argument("--opensearch-url", default="https://localhost:9200")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password", default="")
    parser.add_argument("--verify-tls", choices=["true", "false"], default="true")
    parser.add_argument("--opensearch-required", action="store_true")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def makefile_has_target(makefile_text: str, target: str) -> bool:
    return f"{target}:" in makefile_text


def check_project(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    makefile = root / "Makefile"
    if not makefile.exists():
        rows.append(check_row("makefile", "project", "FAIL", "Hiányzik a Makefile.", "A mérés előtt pótolni kell."))
        return rows
    text = makefile.read_text(encoding="utf-8")
    rows.append(check_row("makefile", "project", "PASS", "A Makefile létezik."))
    for target in REQUIRED_TARGETS:
        status = "PASS" if makefile_has_target(text, target) else "FAIL"
        rows.append(
            check_row(
                f"make_target_{target}",
                "project",
                status,
                f"Make target ellenőrzés: {target}",
                "" if status == "PASS" else "A Makefile targetet pótolni kell.",
            )
        )

    required_files = [
        (Path("experiments/final/ae_minimal.yaml"), "ae_minimal_config", "FAIL"),
        (Path("artifacts/final/final-ae-minimal-v1"), "ae_minimal_model_root", "WARN"),
        (Path("data/processed/final/ae_minimal/preprocess.pkl"), "ae_minimal_preprocess", "WARN"),
    ]
    for rel_path, check_id, missing_status in required_files:
        path = root / rel_path
        if path.exists():
            rows.append(check_row(check_id, "project", "PASS", f"Elérhető: {rel_path}"))
        else:
            rows.append(
                check_row(
                    check_id,
                    "project",
                    missing_status,
                    f"Hiányzik: {rel_path}",
                    "A tényleges AE scoring előtt szükséges, de a session terv elkészítését nem akadályozza."
                    if missing_status == "WARN"
                    else "A mérés előtt pótolni kell.",
                )
            )
    return rows


def check_directories(root: Path, output_dir: Path) -> list[dict[str, str]]:
    rows = []
    directories = DIRECTORIES + [output_dir]
    for rel_path in directories:
        path = root / rel_path
        if path.exists():
            rows.append(check_row(rel_path.as_posix(), "directory", "PASS", f"Könyvtár létezik: {rel_path}"))
        else:
            path.mkdir(parents=True, exist_ok=True)
            rows.append(
                check_row(
                    rel_path.as_posix(),
                    "directory",
                    "WARN",
                    f"Könyvtár létrehozva: {rel_path}",
                    "A könyvtár üres, a valós inputokat az operátornak kell előállítania.",
                )
            )
    return rows


def check_paths(root: Path, paths: list[Path], category: str) -> list[dict[str, str]]:
    rows = []
    for rel_path in paths:
        status = "PASS" if (root / rel_path).exists() else "FAIL"
        rows.append(
            check_row(
                rel_path.as_posix(),
                category,
                status,
                f"{'Elérhető' if status == 'PASS' else 'Hiányzik'}: {rel_path}",
                "" if status == "PASS" else "A session előtt pótolni kell.",
            )
        )
    return rows


def check_imports() -> list[dict[str, str]]:
    rows = []
    for module_name in IMPORTS:
        try:
            importlib.import_module(module_name)
            rows.append(check_row(module_name, "python_import", "PASS", f"Importálható: {module_name}"))
        except Exception as exc:  # noqa: BLE001
            rows.append(
                check_row(
                    module_name,
                    "python_import",
                    "FAIL",
                    f"Nem importálható: {module_name}: {exc}",
                    "Futtasd a final-validate célpontot, és javítsd az import hibát.",
                )
            )
    return rows


def check_opensearch(
    *,
    url: str,
    username: str,
    password: str,
    verify_tls: bool,
    required: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not password:
        rows.append(
            check_row(
                "opensearch_password",
                "opensearch",
                "FAIL" if required else "WARN",
                "OpenSearch jelszó nincs megadva.",
                "Add meg a jelszót csak lokális környezetben vagy környezeti változóként.",
            )
        )
        return rows
    try:
        response = requests.get(url, auth=(username, password), verify=verify_tls, timeout=5)
        status = "PASS" if response.status_code < 400 else ("FAIL" if required else "WARN")
        rows.append(
            check_row(
                "opensearch_connectivity",
                "opensearch",
                status,
                f"OpenSearch HTTP státusz: {response.status_code}",
                "" if status == "PASS" else "Ellenőrizd az URL-t, hitelesítést és hálózati elérést.",
            )
        )
    except Exception as exc:  # noqa: BLE001
        rows.append(
            check_row(
                "opensearch_connectivity",
                "opensearch",
                "FAIL" if required else "WARN",
                f"OpenSearch nem elérhető: {exc}",
                "A mérés előtt ellenőrizd a Wazuh/OpenSearch szolgáltatást.",
            )
        )
    if not verify_tls:
        rows.append(
            check_row(
                "opensearch_tls",
                "opensearch",
                "WARN",
                "A TLS tanúsítványellenőrzés ki van kapcsolva.",
                "Izolált lab környezetben elfogadható, éles környezetben nem javasolt.",
            )
        )
    return rows


def run_session_doctor(
    *,
    root: Path,
    output_dir: Path,
    check_opensearch_enabled: bool = False,
    opensearch_url: str = "https://localhost:9200",
    username: str = "admin",
    password: str = "",
    verify_tls: bool = True,
    opensearch_required: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    rows.extend(check_project(root))
    rows.extend(check_directories(root, output_dir))
    rows.extend(check_paths(root, TEMPLATES, "template"))
    rows.extend(check_paths(root, DOCUMENTS, "documentation"))
    rows.extend(check_imports())
    if check_opensearch_enabled:
        rows.extend(
            check_opensearch(
                url=opensearch_url,
                username=username,
                password=password,
                verify_tls=verify_tls,
                required=opensearch_required,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "session_doctor_report.md"
    summary_path = output_dir / "session_doctor_summary.csv"
    metadata_path = output_dir / "session_doctor_metadata.json"
    write_check_report(
        rows,
        report_path,
        title="Lab session doctor",
        intro="Ez a riport a valós lab mérés előtti operátori készültséget ellenőrzi.",
    )
    write_summary_csv(rows, summary_path)
    write_metadata(
        metadata_path,
        {
            "created_at": now_iso(),
            "status": overall_status(rows),
            "check_opensearch": check_opensearch_enabled,
            "opensearch_url": opensearch_url if check_opensearch_enabled else "",
            "verify_tls": verify_tls if check_opensearch_enabled else None,
            "password_recorded": False,
        },
    )
    return {
        "status": overall_status(rows),
        "rows": rows,
        "report": report_path,
        "summary": summary_path,
        "metadata": metadata_path,
    }


def main() -> None:
    args = parse_args()
    result = run_session_doctor(
        root=Path(args.root),
        output_dir=Path(args.output_dir),
        check_opensearch_enabled=args.check_opensearch,
        opensearch_url=args.opensearch_url,
        username=args.username,
        password=args.password,
        verify_tls=args.verify_tls == "true",
        opensearch_required=args.opensearch_required,
    )
    print(f"[OK] Kimenet: {result['report']}")
    print(f"[OK] Kimenet: {result['summary']}")
    print(f"[OK] Kimenet: {result['metadata']}")
    if has_fail(result["rows"]):
        sys.exit(1)


if __name__ == "__main__":
    main()

