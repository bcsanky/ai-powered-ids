from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from ml.src.lab_session.common import now_iso


PLANNED_SCENARIOS = [
    "benign_ssh_login",
    "benign_package_update",
    "port_scan",
    "ssh_failed_logins",
    "ssh_bruteforce",
    "file_integrity_change",
    "privilege_change",
]
REQUIRED_OUTPUTS = [
    "data/lab/lab_ground_truth.csv",
    "data/lab/lab_features.csv",
    "data/wazuh/alerts.jsonl",
    "reports/real_measurement/measurement_provenance.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--attacker-ip", default="")
    parser.add_argument("--target-ip", default="")
    parser.add_argument("--wazuh-manager", default="")
    parser.add_argument("--output-dir", default="reports/lab_session")
    return parser.parse_args()


def build_plan(session_id: str, attacker_ip: str, target_ip: str, wazuh_manager: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "created_at": now_iso(),
        "attacker_ip": attacker_ip,
        "target_ip": target_ip,
        "wazuh_manager": wazuh_manager,
        "planned_scenarios": PLANNED_SCENARIOS,
        "required_outputs": REQUIRED_OUTPUTS,
        "note": "Ez session terv és operátori naplózási segédlet; nem mérési eredmény.",
    }


def markdown_plan(plan: dict[str, Any]) -> str:
    lines = [
        "# Lab session terv",
        "",
        "Ez a dokumentum a valós lab mérés előkészítését segíti. Nem tartalmaz mérési eredményt.",
        "",
        f"- Session ID: `{plan['session_id']}`",
        f"- Létrehozva: `{plan['created_at']}`",
        f"- Attacker IP: `{plan['attacker_ip'] or 'kitöltendő'}`",
        f"- Target IP: `{plan['target_ip'] or 'kitöltendő'}`",
        f"- Wazuh manager: `{plan['wazuh_manager'] or 'kitöltendő'}`",
        "",
        "## Tervezett szcenáriók",
        "",
    ]
    lines.extend(f"- `{scenario}`" for scenario in plan["planned_scenarios"])
    lines.extend(["", "## Elvárt kimenetek", ""])
    lines.extend(f"- `{path}`" for path in plan["required_outputs"])
    lines.extend(
        [
            "",
            "## Megjegyzés",
            "",
            "A felsorolt kimeneteket tényleges lab futásból kell előállítani. A session terv nem helyettesíti a ground truth, Wazuh export vagy flow/Zeek bemenetet.",
        ]
    )
    return "\n".join(lines) + "\n"


def operator_log_template(plan: dict[str, Any]) -> str:
    lines = [
        "# Operátori parancsnapló sablon",
        "",
        f"Session ID: `{plan['session_id']}`",
        "",
        "A táblázatot a tényleges mérés közben kell kitölteni. Ez a sablon nem mérési eredmény.",
        "",
        "| Időpont | Futtatott parancs | Célgép | Várt hatás | Megjegyzés |",
        "| --- | --- | --- | --- | --- |",
    ]
    for scenario in plan["planned_scenarios"]:
        lines.append(f"|  |  |  | `{scenario}` szcenárió végrehajtása |  |")
    return "\n".join(lines) + "\n"


def create_session_plan(
    *,
    session_id: str,
    attacker_ip: str,
    target_ip: str,
    wazuh_manager: str,
    output_dir: Path,
) -> dict[str, Path]:
    plan = build_plan(session_id, attacker_ip, target_ip, wazuh_manager)
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "session_plan.yaml"
    md_path = output_dir / "session_plan.md"
    log_path = output_dir / "operator_command_log_template.md"
    yaml_path.write_text(yaml.safe_dump(plan, sort_keys=False, allow_unicode=True), encoding="utf-8")
    md_path.write_text(markdown_plan(plan), encoding="utf-8")
    log_path.write_text(operator_log_template(plan), encoding="utf-8")
    return {"yaml": yaml_path, "markdown": md_path, "operator_log_template": log_path}


def main() -> None:
    args = parse_args()
    outputs = create_session_plan(
        session_id=args.session_id,
        attacker_ip=args.attacker_ip,
        target_ip=args.target_ip,
        wazuh_manager=args.wazuh_manager,
        output_dir=Path(args.output_dir),
    )
    for path in outputs.values():
        print(f"[OK] Kimenet: {path}")


if __name__ == "__main__":
    main()

