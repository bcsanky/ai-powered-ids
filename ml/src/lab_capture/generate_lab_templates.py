from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


GROUND_TRUTH_COLUMNS = [
    "event_id",
    "timestamp_start",
    "timestamp_end",
    "scenario",
    "label",
    "attack_type",
    "source_ip",
    "target_ip",
]
LAB_FEATURE_COLUMNS = [
    "event_id",
    "timestamp",
    "destination_port",
    "flow_duration",
    "total_fwd_packets",
    "total_backward_packets",
    "flow_bytes_per_sec",
    "flow_packets_per_sec",
    "protocol",
    "source_ip",
    "target_ip",
    "scenario",
]
SCENARIOS_YAML = """\
scenarios:
  - name: benign_ssh_login
    label: benign
    attack_type: ""
    description: "Sikeres, engedélyezett SSH belépés saját lab gépre."
  - name: benign_package_update
    label: benign
    attack_type: ""
    description: "Csomaglista frissítése vagy csomagtelepítés saját lab gépen."
  - name: port_scan
    label: attack
    attack_type: port_scan
    description: "Kontrollált port scan saját izolált lab célgépen."
  - name: ssh_failed_logins
    label: attack
    attack_type: ssh_failed_logins
    description: "Néhány szándékosan sikertelen SSH belépési kísérlet saját teszt userrel."
  - name: ssh_bruteforce
    label: attack
    attack_type: ssh_bruteforce
    description: "Korlátozott, izolált lab brute force jellegű SSH eseménysor."
  - name: file_integrity_change
    label: attack
    attack_type: file_integrity_change
    description: "Wazuh FIM által figyelt tesztfájl kontrollált módosítása."
  - name: privilege_change
    label: attack
    attack_type: privilege_change
    description: "Saját lab user jogosultságváltozásának kontrollált tesztje."
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="templates/lab")
    return parser.parse_args()


def generate_templates(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_path = output_dir / "lab_ground_truth_template.csv"
    features_path = output_dir / "lab_features_template.csv"
    scenarios_path = output_dir / "lab_scenarios_template.yaml"

    pd.DataFrame(columns=GROUND_TRUTH_COLUMNS).to_csv(ground_truth_path, index=False)
    pd.DataFrame(columns=LAB_FEATURE_COLUMNS).to_csv(features_path, index=False)
    scenarios_path.write_text(SCENARIOS_YAML, encoding="utf-8")
    return {
        "ground_truth": ground_truth_path,
        "features": features_path,
        "scenarios": scenarios_path,
    }


def main() -> None:
    args = parse_args()
    outputs = generate_templates(Path(args.output_dir))
    for path in outputs.values():
        print(f"[OK] Sablon: {path}")


if __name__ == "__main__":
    main()
