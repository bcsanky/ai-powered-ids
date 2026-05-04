from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ml.src.live_smoke.common import check_row, has_fail, path_guard_status, write_check_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", default="data/lab/lab_ground_truth.csv")
    parser.add_argument("--lab-features", default="data/lab/lab_features.csv")
    parser.add_argument("--wazuh-alerts", default="data/wazuh/alerts.jsonl")
    parser.add_argument("--output-dir", default="reports/live_smoke")
    return parser.parse_args()


def check_path(label: str, path: Path) -> dict[str, str]:
    allowed, reason = path_guard_status(path)
    if not allowed:
        return check_row(
            f"path_{label}",
            "Real input útvonal",
            "FAIL",
            f"tiltott real-lab input útvonal: {reason}",
            "Válassz tényleges lab futásból származó data/lab vagy data/wazuh útvonalat.",
            path.as_posix(),
        )
    if not path.exists():
        return check_row(
            f"path_{label}",
            "Real input útvonal",
            "WARN",
            "a fájl még nem létezik",
            "A tényleges mérés során kell előállítani, ez mérés előtt elfogadható.",
            path.as_posix(),
        )
    if path.stat().st_size <= 0:
        return check_row(
            f"path_{label}",
            "Real input útvonal",
            "FAIL",
            "a fájl létezik, de üres",
            "Ellenőrizd, hogy tényleges export vagy mérési input került-e ide.",
            path.as_posix(),
        )
    return check_row(
        f"path_{label}",
        "Real input útvonal",
        "PASS",
        f"a fájl létezik és nem üres ({path.stat().st_size} bájt)",
        "",
        path.as_posix(),
    )


def run_check(output_dir: Path, *, ground_truth: Path, lab_features: Path, wazuh_alerts: Path) -> dict[str, Any]:
    rows = [
        check_path("ground_truth", ground_truth),
        check_path("lab_features", lab_features),
        check_path("wazuh_alerts", wazuh_alerts),
    ]
    return write_check_outputs(
        output_dir=output_dir,
        basename="real_input_paths_check",
        title="Real-lab input útvonalak ellenőrzése",
        rows=rows,
        intro="Az ellenőrzés csak az input útvonalak eredetét és létezését vizsgálja. Hiányzó fájl mérés előtt figyelmeztetés, tiltott demo/teszt útvonal hiba.",
    )


def main() -> None:
    args = parse_args()
    result = run_check(
        Path(args.output_dir),
        ground_truth=Path(args.ground_truth),
        lab_features=Path(args.lab_features),
        wazuh_alerts=Path(args.wazuh_alerts),
    )
    print(f"[OK] Kimenet: {result['markdown']}")
    if has_fail(result["rows"]):
        sys.exit(1)


if __name__ == "__main__":
    main()

