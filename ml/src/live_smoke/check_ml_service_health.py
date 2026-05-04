from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import requests

from ml.src.live_smoke.common import check_row, has_fail, parse_bool, write_check_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/health")
    parser.add_argument("--required", default="false")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output-dir", default="reports/live_smoke")
    return parser.parse_args()


def run_check(output_dir: Path, url: str, *, required: bool = False, timeout: float = 5.0) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    try:
        response = requests.get(url, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        rows.append(
            check_row(
                "ml_health_http",
                "ML service",
                "FAIL" if required else "WARN",
                f"ML service nem elérhető: {exc}",
                "Indítsd el az ML service-t, ha élő integrációs mérés előtt szükséges.",
                url,
            )
        )
        return write_check_outputs(
            output_dir=output_dir,
            basename="ml_service_health_check",
            title="ML service health ellenőrzés",
            rows=rows,
            intro="Az ellenőrzés kizárólag a /health végpontot hívja meg, scoring kérést nem küld.",
        )

    rows.append(
        check_row(
            "ml_health_status_code",
            "ML service",
            "PASS" if response.status_code == 200 else ("FAIL" if required else "WARN"),
            f"HTTP státusz: {response.status_code}",
            "" if response.status_code == 200 else "Ellenőrizd az ML service konfigurációját.",
            url,
        )
    )
    try:
        payload = response.json()
        rows.append(check_row("ml_health_json", "ML service", "PASS", "JSON válasz értelmezhető", path=url))
    except ValueError:
        payload = {}
        rows.append(
            check_row(
                "ml_health_json",
                "ML service",
                "FAIL" if required else "WARN",
                "a /health válasz nem JSON",
                "A szolgáltatás health válaszát ellenőrizni kell.",
                url,
            )
        )

    if payload:
        if "model_loaded" in payload:
            status = "PASS" if bool(payload.get("model_loaded")) else ("FAIL" if required else "WARN")
            rows.append(
                check_row(
                    "ml_health_model_loaded",
                    "ML service",
                    status,
                    f"model_loaded={payload.get('model_loaded')}",
                    "" if status == "PASS" else "A modellbetöltést ellenőrizni kell az élő mérés előtt.",
                    url,
                )
            )
        else:
            rows.append(
                check_row(
                    "ml_health_model_loaded",
                    "ML service",
                    "WARN",
                    "model_loaded mező nincs a válaszban",
                    "Nem minden health implementáció adja vissza, de mérés előtt érdemes ellenőrizni.",
                    url,
                )
            )
        rows.append(
            check_row(
                "ml_health_model_version",
                "ML service",
                "PASS" if payload.get("model_version") else "WARN",
                f"model_version={payload.get('model_version', '')}" if payload.get("model_version") else "model_version mező nem érhető el",
                "" if payload.get("model_version") else "A verziómező hiánya nem blokkoló, de a reprodukálhatósághoz hasznos.",
                url,
            )
        )

    return write_check_outputs(
        output_dir=output_dir,
        basename="ml_service_health_check",
        title="ML service health ellenőrzés",
        rows=rows,
        intro="Az ellenőrzés kizárólag a /health végpontot hívja meg, scoring kérést nem küld.",
        extra_payload={"health_payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else []},
    )


def main() -> None:
    args = parse_args()
    result = run_check(Path(args.output_dir), args.url, required=parse_bool(args.required), timeout=args.timeout)
    print(f"[OK] Kimenet: {result['markdown']}")
    if has_fail(result["rows"]):
        sys.exit(1)


if __name__ == "__main__":
    main()

