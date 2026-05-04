from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import requests

from ml.src.live_smoke.common import check_row, has_fail, parse_bool, redact_secret, write_check_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opensearch-url", default="https://localhost:9200")
    parser.add_argument("--index-pattern", default="wazuh-alerts-*")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password", default="")
    parser.add_argument("--verify-tls", default="false")
    parser.add_argument("--required", default="false")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output-dir", default="reports/live_smoke")
    return parser.parse_args()


def run_check(
    output_dir: Path,
    *,
    opensearch_url: str,
    index_pattern: str,
    username: str,
    password: str,
    verify_tls: bool,
    required: bool = False,
    timeout: float = 5.0,
) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    extra: dict[str, Any] = {
        "opensearch_url": opensearch_url,
        "index_pattern": index_pattern,
        "verify_tls": verify_tls,
        "password_recorded": False,
    }
    if not verify_tls:
        rows.append(
            check_row(
                "opensearch_tls",
                "OpenSearch",
                "WARN",
                "TLS tanúsítványellenőrzés ki van kapcsolva",
                "Izolált laborban elfogadható lehet, éles környezetben nem javasolt.",
                opensearch_url,
            )
        )
    if not password:
        rows.append(
            check_row(
                "opensearch_password",
                "OpenSearch",
                "FAIL" if required else "WARN",
                "OpenSearch jelszó nincs megadva, ezért nem történt hálózati ellenőrzés",
                "Add meg az OPENSEARCH_PASSWORD értékét, ha a kapcsolatot is ellenőrizni kell.",
                opensearch_url,
            )
        )
        return write_check_outputs(
            output_dir=output_dir,
            basename="opensearch_connection_check",
            title="OpenSearch kapcsolat ellenőrzés",
            rows=rows,
            intro="Az ellenőrzés csak olvasási próba és count lekérdezés; alert exportot és indexírást nem végez.",
            extra_payload=extra,
        )

    auth = (username, password)
    try:
        response = requests.get(opensearch_url, auth=auth, verify=verify_tls, timeout=timeout)
        ok = response.status_code < 400
        rows.append(
            check_row(
                "opensearch_root",
                "OpenSearch",
                "PASS" if ok else ("FAIL" if required else "WARN"),
                f"root endpoint HTTP státusz: {response.status_code}",
                "" if ok else "Ellenőrizd az URL-t, a hitelesítést és a szolgáltatás állapotát.",
                opensearch_url,
            )
        )
    except Exception as exc:  # noqa: BLE001
        rows.append(
            check_row(
                "opensearch_root",
                "OpenSearch",
                "FAIL" if required else "WARN",
                redact_secret(f"root endpoint nem elérhető: {exc}", password),
                "A mérés előtt ellenőrizd a Wazuh/OpenSearch elérést.",
                opensearch_url,
            )
        )

    count_url = f"{opensearch_url.rstrip('/')}/{index_pattern}/_count"
    try:
        count_response = requests.get(count_url, auth=auth, verify=verify_tls, timeout=timeout)
        if count_response.status_code < 400:
            payload = count_response.json()
            rows.append(
                check_row(
                    "opensearch_count",
                    "OpenSearch",
                    "PASS",
                    f"index count lekérdezés sikeres, count={payload.get('count', '')}",
                    "",
                    count_url,
                )
            )
        else:
            rows.append(
                check_row(
                    "opensearch_count",
                    "OpenSearch",
                    "FAIL" if required else "WARN",
                    f"index count HTTP státusz: {count_response.status_code}",
                    "Ellenőrizd az index mintát és jogosultságot.",
                    count_url,
                )
            )
    except Exception as exc:  # noqa: BLE001
        rows.append(
            check_row(
                "opensearch_count",
                "OpenSearch",
                "FAIL" if required else "WARN",
                redact_secret(f"index count nem sikerült: {exc}", password),
                "A mérés előtt ellenőrizd az index mintát és jogosultságot.",
                count_url,
            )
        )

    return write_check_outputs(
        output_dir=output_dir,
        basename="opensearch_connection_check",
        title="OpenSearch kapcsolat ellenőrzés",
        rows=rows,
        intro="Az ellenőrzés csak olvasási próba és count lekérdezés; alert exportot és indexírást nem végez.",
        extra_payload=extra,
    )


def main() -> None:
    args = parse_args()
    result = run_check(
        Path(args.output_dir),
        opensearch_url=args.opensearch_url,
        index_pattern=args.index_pattern,
        username=args.username,
        password=args.password,
        verify_tls=parse_bool(args.verify_tls),
        required=parse_bool(args.required),
        timeout=args.timeout,
    )
    print(f"[OK] Kimenet: {result['markdown']}")
    if has_fail(result["rows"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
