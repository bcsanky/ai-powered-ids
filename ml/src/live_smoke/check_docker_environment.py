from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ml.src.live_smoke.common import check_row, has_fail, run_command, summarize_command_result, write_check_outputs


CONTAINER_PATTERNS = ("wazuh", "opensearch", "indexer", "mlservice")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/live_smoke")
    parser.add_argument("--compose-file", default="infra/docker-compose.yml")
    return parser.parse_args()


def run_check(output_dir: Path, compose_file: Path = Path("infra/docker-compose.yml")) -> dict[str, Any]:
    rows: list[dict[str, str]] = []

    docker_version = run_command(["docker", "--version"], timeout_seconds=10)
    docker_available = docker_version.returncode == 0
    rows.append(
        check_row(
            "docker_command",
            "Docker",
            "PASS" if docker_available else "WARN",
            "docker parancs elérhető" if docker_available else "docker parancs nem elérhető",
            "" if docker_available else "Telepítsd vagy indítsd el a Docker környezetet a mérés előtt.",
        )
    )

    compose_version = run_command(["docker", "compose", "version"], timeout_seconds=10)
    compose_available = compose_version.returncode == 0
    rows.append(
        check_row(
            "docker_compose_command",
            "Docker",
            "PASS" if compose_available else "WARN",
            "docker compose elérhető" if compose_available else "docker compose nem elérhető",
            "" if compose_available else "Ellenőrizd a Docker Compose telepítést.",
        )
    )

    rows.append(
        check_row(
            "compose_file",
            "Docker Compose",
            "PASS" if compose_file.exists() else "FAIL",
            f"compose fájl elérhető: {compose_file}" if compose_file.exists() else f"hiányzó compose fájl: {compose_file}",
            "" if compose_file.exists() else "A konténeres mérési környezethez szükséges compose fájlt pótolni kell.",
            compose_file.as_posix(),
        )
    )

    if docker_available and compose_available and compose_file.exists():
        config = run_command(["docker", "compose", "-f", compose_file.as_posix(), "config"], timeout_seconds=30)
        rows.append(
            check_row(
                "compose_config",
                "Docker Compose",
                "PASS" if config.returncode == 0 else "WARN",
                "docker compose config lefutott" if config.returncode == 0 else summarize_command_result(config),
                "" if config.returncode == 0 else "Ellenőrizd a compose konfigurációt a mérés előtt.",
                compose_file.as_posix(),
            )
        )
    else:
        rows.append(
            check_row(
                "compose_config",
                "Docker Compose",
                "SKIP",
                "compose config ellenőrzés kihagyva, mert a Docker/Compose nem elérhető",
                "Indítsd el vagy telepítsd a konténerkörnyezetet.",
                compose_file.as_posix(),
            )
        )

    if docker_available:
        ps = run_command(["docker", "ps", "--format", "{{.Names}}"], timeout_seconds=15)
        if ps.returncode == 0:
            names = [line.strip() for line in ps.stdout.splitlines() if line.strip()]
            matched = [name for name in names if any(pattern in name.lower() for pattern in CONTAINER_PATTERNS)]
            rows.append(
                check_row(
                    "docker_ps",
                    "Docker",
                    "PASS",
                    f"futó konténerek listázva: {len(names)}",
                    "",
                )
            )
            rows.append(
                check_row(
                    "expected_containers",
                    "Docker",
                    "PASS" if matched else "WARN",
                    "mérési stackhez kapcsolódó konténerminta látható"
                    if matched
                    else "nem látszik Wazuh/OpenSearch/mlservice konténerminta",
                    "" if matched else "Ha a méréshez konténeres stack kell, indítsd el a mérés előtt.",
                )
            )
        else:
            rows.append(
                check_row(
                    "docker_ps",
                    "Docker",
                    "WARN",
                    summarize_command_result(ps),
                    "Ellenőrizd, hogy a Docker daemon fut-e.",
                )
            )
    else:
        rows.append(check_row("docker_ps", "Docker", "SKIP", "konténerlista kihagyva", "Docker nélkül nem futtatható."))

    return write_check_outputs(
        output_dir=output_dir,
        basename="docker_environment_check",
        title="Docker környezet ellenőrzés",
        rows=rows,
        intro="Ez az ellenőrzés csak a konténeres környezet állapotát vizsgálja, konténert nem indít és nem állít le.",
    )


def main() -> None:
    args = parse_args()
    result = run_check(Path(args.output_dir), Path(args.compose_file))
    print(f"[OK] Kimenet: {result['markdown']}")
    if has_fail(result["rows"]):
        sys.exit(1)


if __name__ == "__main__":
    main()

