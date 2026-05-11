from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ml.src.infra_diagnostics.check_wazuh_bind_mounts import aggregate_status, path_type, recommendation_for


FIELDS = [
    "check_id",
    "service",
    "source_path",
    "target_path",
    "expected_type",
    "actual_type",
    "status",
    "message",
    "recommendation",
]
FILE_TARGET_SUFFIXES = (".yml", ".yaml", ".pem", ".key", ".conf", ".json", ".crt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose-file", default="infra/docker-compose.yml")
    parser.add_argument("--output-dir", default="reports/infra_diagnostics")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_volume_string(value: str) -> tuple[str, str] | None:
    parts = value.split(":")
    if len(parts) < 2:
        return None
    source, target = parts[0], parts[1]
    if not source.startswith((".", "/")):
        return None
    return source, target


def parse_bind_mounts(compose_path: Path) -> list[dict[str, str]]:
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8")) or {}
    compose_dir = compose_path.parent
    mounts: list[dict[str, str]] = []
    for service, service_config in sorted((compose.get("services") or {}).items()):
        for volume in service_config.get("volumes", []) or []:
            parsed: tuple[str, str] | None = None
            if isinstance(volume, str):
                parsed = parse_volume_string(volume)
            elif isinstance(volume, dict) and volume.get("type") == "bind":
                parsed = (str(volume.get("source", "")), str(volume.get("target", "")))
            if not parsed:
                continue
            source, target = parsed
            source_path = Path(source)
            if not source_path.is_absolute():
                source_path = (compose_dir / source_path).resolve()
            mounts.append(
                {
                    "service": str(service),
                    "source_raw": source,
                    "source_path": source_path.as_posix(),
                    "target_path": target,
                    "expected_type": expected_type_for_target(target),
                }
            )
    return mounts


def expected_type_for_target(target: str) -> str:
    if target.endswith(FILE_TARGET_SUFFIXES):
        return "file"
    if Path(target).name and "." in Path(target).name:
        return "file"
    return "directory_or_volume"


def check_compose_mount_policy(root: Path, compose_file: Path) -> list[dict[str, str]]:
    compose_path = compose_file if compose_file.is_absolute() else root / compose_file
    rows: list[dict[str, str]] = []
    if not compose_path.exists():
        return [
            {
                "check_id": "compose_file",
                "service": "",
                "source_path": compose_path.as_posix(),
                "target_path": "",
                "expected_type": "file",
                "actual_type": "missing",
                "status": "FAIL",
                "message": "docker-compose.yml hiányzik.",
                "recommendation": "Állítsd helyre az infra/docker-compose.yml fájlt.",
            }
        ]
    try:
        mounts = parse_bind_mounts(compose_path)
    except yaml.YAMLError as exc:
        return [
            {
                "check_id": "compose_parse",
                "service": "",
                "source_path": compose_path.as_posix(),
                "target_path": "",
                "expected_type": "valid_yaml",
                "actual_type": "parse_error",
                "status": "FAIL",
                "message": f"A compose YAML nem parse-olható: {exc}",
                "recommendation": "Javítsd a YAML szintaxist.",
            }
        ]
    for index, mount in enumerate(mounts, start=1):
        expected = mount["expected_type"]
        source_path = Path(mount["source_path"])
        actual = path_type(source_path)
        if expected == "file":
            if actual == "file":
                status = "PASS"
                message = "Fájl targethez fájl source tartozik."
            elif actual == "directory":
                status = "FAIL"
                message = "Fájl targethez könyvtár source tartozik."
            elif actual == "missing":
                status = "WARN"
                message = "Fájl targethez tartozó source hiányzik."
            else:
                status = "FAIL"
                message = "Fájl targethez nem normál fájl source tartozik."
        else:
            status = "SKIP"
            message = "Nem fájl target; a named volume vagy könyvtár jellegű mount policy itt nem releváns."
        source_rel = source_path.as_posix()
        try:
            source_rel = source_path.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
        rows.append(
            {
                "check_id": f"compose_mount_{index:02d}",
                "service": mount["service"],
                "source_path": source_rel,
                "target_path": mount["target_path"],
                "expected_type": expected,
                "actual_type": actual,
                "status": status,
                "message": message,
                "recommendation": recommendation_for(source_rel, actual) if expected == "file" else "",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    relevant = [row for row in rows if row["status"] != "SKIP"]
    status = aggregate_status(relevant) if relevant else "PASS"
    lines = [
        "# Docker Compose bind mount policy",
        "",
        f"Összesített státusz: **{status}**",
        "",
        "Ez a riport a compose fájl bind mountjait vizsgálja. Nem módosítja a host fájlokat és nem indít konténert.",
        "",
        "| Service | Source | Target | Expected | Actual | Status | Message | Recommendation |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['service']} | `{row['source_path']}` | `{row['target_path']}` | {row['expected_type']} | "
            f"{row['actual_type']} | {row['status']} | {row['message']} | {row['recommendation']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(rows: list[dict[str, str]], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "docker_compose_mount_policy.csv"
    md_path = output_dir / "docker_compose_mount_policy.md"
    json_path = output_dir / "docker_compose_mount_policy.json"
    relevant = [row for row in rows if row["status"] != "SKIP"]
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    json_path.write_text(
        json.dumps(
            {"created_at": now_utc(), "status": aggregate_status(relevant) if relevant else "PASS", "checks": rows},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"csv": csv_path, "markdown": md_path, "json": json_path}


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows = check_compose_mount_policy(root, Path(args.compose_file))
    outputs = write_outputs(rows, Path(args.output_dir))
    for path in outputs.values():
        print(f"[OK] Kimenet: {path}")


if __name__ == "__main__":
    main()
