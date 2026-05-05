from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_WAZUH_FILE_MOUNTS = [
    ("infra/wazuh/certs/root-ca-manager.pem", "/etc/ssl/root-ca.pem"),
    ("infra/wazuh/certs/wazuh.manager.pem", "/etc/ssl/filebeat.pem"),
    ("infra/wazuh/certs/wazuh.manager-key.pem", "/etc/ssl/filebeat.key"),
    ("infra/wazuh/config/wazuh_cluster/wazuh_manager.conf", "/wazuh-config-mount/etc/ossec.conf"),
    ("infra/wazuh/certs/root-ca.pem", "/usr/share/wazuh-indexer/config/certs/root-ca.pem"),
    ("infra/wazuh/certs/wazuh.indexer-key.pem", "/usr/share/wazuh-indexer/config/certs/wazuh.indexer.key"),
    ("infra/wazuh/certs/wazuh.indexer.pem", "/usr/share/wazuh-indexer/config/certs/wazuh.indexer.pem"),
    ("infra/wazuh/certs/admin.pem", "/usr/share/wazuh-indexer/config/certs/admin.pem"),
    ("infra/wazuh/certs/admin-key.pem", "/usr/share/wazuh-indexer/config/certs/admin-key.pem"),
    ("infra/wazuh/config/wazuh_indexer/wazuh.indexer.yml", "/usr/share/wazuh-indexer/config/opensearch.yml"),
    (
        "infra/wazuh/config/wazuh_indexer/internal_users.yml",
        "/usr/share/wazuh-indexer/config/opensearch-security/internal_users.yml",
    ),
    ("infra/wazuh/certs/wazuh.dashboard.pem", "/usr/share/wazuh-dashboard/certs/wazuh-dashboard.pem"),
    ("infra/wazuh/certs/wazuh.dashboard-key.pem", "/usr/share/wazuh-dashboard/certs/wazuh-dashboard-key.pem"),
    (
        "infra/wazuh/config/wazuh_dashboard/opensearch_dashboards.yml",
        "/usr/share/wazuh-dashboard/config/opensearch_dashboards.yml",
    ),
    ("infra/wazuh/config/wazuh_dashboard/wazuh.yml", "/usr/share/wazuh-dashboard/data/wazuh/config/wazuh.yml"),
]
FIELDS = [
    "check_id",
    "source_path",
    "target_path",
    "expected_type",
    "actual_type",
    "status",
    "message",
    "recommendation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/infra_diagnostics")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def path_type(path: Path) -> str:
    if path.is_file():
        return "file"
    if path.is_dir():
        return "directory"
    if path.exists():
        return "other"
    return "missing"


def recommendation_for(relative_path: str, actual_type: str) -> str:
    is_cert = "/certs/" in relative_path
    is_config = "/config/" in relative_path
    if actual_type == "directory":
        base = "A host oldali path könyvtárként létezik, miközben fájl mount szükséges. Kézzel ellenőrizd, állítsd le a stack-et, majd csak jóváhagyott karbantartási lépésként távolítsd el a könyvtárat és állítsd helyre valódi fájlként."
        if is_config:
            return base + " Verziókezelt konfigurációnál `git restore -- <path>` javasolható, ha a fájl Gitben szerepel."
        if is_cert:
            return base + " Tanúsítványnál ne használj nem valós lab tanúsítványt; futtasd a dokumentált Wazuh lab cert generálást."
        return base
    if actual_type == "missing":
        if is_config:
            return "Hiányzó konfigurációs fájl. Ha verziókezelt, `git restore -- <path>` használható; egyébként a Wazuh setup dokumentált template-je alapján kell létrehozni."
        if is_cert:
            return "Hiányzó tanúsítvány. Ne készíts kézzel pótlólagos mérési certet; futtasd a dokumentált Wazuh lab cert generálást vagy setup folyamatot."
        return "Hiányzó fájl. Állítsd helyre dokumentált forrásból."
    if actual_type == "other":
        return "A path létezik, de nem normál fájl. Ellenőrizd kézzel a Docker bind mount előtt."
    return ""


def check_mount_paths(root: Path, mounts: list[tuple[str, str]] | None = None) -> list[dict[str, str]]:
    mounts = mounts or EXPECTED_WAZUH_FILE_MOUNTS
    rows: list[dict[str, str]] = []
    for index, (source, target) in enumerate(mounts, start=1):
        actual = path_type(root / source)
        if actual == "file":
            status = "PASS"
            message = "Host oldali bind mount source létezik és fájl."
        elif actual == "directory":
            status = "FAIL"
            message = "Host oldali bind mount source könyvtár, de fájlnak kellene lennie."
        elif actual == "missing":
            status = "WARN"
            message = "Host oldali bind mount source hiányzik."
        else:
            status = "FAIL"
            message = "Host oldali bind mount source létezik, de nem normál fájl."
        rows.append(
            {
                "check_id": f"wazuh_bind_mount_{index:02d}",
                "source_path": source,
                "target_path": target,
                "expected_type": "file",
                "actual_type": actual,
                "status": status,
                "message": message,
                "recommendation": recommendation_for(source, actual),
            }
        )
    return rows


def aggregate_status(rows: list[dict[str, str]]) -> str:
    if any(row["status"] == "FAIL" for row in rows):
        return "FAIL"
    if any(row["status"] == "WARN" for row in rows):
        return "WARN"
    return "PASS"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    status = aggregate_status(rows)
    lines = [
        "# Wazuh bind mount diagnosztika",
        "",
        f"Összesített státusz: **{status}**",
        "",
        "Ez a riport csak diagnosztika. Nem töröl fájlt, nem generál tanúsítványt, nem hoz létre Wazuh alertet vagy mérési kimenetet.",
        "",
        "| Source | Target | Expected | Actual | Status | Message | Recommendation |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['source_path']}` | `{row['target_path']}` | {row['expected_type']} | {row['actual_type']} | "
            f"{row['status']} | {row['message']} | {row['recommendation']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(rows: list[dict[str, str]], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "wazuh_bind_mount_check.csv"
    md_path = output_dir / "wazuh_bind_mount_check.md"
    json_path = output_dir / "wazuh_bind_mount_check.json"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    json_path.write_text(
        json.dumps({"created_at": now_utc(), "status": aggregate_status(rows), "checks": rows}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return {"csv": csv_path, "markdown": md_path, "json": json_path}


def main() -> None:
    args = parse_args()
    outputs = write_outputs(check_mount_paths(Path(args.root)), Path(args.output_dir))
    for path in outputs.values():
        print(f"[OK] Kimenet: {path}")


if __name__ == "__main__":
    main()

