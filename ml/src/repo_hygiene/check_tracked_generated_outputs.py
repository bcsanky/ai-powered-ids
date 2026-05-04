from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ml.src.repo_hygiene.common import is_forbidden_tracked_generated_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/repo_hygiene")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def list_tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def find_forbidden_tracked_outputs(tracked_files: list[str]) -> list[str]:
    return sorted(path for path in tracked_files if is_forbidden_tracked_generated_output(path))


def write_markdown(status: str, forbidden: list[str], output_path: Path) -> None:
    lines = [
        "# Verziózott generált kimenetek ellenőrzése",
        "",
        f"Státusz: **{status}**",
        "",
    ]
    if forbidden:
        lines.extend(
            [
                "A következő futási kimenetek nem maradhatnak Gitben:",
                "",
                "| Fájl | Teendő |",
                "| --- | --- |",
            ]
        )
        for path in forbidden:
            lines.append(f"| `{path}` | remove_from_git |")
    else:
        lines.append("Nem található tiltott verziózott generált kimenet.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_tracked_generated_outputs(
    *,
    root: Path,
    output_dir: Path,
    tracked_files: list[str] | None = None,
) -> dict[str, Any]:
    tracked = tracked_files if tracked_files is not None else list_tracked_files(root)
    forbidden = find_forbidden_tracked_outputs(tracked)
    status = "FAIL" if forbidden else "PASS"
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "tracked_generated_outputs_check.md"
    json_path = output_dir / "tracked_generated_outputs_check.json"
    write_markdown(status, forbidden, md_path)
    payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "forbidden_tracked_outputs": forbidden,
        "forbidden_count": len(forbidden),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"status": status, "forbidden": forbidden, "markdown": md_path, "json": json_path}


def main() -> None:
    args = parse_args()
    result = check_tracked_generated_outputs(root=Path(args.root), output_dir=Path(args.output_dir))
    print(f"[OK] Kimenet: {result['markdown']}")
    print(f"[OK] Kimenet: {result['json']}")
    if result["status"] == "FAIL":
        print("Tiltott verziózott generált kimenetek találhatók:")
        for path in result["forbidden"]:
            print(f"- {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()

