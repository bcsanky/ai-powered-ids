from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT = Path("dist/submission/ai_powered_ids_runtime_bundle.zip")
DEFAULT_REPORT_DIR = Path("reports/runtime_bundle")
MANIFEST_NAME = "runtime_bundle_manifest.csv"
REPORT_NAME = "runtime_bundle_report.md"
INSPECTION_NAME = "runtime_bundle_inspection.md"


@dataclass(frozen=True)
class BundleSpec:
    path: str
    category: str
    note: str = ""


REQUIRED_SPECS = [
    BundleSpec("README.md", "project"),
    BundleSpec("Makefile", "project"),
    BundleSpec("requirements-dev.txt", "project"),
    BundleSpec("ml/src", "source"),
    BundleSpec("ml/tests", "tests"),
    BundleSpec("experiments/final/ae_minimal.yaml", "config"),
    BundleSpec("experiments/final/ae_context.yaml", "config"),
    BundleSpec("docs", "documentation"),
    BundleSpec("templates/lab", "lab_templates"),
    BundleSpec("artifacts/final/final-ae-minimal-v1", "model_artifact"),
    BundleSpec("data/processed/final/ae_minimal/preprocess.pkl", "model_artifact"),
    BundleSpec("reports/real_measurement", "real_measurement_report"),
    BundleSpec("reports/real_measurement_qa", "qa_report"),
    BundleSpec("reports/measurement_quality", "quality_report"),
    BundleSpec("results/wazuh_real", "real_result"),
    BundleSpec("results/ae_lab", "real_result"),
    BundleSpec("results/hybrid_real", "real_result"),
    BundleSpec("results/real_comparison", "real_result"),
    BundleSpec("data/lab/lab_ground_truth.csv", "runtime_input"),
    BundleSpec("data/lab/lab_features.csv", "runtime_input"),
    BundleSpec("data/lab/zeek/conn.log", "runtime_input"),
    BundleSpec("data/lab/flows.csv", "runtime_input", "Optional flow-CSV input; Zeek conn.log is the measured input when this is absent."),
    BundleSpec("data/wazuh/alerts.jsonl", "runtime_input"),
    BundleSpec("raw/wazuh_export.json", "runtime_input", "Optional raw manual Wazuh export; normalized alerts.jsonl is sufficient when this is absent."),
]


MANIFEST_FIELDS = ["path", "size_bytes", "sha256", "category", "included", "note"]
INSPECTION_FIELDS = ["check_id", "category", "status", "message", "recommendation"]
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".pem", ".key", ".crt", ".p12", ".jks"}
FORBIDDEN_DIR_PARTS = {"__pycache__", ".pytest_cache", ".venv", ".venv39"}
PRIVATE_MATERIAL_RE = re.compile(rb"-----BEGIN [A-Z ]*(PRIVATE KEY|CERTIFICATE)-----")
CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"""(?ix)
    \b(password|passwd|secret|token|api[_-]?key|private[_-]?key)\b
    \s*[:=]\s*
    ['"]?
    ([^'"\s,#}\]]+)
    """
)
BEARER_RE = re.compile(r"(?i)\bauthorization\s*:\s*bearer\s+([A-Za-z0-9._~+/=-]{12,})")
PLACEHOLDER_VALUES = {
    "",
    "none",
    "null",
    "false",
    "true",
    "changeme",
    "change-me",
    "placeholder",
    "redacted",
    "REDACTED",
    "x",
    "secret-password",
    "test-token",
    "dummy-token",
    "dummy-password",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "inspect", "report"], nargs="?", default="build")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_parts(path: str | Path) -> set[str]:
    return {part.lower() for part in Path(path).as_posix().split("/")}


def excluded_reason(relative_path: str) -> str | None:
    lower = relative_path.lower()
    name = Path(relative_path).name
    suffix = Path(relative_path).suffix.lower()
    parts = path_parts(relative_path)
    if FORBIDDEN_DIR_PARTS & parts:
        return "excluded cache/virtualenv directory"
    if suffix in FORBIDDEN_SUFFIXES:
        return f"excluded forbidden suffix {suffix}"
    if name == ".env" or name.startswith(".env."):
        return "excluded environment file"
    if lower.startswith("infra/wazuh/certs/"):
        return "excluded Wazuh certificate path"
    if lower.startswith("infra/wazuh/") and "/data/" in lower:
        return "excluded Wazuh runtime data path"
    return None


def iter_files(root: Path, rel_path: str) -> Iterable[Path]:
    path = root / rel_path
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from sorted(candidate for candidate in path.rglob("*") if candidate.is_file())


def normalize_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def manifest_row(path: str, size: int | str, digest: str, category: str, included: bool, note: str) -> dict[str, Any]:
    return {
        "path": path,
        "size_bytes": size,
        "sha256": digest,
        "category": category,
        "included": str(included).lower(),
        "note": note,
    }


def collect_manifest(root: Path) -> tuple[list[dict[str, Any]], list[tuple[Path, str]], list[str], list[str]]:
    rows: list[dict[str, Any]] = []
    files: list[tuple[Path, str]] = []
    missing: list[str] = []
    excluded: list[str] = []
    seen: set[str] = set()

    for spec in REQUIRED_SPECS:
        source = root / spec.path
        if not source.exists():
            note = "MISSING: not packaged"
            if spec.note:
                note += f"; {spec.note}"
            rows.append(manifest_row(spec.path, "", "", spec.category, False, note))
            missing.append(spec.path)
            continue
        for file_path in iter_files(root, spec.path):
            rel = normalize_relative(file_path, root)
            if rel in seen:
                continue
            seen.add(rel)
            reason = excluded_reason(rel)
            size = file_path.stat().st_size
            if reason:
                rows.append(manifest_row(rel, size, "", spec.category, False, reason))
                excluded.append(rel)
                continue
            digest = sha256_file(file_path)
            rows.append(manifest_row(rel, size, digest, spec.category, True, spec.note))
            files.append((file_path, rel))
    return rows, files, missing, excluded


def looks_placeholder(value: str) -> bool:
    stripped = value.strip().strip("'\"),")
    normalized = stripped.replace("\\n", "").replace("\\r", "")
    if stripped in PLACEHOLDER_VALUES:
        return True
    if stripped.lower() in PLACEHOLDER_VALUES:
        return True
    if normalized in PLACEHOLDER_VALUES or normalized.lower() in PLACEHOLDER_VALUES:
        return True
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?(\(\))?", stripped):
        return True
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*\(\)", stripped):
        return True
    if stripped.startswith(("args.", "self.", "row.", "result.", "os.environ", "Path(")):
        return True
    return stripped.startswith(("<", "${", "$", "%", "YOUR_", "REPLACE_"))


def sensitive_findings_for_bytes(relative_path: str, data: bytes) -> list[str]:
    findings: list[str] = []
    if PRIVATE_MATERIAL_RE.search(data):
        findings.append(f"{relative_path}: private key or certificate block")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return findings
    for line_no, line in enumerate(text.splitlines(), start=1):
        bearer = BEARER_RE.search(line)
        if bearer and not looks_placeholder(bearer.group(1)):
            findings.append(f"{relative_path}:{line_no}: bearer token-like value")
        for match in CREDENTIAL_ASSIGNMENT_RE.finditer(line):
            value = match.group(2)
            if not looks_placeholder(value):
                findings.append(f"{relative_path}:{line_no}: credential-like assignment for {match.group(1)}")
    return findings


def scan_sensitive_files(files: list[tuple[Path, str]]) -> list[str]:
    findings: list[str] = []
    for path, rel in files:
        findings.extend(sensitive_findings_for_bytes(rel, path.read_bytes()))
    return findings


def scan_sensitive_zip(zip_path: Path) -> list[str]:
    findings: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in sorted(zf.namelist()):
            findings.extend(sensitive_findings_for_bytes(name, zf.read(name)))
    return findings


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def human_size(size: int) -> str:
    value = float(size)
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        values = [str(row.get(field, "")).replace("|", "\\|").replace("\n", " ") for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def required_status_rows(root: Path, names_in_zip: set[str] | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in REQUIRED_SPECS:
        source = root / spec.path
        present = source.exists()
        if names_in_zip is None:
            included = present
        elif source.is_dir():
            prefix = spec.path.rstrip("/") + "/"
            included = any(name.startswith(prefix) for name in names_in_zip)
        else:
            included = spec.path in names_in_zip
        status = "OK" if present and included else "MISSING" if not present else "NOT_IN_ZIP"
        rows.append({"path": spec.path, "category": spec.category, "status": status, "note": spec.note})
    return rows


def write_report(
    *,
    root: Path,
    output_zip: Path,
    report_dir: Path,
    manifest_rows: list[dict[str, Any]],
    missing: list[str],
    excluded: list[str],
    sensitive_findings: list[str],
    created_at: str,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    included_rows = [row for row in manifest_rows if row["included"] == "true"]
    total_size = sum(int(row["size_bytes"]) for row in included_rows if str(row["size_bytes"]).isdigit())
    status = "FAIL" if sensitive_findings else "WARN" if missing else "PASS"
    required_rows = required_status_rows(root)
    lines = [
        "# Runtime reproduction bundle riport",
        "",
        f"Státusz: **{status}**",
        f"Csomag neve: `{output_zip.as_posix()}`",
        f"Létrehozás ideje: `{created_at}`",
        f"Összes becsomagolt fájl: **{len(included_rows)}**",
        f"Becsült összméret: **{human_size(total_size)}**",
        "",
        "## Kötelező elemek státusza",
        "",
        markdown_table(required_rows, ["path", "category", "status", "note"]),
        "",
        "## Hiányzó elemek",
        "",
    ]
    if missing:
        lines.extend(f"- `{path}`" for path in missing)
    else:
        lines.append("- Nincs hiányzó kötelező elem.")
    lines.extend(
        [
            "",
            "## Kizárt elemek összefoglalása",
            "",
            f"- Kizárt fájlok száma: `{len(excluded)}`",
        ]
    )
    if excluded:
        lines.extend(f"- `{path}`" for path in excluded[:20])
        if len(excluded) > 20:
            lines.append(f"- ... további {len(excluded) - 20} fájl")
    lines.extend(["", "## Érzékenyadat-ellenőrzés", ""])
    if sensitive_findings:
        lines.append("A csomagolás megállt, mert érzékeny adatnak tűnő tartalom található:")
        lines.extend(f"- {finding}" for finding in sensitive_findings)
    else:
        lines.append("Nem találtam jelszó-, token-, privátkulcs- vagy tanúsítványtartalomra utaló mintát.")
    lines.extend(
        [
            "",
            "## Reprodukciós korlátok",
            "",
            "- A csomag reprodukciós és újraellenőrzési célú, nem publikus forrás-only beadási csomag.",
            "- A raw Wazuh/Zeek/lab bemenetek a runtime ellenőrzés miatt kerülnek bele.",
            "- A csomag nem telepíti újra automatikusan a teljes Wazuh, Zeek vagy VirtualBox labor infrastruktúrát.",
            "- Hiányzó elem esetén nem készül kézzel pótolt mérési adat; a hiány a manifestben és ebben a riportban szerepel.",
            "",
            "## Javasolt futtatási parancsok",
            "",
            "```bash",
            "python -m venv .venv",
            "source .venv/bin/activate",
            "pip install -r requirements-dev.txt",
            "python -m py_compile $(find ml/src -name \"*.py\")",
            "python -m pytest ml/tests",
            "make wazuh-parse-alerts",
            "make wazuh-correlate",
            "make wazuh-eval-real",
            "make lab-ae-validate-features",
            "make lab-ae-score",
            "make lab-ae-eval",
            "make hybrid-real-eval",
            "make real-compare",
            "make real-plot",
            "make final-real-measurement-thesis-ready",
            "make final-measurement-quality MEASUREMENT_QUALITY_THRESHOLDS=docs/measurement_quality_thresholds_real_lab_100.yaml",
            "```",
        ]
    )
    path = report_dir / REPORT_NAME
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        report_dir / "runtime_bundle_report.json",
        {
            "status": status,
            "zip_path": output_zip.as_posix(),
            "created_at": created_at,
            "included_file_count": len(included_rows),
            "total_size_bytes": total_size,
            "missing": missing,
            "excluded_count": len(excluded),
            "sensitive_findings": sensitive_findings,
        },
    )
    return path


def create_zip(output_zip: Path, files: list[tuple[Path, str]]) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for source, arcname in sorted(files, key=lambda item: item[1]):
            zf.write(source, arcname)


def build(root: Path, output_zip: Path, report_dir: Path) -> int:
    created_at = now_utc()
    manifest_rows, files, missing, excluded = collect_manifest(root)
    sensitive_findings = scan_sensitive_files(files)
    write_csv(report_dir / MANIFEST_NAME, manifest_rows, MANIFEST_FIELDS)
    if sensitive_findings:
        write_report(
            root=root,
            output_zip=output_zip,
            report_dir=report_dir,
            manifest_rows=manifest_rows,
            missing=missing,
            excluded=excluded,
            sensitive_findings=sensitive_findings,
            created_at=created_at,
        )
        return 1
    create_zip(output_zip, files)
    write_report(
        root=root,
        output_zip=output_zip,
        report_dir=report_dir,
        manifest_rows=manifest_rows,
        missing=missing,
        excluded=excluded,
        sensitive_findings=[],
        created_at=created_at,
    )
    metadata = {
        "created_at": created_at,
        "zip_path": output_zip.as_posix(),
        "zip_sha256": sha256_file(output_zip),
        "manifest_path": (report_dir / MANIFEST_NAME).as_posix(),
        "report_path": (report_dir / REPORT_NAME).as_posix(),
        "included_file_count": len(files),
        "missing_count": len(missing),
    }
    write_json(report_dir / "runtime_bundle_metadata.json", metadata)
    print(f"[OK] Runtime bundle: {output_zip}")
    if missing:
        print("[WARN] Missing runtime inputs are documented in the report.")
    return 0


def check_row(check_id: str, category: str, status: str, message: str, recommendation: str = "") -> dict[str, Any]:
    return {
        "check_id": check_id,
        "category": category,
        "status": status,
        "message": message,
        "recommendation": recommendation,
    }


def inspect(root: Path, output_zip: Path, report_dir: Path) -> int:
    rows: list[dict[str, Any]] = []
    if not output_zip.exists():
        rows.append(check_row("zip_exists", "zip", "FAIL", f"Hiányzó ZIP: {output_zip}", "Futtasd: make runtime-bundle"))
        write_inspection(report_dir, rows)
        return 1
    rows.append(check_row("zip_exists", "zip", "PASS", f"ZIP elérhető: {output_zip}"))
    with zipfile.ZipFile(output_zip, "r") as zf:
        names = sorted(zf.namelist())
    name_set = set(names)
    rows.append(check_row("zip_file_count", "zip", "PASS", f"Becsomagolt fájlok száma: {len(names)}"))

    forbidden = [name for name in names if excluded_reason(name)]
    rows.append(
        check_row(
            "forbidden_paths",
            "security",
            "FAIL" if forbidden else "PASS",
            "Tiltott fájl a ZIP-ben: " + ", ".join(forbidden[:10]) if forbidden else "Nincs tiltott cache/env/key/cert útvonal a ZIP-ben.",
            "Távolítsd el a tiltott fájlokat." if forbidden else "",
        )
    )
    sensitive = scan_sensitive_zip(output_zip)
    rows.append(
        check_row(
            "sensitive_content",
            "security",
            "FAIL" if sensitive else "PASS",
            "Érzékeny tartalomgyanú: " + "; ".join(sensitive[:5]) if sensitive else "Nem találtam jelszó/token/private key/cert mintát.",
            "Állítsd meg a beadást és tisztítsd a csomagot." if sensitive else "",
        )
    )
    for spec in REQUIRED_SPECS:
        source = root / spec.path
        if source.is_dir():
            present = any(name.startswith(spec.path.rstrip("/") + "/") for name in name_set)
        else:
            present = spec.path in name_set
        if present:
            status = "PASS"
            message = f"{spec.path} szerepel a ZIP-ben."
        elif source.exists():
            status = "FAIL"
            message = f"{spec.path} létezik, de nem szerepel a ZIP-ben."
        else:
            status = "WARN"
            message = f"{spec.path} nem létezik a forrásfában, ezért nem került ZIP-be."
        rows.append(check_row(f"required_{spec.path.replace('/', '_')}", "required_content", status, message, spec.note))
    write_inspection(report_dir, rows)
    if any(row["status"] == "FAIL" for row in rows):
        return 1
    print(f"[OK] Runtime bundle inspection: {report_dir / INSPECTION_NAME}")
    return 0


def write_inspection(report_dir: Path, rows: list[dict[str, Any]]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    status = "FAIL" if any(row["status"] == "FAIL" for row in rows) else "WARN" if any(row["status"] == "WARN" for row in rows) else "PASS"
    write_csv(report_dir / "runtime_bundle_inspection.csv", rows, INSPECTION_FIELDS)
    lines = [
        "# Runtime bundle inspection",
        "",
        f"Összesített státusz: **{status}**",
        "",
        markdown_table(rows, INSPECTION_FIELDS),
    ]
    (report_dir / INSPECTION_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report_dir / "runtime_bundle_inspection.json", {"status": status, "checks": rows})


def report(root: Path, output_zip: Path, report_dir: Path) -> int:
    manifest_path = report_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return build(root, output_zip, report_dir)
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    missing = [row["path"] for row in rows if str(row.get("note", "")).startswith("MISSING")]
    excluded = [row["path"] for row in rows if row.get("included") == "false" and not str(row.get("note", "")).startswith("MISSING")]
    write_report(
        root=root,
        output_zip=output_zip,
        report_dir=report_dir,
        manifest_rows=rows,
        missing=missing,
        excluded=excluded,
        sensitive_findings=[],
        created_at=now_utc(),
    )
    print(f"[OK] Runtime bundle report: {report_dir / REPORT_NAME}")
    return 0


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_zip = Path(args.output)
    report_dir = Path(args.report_dir)
    if args.command == "build":
        raise SystemExit(build(root, output_zip, report_dir))
    if args.command == "inspect":
        raise SystemExit(inspect(root, output_zip, report_dir))
    if args.command == "report":
        raise SystemExit(report(root, output_zip, report_dir))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
