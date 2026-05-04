from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib

from ml.src.live_smoke.common import check_row, has_fail, write_check_outputs
from ml.src.repo_hygiene.common import sha256_file
from ml.src.scoring_runtime import resolve_scoring_paths, select_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", default="artifacts/final/final-ae-minimal-v1")
    parser.add_argument("--preprocess", default="data/processed/final/ae_minimal/preprocess.pkl")
    parser.add_argument("--output-dir", default="reports/live_smoke")
    return parser.parse_args()


def run_check(output_dir: Path, model_root: Path, preprocess: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    hashes: dict[str, str] = {}
    paths = None

    rows.append(
        check_row(
            "model_root_exists",
            "Modellállományok",
            "PASS" if model_root.exists() else "FAIL",
            f"modellgyökér elérhető: {model_root}" if model_root.exists() else f"hiányzó modellgyökér: {model_root}",
            "" if model_root.exists() else "A végleges AE-Minimal modellkönyvtár szükséges az élő scoringhoz.",
            model_root.as_posix(),
        )
    )
    rows.append(
        check_row(
            "preprocess_exists",
            "Modellállományok",
            "PASS" if preprocess.exists() else "FAIL",
            f"preprocess elérhető: {preprocess}" if preprocess.exists() else f"hiányzó preprocess: {preprocess}",
            "" if preprocess.exists() else "A preprocess.pkl szükséges az AE-Minimal feature transzformációhoz.",
            preprocess.as_posix(),
        )
    )

    if model_root.exists():
        try:
            paths = resolve_scoring_paths(model_root, preprocess)
            rows.append(check_row("model_run_resolved", "Modellállományok", "PASS", f"futtatási könyvtár: {paths.run_dir}", path=paths.run_dir.as_posix()))
        except Exception as exc:  # noqa: BLE001
            rows.append(
                check_row(
                    "model_run_resolved",
                    "Modellállományok",
                    "FAIL",
                    f"nem oldható fel használható modellfuttatás: {exc}",
                    "Ellenőrizd, hogy van-e model.joblib és thresholds.json a modellgyökér alatt.",
                    model_root.as_posix(),
                )
            )

    if paths is not None:
        for check_id, path, label in [
            ("model_joblib_exists", paths.model_path, "model.joblib"),
            ("thresholds_exists", paths.thresholds_path, "thresholds.json"),
            ("preprocess_resolved_exists", paths.preprocess_path, "preprocess.pkl"),
        ]:
            rows.append(
                check_row(
                    check_id,
                    "Modellállományok",
                    "PASS" if path.exists() else "FAIL",
                    f"{label} elérhető" if path.exists() else f"{label} hiányzik",
                    "" if path.exists() else "A fájlt pótolni kell scoring előtt.",
                    path.as_posix(),
                )
            )

        if paths.model_path.exists():
            try:
                joblib.load(paths.model_path)
                hashes["model_joblib_sha256"] = sha256_file(paths.model_path)
                rows.append(check_row("model_joblib_load", "Modellállományok", "PASS", "model.joblib betölthető", path=paths.model_path.as_posix()))
            except Exception as exc:  # noqa: BLE001
                rows.append(check_row("model_joblib_load", "Modellállományok", "FAIL", f"model.joblib nem tölthető be: {exc}", "Ellenőrizd a modellfájlt.", paths.model_path.as_posix()))

        if paths.preprocess_path.exists():
            try:
                joblib.load(paths.preprocess_path)
                hashes["preprocess_sha256"] = sha256_file(paths.preprocess_path)
                rows.append(check_row("preprocess_load", "Modellállományok", "PASS", "preprocess.pkl betölthető", path=paths.preprocess_path.as_posix()))
            except Exception as exc:  # noqa: BLE001
                rows.append(check_row("preprocess_load", "Modellállományok", "FAIL", f"preprocess.pkl nem tölthető be: {exc}", "Ellenőrizd a preprocess fájlt.", paths.preprocess_path.as_posix()))

        if paths.thresholds_path.exists():
            try:
                thresholds = json.loads(paths.thresholds_path.read_text(encoding="utf-8"))
                choice = select_threshold(thresholds)
                hashes["thresholds_sha256"] = sha256_file(paths.thresholds_path)
                rows.append(
                    check_row(
                        "thresholds_parse",
                        "Modellállományok",
                        "PASS",
                        f"thresholds.json értelmezhető, kiválasztott küszöb: {choice.name}",
                        path=paths.thresholds_path.as_posix(),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(check_row("thresholds_parse", "Modellállományok", "FAIL", f"thresholds.json hibás: {exc}", "Ellenőrizd a küszöbfájlt.", paths.thresholds_path.as_posix()))

    return write_check_outputs(
        output_dir=output_dir,
        basename="model_artifacts_check",
        title="AE-Minimal modellállomány ellenőrzés",
        rows=rows,
        intro="Az ellenőrzés a modell- és preprocess fájlok betölthetőségét vizsgálja. Nem futtat scoringot bemeneti eseményen.",
        extra_payload={"hashes": hashes},
    )


def main() -> None:
    args = parse_args()
    result = run_check(Path(args.output_dir), Path(args.model_root), Path(args.preprocess))
    print(f"[OK] Kimenet: {result['markdown']}")
    if has_fail(result["rows"]):
        sys.exit(1)


if __name__ == "__main__":
    main()

