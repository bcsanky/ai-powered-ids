from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.measurement_quality.common import (
    check_row,
    has_fail,
    is_forbidden_real_path,
    provenance_row,
    read_csv_optional,
    read_yaml,
    to_numeric_series,
    validate_verified_provenance,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", default="data/lab/lab_ground_truth.csv")
    parser.add_argument("--lab-features", default="data/lab/lab_features.csv")
    parser.add_argument("--wazuh-predictions", default="results/wazuh_real/predictions.csv")
    parser.add_argument("--ae-predictions", default="results/ae_lab/predictions.csv")
    parser.add_argument("--hybrid-predictions", default="results/hybrid_real/predictions.csv")
    parser.add_argument("--thresholds", default="docs/measurement_quality_thresholds.yaml")
    parser.add_argument("--provenance", default="reports/real_measurement/measurement_provenance.json")
    parser.add_argument("--output-dir", default="reports/measurement_quality")
    return parser.parse_args()


def id_set(df: pd.DataFrame | None) -> set[str]:
    if df is None or df.empty or "event_id" not in df.columns:
        return set()
    return set(df["event_id"].astype(str))


def set_row(check_id: str, expected: set[str], actual: set[str], label: str) -> dict[str, Any]:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    ok = not missing and not extra
    return check_row(
        check_id,
        "Event alignment",
        "PASS" if ok else "FAIL",
        f"{label} event_id készlet egyezik" if ok else f"{label} eltérés: missing={missing[:5]}, extra={extra[:5]}",
        "" if ok else "Ugyanazon ground truth eseményhalmazra kell futtatni minden komponenst.",
        f"missing={len(missing)}, extra={len(extra)}",
    )


def run_check(
    *,
    ground_truth: Path,
    lab_features: Path,
    wazuh_predictions: Path,
    ae_predictions: Path,
    hybrid_predictions: Path,
    thresholds_path: Path,
    provenance_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    thresholds = read_yaml(thresholds_path)
    max_missing_ae = float(thresholds.get("minimums", {}).get("max_missing_ae_score_ratio", 0.1))
    max_unmatched = float(thresholds.get("minimums", {}).get("max_unmatched_alert_ratio", 0.5))
    rows: list[dict[str, Any]] = [provenance_row(provenance_path)]
    _, _, provenance = validate_verified_provenance(provenance_path)

    for label, path in [
        ("ground_truth", ground_truth),
        ("lab_features", lab_features),
    ]:
        rows.append(
            check_row(
                f"{label}_path_guard",
                "Adateredet",
                "FAIL" if is_forbidden_real_path(path) else "PASS",
                f"{label} útvonal {'tiltott' if is_forbidden_real_path(path) else 'elfogadható'}",
                "Valós lab input útvonal szükséges." if is_forbidden_real_path(path) else "",
                path.as_posix(),
            )
        )
    if provenance:
        for key, path in [("ground_truth_path", ground_truth), ("lab_features_path", lab_features)]:
            expected_path = str(provenance.get(key, ""))
            rows.append(
                check_row(
                    f"provenance_{key}",
                    "Adateredet",
                    "PASS" if expected_path == path.as_posix() else "WARN",
                    "útvonal egyezik a provenance-szel" if expected_path == path.as_posix() else f"eltér a provenance-től: {expected_path}",
                    "Ellenőrizd, hogy ugyanazt a mérési inputot használod-e.",
                    path.as_posix(),
                )
            )

    frames = {
        "ground_truth": read_csv_optional(ground_truth),
        "lab_features": read_csv_optional(lab_features),
        "wazuh_predictions": read_csv_optional(wazuh_predictions),
        "ae_predictions": read_csv_optional(ae_predictions),
        "hybrid_predictions": read_csv_optional(hybrid_predictions),
    }
    for name, df in frames.items():
        rows.append(
            check_row(
                f"{name}_readable",
                "Bemenet",
                "PASS" if df is not None and not df.empty else "FAIL",
                f"{name} olvasható" if df is not None and not df.empty else f"{name} hiányzik vagy üres",
                "Futtasd a real-lab pipeline megfelelő lépését.",
                0 if df is None else len(df),
            )
        )

    gt_ids = id_set(frames["ground_truth"])
    if gt_ids:
        rows.extend(
            [
                set_row("features_event_set", gt_ids, id_set(frames["lab_features"]), "lab_features"),
                set_row("wazuh_event_set", gt_ids, id_set(frames["wazuh_predictions"]), "Wazuh predictions"),
                set_row("ae_event_set", gt_ids, id_set(frames["ae_predictions"]), "AE predictions"),
                set_row("hybrid_event_set", gt_ids, id_set(frames["hybrid_predictions"]), "Hybrid predictions"),
            ]
        )

    ae = frames["ae_predictions"]
    if ae is not None and not ae.empty:
        if "anomaly_score" in ae.columns:
            missing_ratio = float(to_numeric_series(ae, "anomaly_score").isna().mean())
            rows.append(
                check_row(
                    "missing_ae_score_ratio",
                    "AE scoring",
                    "PASS" if missing_ratio <= max_missing_ae else "FAIL",
                    f"hiányzó AE score arány: {missing_ratio:.4f}",
                    "A lab feature mappinget és AE scoring kimenetet ellenőrizni kell.",
                    missing_ratio,
                )
            )
        else:
            rows.append(check_row("missing_ae_score_ratio", "AE scoring", "FAIL", "anomaly_score oszlop hiányzik"))

    wazuh = frames["wazuh_predictions"]
    if wazuh is not None and not wazuh.empty and "y_true" in wazuh.columns and "wazuh_pred" in wazuh.columns:
        attacks = wazuh[to_numeric_series(wazuh, "y_true").fillna(0).astype(int) == 1]
        if attacks.empty:
            rows.append(check_row("attack_without_wazuh_ratio", "Wazuh matching", "FAIL", "nincs támadó esemény a Wazuh predikcióban"))
        else:
            no_alert_ratio = float((to_numeric_series(attacks, "wazuh_pred").fillna(0).astype(int) == 0).mean())
            rows.append(
                check_row(
                    "attack_without_wazuh_ratio",
                    "Wazuh matching",
                    "PASS" if no_alert_ratio <= max_unmatched else "WARN",
                    f"Wazuh alert nélküli attack események aránya: {no_alert_ratio:.4f}",
                    "Magas arány esetén a Wazuh export, időablak vagy IP-korreláció ellenőrzendő.",
                    no_alert_ratio,
                )
            )

    for name in ["ground_truth", "lab_features"]:
        df = frames[name]
        if df is not None and not df.empty:
            for col in ["source_ip", "target_ip"]:
                if col in df.columns:
                    filled = float(df[col].astype(str).str.strip().replace("nan", "").ne("").mean())
                    rows.append(
                        check_row(
                            f"{name}_{col}_filled",
                            "Mezőkitöltöttség",
                            "PASS" if filled >= 0.8 else "WARN",
                            f"{name}.{col} kitöltöttsége: {filled:.4f}",
                            "Alacsony kitöltöttség ronthatja az alert-event korrelációt.",
                            filled,
                        )
                    )

    return write_outputs(
        output_dir=output_dir,
        basename="feature_alert_alignment",
        title="Feature, alert és predikció alignment ellenőrzés",
        rows=rows,
        intro="A riport azt vizsgálja, hogy a ground truth, feature, Wazuh, AE és hibrid kimenetek ugyanarra az event_id készletre vonatkoznak-e.",
    )


def main() -> None:
    args = parse_args()
    result = run_check(
        ground_truth=Path(args.ground_truth),
        lab_features=Path(args.lab_features),
        wazuh_predictions=Path(args.wazuh_predictions),
        ae_predictions=Path(args.ae_predictions),
        hybrid_predictions=Path(args.hybrid_predictions),
        thresholds_path=Path(args.thresholds),
        provenance_path=Path(args.provenance),
        output_dir=Path(args.output_dir),
    )
    print(f"[OK] Kimenet: {result['markdown']}")
    if has_fail(result["rows"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
