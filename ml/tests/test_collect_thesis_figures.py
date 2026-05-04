from __future__ import annotations

import pandas as pd

from ml.src.collect_thesis_figures import collect_figures


def test_collect_figures_writes_manifest_when_sources_are_missing(tmp_path):
    manifest = collect_figures(
        output_dir=tmp_path / "figures",
        ae_minimal_root=tmp_path / "missing_ae_min",
        ae_context_root=tmp_path / "missing_ae_ctx",
        comparison_dir=tmp_path / "missing_comparison",
        lab_dir=tmp_path / "missing_lab",
        performance_dir=tmp_path / "missing_performance",
    )

    manifest_path = tmp_path / "figures" / "figure_manifest.csv"
    assert manifest_path.exists()
    assert set(manifest["status"]).issubset({"copied", "missing"})
    assert (manifest["status"] == "missing").all()


def test_collect_figures_copies_existing_files(tmp_path):
    ae_root = tmp_path / "ae"
    ae_run = ae_root / "ae_v1_test"
    comparison = tmp_path / "comparison"
    lab = tmp_path / "lab"
    ae_run.mkdir(parents=True)
    comparison.mkdir()
    lab.mkdir()

    (ae_run / "confusion_matrix.png").write_bytes(b"png")
    (comparison / "fig_comparison_precision_recall_f1.png").write_bytes(b"png")
    (lab / "lab_timeline.png").write_bytes(b"png")

    manifest = collect_figures(
        output_dir=tmp_path / "figures",
        ae_minimal_root=ae_root,
        ae_context_root=tmp_path / "missing_ae_ctx",
        comparison_dir=comparison,
        lab_dir=lab,
        performance_dir=tmp_path / "missing_performance",
    )

    saved = pd.read_csv(tmp_path / "figures" / "figure_manifest.csv")
    assert "copied" in set(saved["status"])
    assert (tmp_path / "figures" / "ae_minimal_confusion_matrix.png").exists()
    assert manifest.loc[manifest["figure_file"].str.endswith("lab_timeline.png"), "status"].iloc[0] == "copied"
