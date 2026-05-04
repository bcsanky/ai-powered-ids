from __future__ import annotations

from ml.src.lab_session.generate_run_commands import generate_run_commands


def test_generate_run_commands_for_zeek(tmp_path):
    output = generate_run_commands("zeek", tmp_path / "reports/lab_session/run_commands.md")
    text = output.read_text(encoding="utf-8")

    assert "make lab-build-features-zeek" in text
    assert "make final-real-measurement-package-with-provenance" in text
    assert "nem hoz létre mérési inputot" in text


def test_generate_run_commands_for_flow_csv(tmp_path):
    output = generate_run_commands("flow-csv", tmp_path / "reports/lab_session/run_commands.md")
    text = output.read_text(encoding="utf-8")

    assert "make lab-build-features-flow-csv" in text
    assert "make final-live-integration" in text

