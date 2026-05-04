from __future__ import annotations

from ml.src.thesis_integration.check_thesis_inputs import run_check
from ml.tests.thesis_integration_fixture import create_thesis_measurement


def test_thesis_input_check_not_ready_without_provenance(tmp_path):
    create_thesis_measurement(tmp_path, with_provenance=False)

    result = run_check(tmp_path, tmp_path / "reports/thesis_integration")

    assert result["status"] == "NOT_READY"


def test_thesis_input_check_ready_with_valid_data(tmp_path):
    create_thesis_measurement(tmp_path)

    result = run_check(tmp_path, tmp_path / "reports/thesis_integration")

    assert result["status"] == "READY"

