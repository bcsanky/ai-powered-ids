from __future__ import annotations

import pandas as pd
import requests

from ml.src.live_smoke import check_ml_service_health


class Response:
    status_code = 200

    @staticmethod
    def json():
        return {"status": "ok", "model_loaded": True, "model_version": "test"}


def test_ml_health_check_passes_on_200_json(monkeypatch, tmp_path):
    monkeypatch.setattr(check_ml_service_health.requests, "get", lambda *args, **kwargs: Response())

    result = check_ml_service_health.run_check(tmp_path / "reports/live_smoke", "http://service/health")
    summary = pd.read_csv(result["csv"])

    assert "FAIL" not in set(summary["status"])
    assert "ml_health_status_code" in set(summary["check_id"])


def test_ml_health_timeout_warns_when_not_required(monkeypatch, tmp_path):
    def raise_timeout(*args, **kwargs):  # noqa: ANN002, ANN003
        raise requests.Timeout("timeout")

    monkeypatch.setattr(check_ml_service_health.requests, "get", raise_timeout)

    result = check_ml_service_health.run_check(
        tmp_path / "reports/live_smoke",
        "http://service/health",
        required=False,
    )
    summary = pd.read_csv(result["csv"])

    assert set(summary["status"]) == {"WARN"}

