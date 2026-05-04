from __future__ import annotations

import json

from ml.src.live_smoke import check_opensearch_connection


class Response:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def test_opensearch_check_does_not_write_password_to_report(monkeypatch, tmp_path):
    def fake_get(url, **kwargs):  # noqa: ANN001, ANN003
        if url.endswith("_count"):
            return Response(200, {"count": 3})
        return Response(200, {"cluster_name": "lab"})

    monkeypatch.setattr(check_opensearch_connection.requests, "get", fake_get)

    result = check_opensearch_connection.run_check(
        tmp_path / "reports/live_smoke",
        opensearch_url="https://localhost:9200",
        index_pattern="wazuh-alerts-*",
        username="admin",
        password="secret-password",
        verify_tls=False,
    )
    text = result["markdown"].read_text(encoding="utf-8")
    payload = json.loads(result["json"].read_text(encoding="utf-8"))

    assert "secret-password" not in text
    assert payload["password_recorded"] is False

