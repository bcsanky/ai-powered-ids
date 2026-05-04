from __future__ import annotations

import pandas as pd

from ml.src.live_smoke import check_docker_environment
from ml.src.live_smoke.common import CommandResult


def test_docker_environment_warns_when_stack_not_running(monkeypatch, tmp_path):
    compose = tmp_path / "infra/docker-compose.yml"
    compose.parent.mkdir(parents=True)
    compose.write_text("services: {}\n", encoding="utf-8")

    def fake_run(args, timeout_seconds=20, cwd=None):  # noqa: ANN001, ANN003
        if args[:2] == ["docker", "--version"]:
            return CommandResult(0, "Docker version test", "")
        if args[:3] == ["docker", "compose", "version"]:
            return CommandResult(0, "Docker Compose version test", "")
        if "config" in args:
            return CommandResult(0, "name: test", "")
        if args[:2] == ["docker", "ps"]:
            return CommandResult(0, "", "")
        return CommandResult(1, "", "unexpected")

    monkeypatch.setattr(check_docker_environment, "run_command", fake_run)

    result = check_docker_environment.run_check(tmp_path / "reports/live_smoke", compose)
    summary = pd.read_csv(result["csv"])

    assert "FAIL" not in set(summary["status"])
    assert "WARN" in set(summary["status"])
