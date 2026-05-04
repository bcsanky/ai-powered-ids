from __future__ import annotations

import pandas as pd

from ml.src.live_smoke import check_make_workflow_dry_run
from ml.src.live_smoke.common import CommandResult


def test_make_dry_run_uses_make_n_without_running_targets(monkeypatch, tmp_path):
    (tmp_path / "Makefile").write_text("final-acceptance:\n\t@echo ok\n", encoding="utf-8")
    calls = []

    def fake_run(args, timeout_seconds=30, cwd=None):  # noqa: ANN001, ANN003
        calls.append(args)
        return CommandResult(returncode=0, stdout="\n".join(args), stderr="")

    monkeypatch.setattr(check_make_workflow_dry_run, "run_command", fake_run)

    result = check_make_workflow_dry_run.run_check(tmp_path / "reports/live_smoke", tmp_path / "Makefile")
    summary = pd.read_csv(result["csv"])

    assert "FAIL" not in set(summary["status"])
    assert calls
    assert all(call[:2] == ["make", "-n"] for call in calls)

