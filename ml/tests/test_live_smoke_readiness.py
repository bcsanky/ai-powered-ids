from __future__ import annotations

import json

from ml.src.live_smoke.generate_live_smoke_readiness import run_readiness


def write_check(path, status):  # noqa: ANN001
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "check_id,category,status,message,recommendation,path\n"
        f"x,test,{status},message,,\n",
        encoding="utf-8",
    )


def test_readiness_not_ready_when_any_fail(tmp_path):
    out = tmp_path / "reports/live_smoke"
    write_check(out / "docker_environment_check.csv", "FAIL")

    result = run_readiness(out)
    payload = json.loads(result["json"].read_text(encoding="utf-8"))

    assert payload["readiness_status"] == "NOT_READY"


def test_readiness_ready_with_warnings_when_only_warn(tmp_path):
    out = tmp_path / "reports/live_smoke"
    write_check(out / "real_input_paths_check.csv", "WARN")

    result = run_readiness(out)
    payload = json.loads(result["json"].read_text(encoding="utf-8"))

    assert payload["readiness_status"] == "READY_WITH_WARNINGS"

