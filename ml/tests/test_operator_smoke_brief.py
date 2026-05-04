from __future__ import annotations

import json

from ml.src.live_smoke.generate_operator_smoke_brief import run_brief


def test_operator_brief_contains_required_commands_without_dummy_generation(tmp_path):
    out = tmp_path / "reports/live_smoke"
    out.mkdir(parents=True)
    (out / "live_smoke_readiness.json").write_text(
        json.dumps({"readiness_status": "READY_WITH_WARNINGS"}),
        encoding="utf-8",
    )

    result = run_brief(out)
    text = result["markdown"].read_text(encoding="utf-8")

    assert "make lab-session-prep" in text
    assert "make final-real-measurement-package-with-provenance" in text
    assert "dummy" not in text.lower()

