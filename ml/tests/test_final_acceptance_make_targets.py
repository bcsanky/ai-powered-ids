from __future__ import annotations

from ml.src.final_acceptance.check_make_targets import REQUIRED_TARGETS, run_check


def test_missing_makefile_target_fails(tmp_path):
    makefile = tmp_path / "Makefile"
    makefile.write_text("final-validate:\n\tpython -V\n", encoding="utf-8")

    result = run_check(makefile, tmp_path / "reports/final_acceptance")

    assert result["status"] == "FAIL"


def test_all_required_makefile_targets_pass(tmp_path):
    makefile = tmp_path / "Makefile"
    target_blocks = "\n".join(f"{target}:\n\t@true\n" for target in REQUIRED_TARGETS)
    makefile.write_text(
        target_blocks
        + "\nfinal-acceptance:\n"
        + "\t$(MAKE) final-acceptance-make-targets\n"
        + "\t$(MAKE) final-acceptance-failure-modes\n"
        + "\t$(MAKE) final-acceptance-provenance-policy\n"
        + "\t$(MAKE) final-acceptance-docs\n"
        + "\t$(MAKE) final-acceptance-runbook-docs\n"
        + "\t$(MAKE) repo-hygiene-check\n"
        + "\t$(MAKE) final-acceptance-readiness\n"
        + "\t$(MAKE) final-acceptance-brief\n",
        encoding="utf-8",
    )

    result = run_check(makefile, tmp_path / "reports/final_acceptance")

    assert result["status"] == "PASS"
