from __future__ import annotations

from pathlib import Path

from ml.src.infra_diagnostics.check_wazuh_bind_mounts import check_mount_paths


def test_wazuh_mount_file_pass(tmp_path: Path) -> None:
    mount_dir = tmp_path / "infra/wazuh/config/wazuh_indexer"
    mount_dir.mkdir(parents=True)
    (mount_dir / "wazuh.indexer.yml").write_text("cluster.name: test\n", encoding="utf-8")

    rows = check_mount_paths(
        tmp_path,
        [
            (
                "infra/wazuh/config/wazuh_indexer/wazuh.indexer.yml",
                "/usr/share/wazuh-indexer/config/opensearch.yml",
            )
        ],
    )

    assert rows[0]["status"] == "PASS"
    assert rows[0]["actual_type"] == "file"


def test_wazuh_mount_directory_fail(tmp_path: Path) -> None:
    (tmp_path / "infra/wazuh/config/wazuh_indexer/wazuh.indexer.yml").mkdir(parents=True)

    rows = check_mount_paths(
        tmp_path,
        [
            (
                "infra/wazuh/config/wazuh_indexer/wazuh.indexer.yml",
                "/usr/share/wazuh-indexer/config/opensearch.yml",
            )
        ],
    )

    assert rows[0]["status"] == "FAIL"
    assert rows[0]["actual_type"] == "directory"


def test_wazuh_mount_missing_warn(tmp_path: Path) -> None:
    rows = check_mount_paths(
        tmp_path,
        [
            (
                "infra/wazuh/config/wazuh_indexer/wazuh.indexer.yml",
                "/usr/share/wazuh-indexer/config/opensearch.yml",
            )
        ],
    )

    assert rows[0]["status"] == "WARN"
    assert rows[0]["actual_type"] == "missing"
