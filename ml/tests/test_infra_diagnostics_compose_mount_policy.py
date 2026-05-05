from __future__ import annotations

from pathlib import Path

from ml.src.infra_diagnostics.check_docker_compose_mount_policy import check_compose_mount_policy, parse_bind_mounts


def write_compose(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        """
services:
  wazuh.indexer:
    image: wazuh/wazuh-indexer:4.14.4
    volumes:
      - ./wazuh/config/wazuh_indexer/wazuh.indexer.yml:/usr/share/wazuh-indexer/config/opensearch.yml
      - wazuh-indexer-data:/var/lib/wazuh-indexer
volumes:
  wazuh-indexer-data:
""",
        encoding="utf-8",
    )


def test_compose_mount_parser_megtalalja_indexer_opensearch_mountot(tmp_path: Path) -> None:
    compose = tmp_path / "infra/docker-compose.yml"
    write_compose(compose)

    mounts = parse_bind_mounts(compose)

    assert any(
        mount["service"] == "wazuh.indexer"
        and mount["source_raw"] == "./wazuh/config/wazuh_indexer/wazuh.indexer.yml"
        and mount["target_path"] == "/usr/share/wazuh-indexer/config/opensearch.yml"
        for mount in mounts
    )


def test_compose_mount_policy_directory_fail(tmp_path: Path) -> None:
    compose = tmp_path / "infra/docker-compose.yml"
    write_compose(compose)
    (tmp_path / "infra/wazuh/config/wazuh_indexer/wazuh.indexer.yml").mkdir(parents=True)

    rows = check_compose_mount_policy(tmp_path, Path("infra/docker-compose.yml"))
    target_rows = [row for row in rows if row["target_path"] == "/usr/share/wazuh-indexer/config/opensearch.yml"]

    assert target_rows[0]["status"] == "FAIL"
    assert target_rows[0]["actual_type"] == "directory"
