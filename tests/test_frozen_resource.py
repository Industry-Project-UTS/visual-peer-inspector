from __future__ import annotations

from pathlib import Path

import pytest

from visual_peer_inspector.frozen_resource import (
    publish_frozen_resource,
    verify_frozen_resource,
)


def test_minimal_frozen_resource_is_exact_and_reusable(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"status":"frozen"}\n', encoding="utf-8")
    root = tmp_path / "resource"
    first = publish_frozen_resource(
        resource_root=root,
        stage="method_release",
        files={"method_freeze.json": source},
        bindings={"method_sha256": "1" * 64},
    )
    second = publish_frozen_resource(
        resource_root=root,
        stage="method_release",
        files={"method_freeze.json": source},
        bindings={"method_sha256": "1" * 64},
    )
    assert first == second
    assert verify_frozen_resource(
        root,
        expected_stage="method_release",
        expected_files={"method_freeze.json"},
    ) == first


def test_frozen_resource_rejects_tampering_and_extra_files(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"status":"frozen"}\n', encoding="utf-8")
    root = tmp_path / "resource"
    publish_frozen_resource(
        resource_root=root,
        stage="method_release",
        files={"method_freeze.json": source},
        bindings={},
    )
    (root / "method_freeze.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        verify_frozen_resource(
            root,
            expected_stage="method_release",
            expected_files={"method_freeze.json"},
        )

    (root / "method_freeze.json").write_text(
        source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (root / "extra.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="allowlist"):
        verify_frozen_resource(
            root,
            expected_stage="method_release",
            expected_files={"method_freeze.json"},
        )
