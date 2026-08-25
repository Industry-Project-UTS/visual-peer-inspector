from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from visual_peer_inspector.data.model_safe_package import canonical_json_bytes, file_sha256
from visual_peer_inspector.evaluation.proposal_history import (
    resolve_committed_proposal_history,
)
from visual_peer_inspector.models.locateanything import (
    EXECUTION_POLICY_SHA256,
    LocateAnythingConfiguration,
    MODEL_PRODUCER_KIND,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _row(
    configuration: LocateAnythingConfiguration,
    *,
    scene: str,
    attempt: int,
    status: str,
    error_type: str | None = None,
) -> dict:
    mode = "hybrid" if attempt <= 4 else "slow"
    failed = status == "error"
    return {
        "schema_version": 1,
        "cohort_id": "test-ds1",
        "input_manifest_sha256": "a" * 64,
        "opaque_scene_id": scene,
        "view_id": "V00",
        "attempt": attempt,
        "image_sha256": ("1" if scene == "S000001" else "2") * 64,
        "configuration_sha256": configuration.sha256,
        "execution_policy_sha256": EXECUTION_POLICY_SHA256,
        "effective_generation_mode": mode,
        "effective_max_new_tokens": 8192,
        "producer_kind": MODEL_PRODUCER_KIND,
        "model_id": configuration.model_id,
        "model_revision": configuration.model_revision,
        "prompt": configuration.prompt,
        "proposal_count": 0 if failed else 1,
        "boxes_xyxy": [] if failed else [[10, 10, 80, 80]],
        "raw_response": "",
        "generation_stats": {
            "effective_generation_mode": mode,
            "effective_max_new_tokens": 8192,
        },
        "response_token_count": 0 if failed else 1,
        "possibly_truncated": error_type == "PossibleTruncation",
        "truncation_check": "token_count_heuristic",
        "runtime_seconds": 0.1,
        "status": status,
        "error_type": error_type if failed else None,
        "error_message": f"Sanitized {error_type}" if failed else None,
    }


def _resolve(path: Path, configuration: LocateAnythingConfiguration, digest: str):
    return resolve_committed_proposal_history(
        path,
        configuration_sha256=configuration.sha256,
        producer_kind=MODEL_PRODUCER_KIND,
        expected_file_sha256=digest,
        expected_result_count=2,
        expected_cohort_id="test-ds1",
        expected_input_manifest_sha256="a" * 64,
        expected_model_id=configuration.model_id,
        expected_model_revision=configuration.model_revision,
    )


def test_resolves_valid_retry_history_to_latest_successes(tmp_path):
    configuration = LocateAnythingConfiguration()
    history = [
        _row(
            configuration,
            scene="S000001",
            attempt=attempt,
            status="error",
            error_type="PossibleTruncation",
        )
        for attempt in range(1, 5)
    ]
    history.extend(
        [
            _row(configuration, scene="S000001", attempt=5, status="ok"),
            _row(configuration, scene="S000002", attempt=1, status="ok"),
        ]
    )
    path = tmp_path / "proposal_results.jsonl"
    _write_jsonl(path, history)

    resolved = _resolve(path, configuration, file_sha256(path))

    assert len(resolved.history_rows) == 6
    assert len(resolved.latest_rows) == 2
    assert [row["attempt"] for row in resolved.latest_rows] == [5, 1]
    assert all(row["status"] == "ok" for row in resolved.latest_rows)


def test_rejects_failed_latest_row(tmp_path):
    configuration = LocateAnythingConfiguration()
    rows = [
        _row(
            configuration,
            scene="S000001",
            attempt=1,
            status="error",
            error_type="NoValidBoxes",
        ),
        _row(configuration, scene="S000002", attempt=1, status="ok"),
    ]
    path = tmp_path / "proposal_results.jsonl"
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="latest rows"):
        _resolve(path, configuration, file_sha256(path))


def test_rejects_invalid_slow_fallback_transition(tmp_path):
    configuration = LocateAnythingConfiguration()
    rows = [
        _row(
            configuration,
            scene="S000001",
            attempt=attempt,
            status="error",
            error_type="NoValidBoxes",
        )
        for attempt in range(1, 5)
    ]
    rows.extend(
        [
            _row(configuration, scene="S000001", attempt=5, status="ok"),
            _row(configuration, scene="S000002", attempt=1, status="ok"),
        ]
    )
    path = tmp_path / "proposal_results.jsonl"
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="directly preceding"):
        _resolve(path, configuration, file_sha256(path))


def test_rejects_retry_after_success(tmp_path):
    configuration = LocateAnythingConfiguration()
    first = _row(configuration, scene="S000001", attempt=1, status="ok")
    retry = deepcopy(first)
    retry.update(
        {
            "attempt": 2,
            "status": "error",
            "proposal_count": 0,
            "boxes_xyxy": [],
            "possibly_truncated": True,
            "error_type": "PossibleTruncation",
            "error_message": "Sanitized PossibleTruncation",
        }
    )
    rows = [
        first,
        retry,
        _row(configuration, scene="S000002", attempt=1, status="ok"),
    ]
    path = tmp_path / "proposal_results.jsonl"
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="continues after success"):
        _resolve(path, configuration, file_sha256(path))


def test_rejects_tampered_history_hash(tmp_path):
    configuration = LocateAnythingConfiguration()
    rows = [
        _row(configuration, scene="S000001", attempt=1, status="ok"),
        _row(configuration, scene="S000002", attempt=1, status="ok"),
    ]
    path = tmp_path / "proposal_results.jsonl"
    _write_jsonl(path, rows)
    committed = file_sha256(path)
    rows[0]["boxes_xyxy"] = [[11, 11, 81, 81]]
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="committed file hash"):
        _resolve(path, configuration, committed)
