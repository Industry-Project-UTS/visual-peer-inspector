from __future__ import annotations

import importlib.util
import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from visual_peer_inspector.workflows.frozen_development_pipeline import (
    FrozenPipelineIdentity,
)
from visual_peer_inspector.workflows.official_inference import (
    LOCK_RESULT_SCHEMA,
    create_recovery_bundle,
    load_lock_result_receipt,
    restore_recovery_bundle,
)
from visual_peer_inspector.workflows.protected_cohort_inference import (
    run_exact_sharded_stage,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPOSITORY_ROOT / "scripts/run_official_inference.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("run_official_inference", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
official_runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(official_runner)


def _digest(text: str) -> str:
    return sha256(text.encode()).hexdigest()


def test_lock_result_requires_exact_method_policy_and_no_change(tmp_path: Path) -> None:
    method = _digest("method")
    policy = _digest("policy")
    path = tmp_path / "internal_lock_result_receipt.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": LOCK_RESULT_SCHEMA,
                "status": "protocol_complete",
                "decision_scope": "evaluation_completed_once_without_method_or_policy_change",
                "scientific_performance_accepted": False,
                "method_record_sha256": method,
                "policy_record_sha256": policy,
                "internal_lock_prediction_lock_sha256": _digest("prediction"),
                "internal_lock_evaluation_manifest_sha256": _digest("evaluation"),
                "method_or_policy_changed": False,
            }
        )
    )
    assert (
        load_lock_result_receipt(path, method_sha256=method, policy_sha256=policy)[
            "status"
        ]
        == "protocol_complete"
    )
    value = json.loads(path.read_text())
    value["method_or_policy_changed"] = True
    path.write_text(json.dumps(value))
    try:
        load_lock_result_receipt(path, method_sha256=method, policy_sha256=policy)
    except ValueError as error:
        assert "unchanged" in str(error)
    else:
        raise AssertionError("A changed post-lock method was accepted")


def test_full_run_publishes_a_role_specific_model_only_resource(
    tmp_path: Path,
) -> None:
    cohort_id = "toysad8k-official-seen-test-715-v1"
    cohort_root = tmp_path / cohort_id
    model_safe = cohort_root / "model-safe"
    safe = cohort_root / "safe"
    model_safe.mkdir(parents=True)
    safe.mkdir()
    (model_safe / "prediction_lock.json").write_text("{}\n", encoding="utf-8")
    (model_safe / "prediction_manifest.jsonl").write_text("{}\n", encoding="utf-8")
    (safe / "run_record.json").write_text("{}\n", encoding="utf-8")
    digest = "a" * 64
    record = official_runner._publish_prediction_resource(
        SimpleNamespace(output_root=tmp_path),
        "seen",
        {"identity_sha256": digest},
        method={"sha256": "b" * 64},
        policy={"sha256": "c" * 64},
        lock_result={"sha256": "d" * 64},
    )
    resource = tmp_path / "publish" / "official-seen-prediction"
    assert record["stage"] == "official_seen_prediction_release"
    assert (resource / "model-safe/prediction_lock.json").is_file()
    assert (resource / "safe/run_record.json").is_file()
    assert not any("evaluator" in path.as_posix() for path in resource.rglob("*"))


def test_recovery_bundle_is_deterministic_and_exact(tmp_path: Path) -> None:
    cohort_id = "toysad8k-official-seen-test-715-v1"
    root = tmp_path / "work" / cohort_id
    (root / "locateanything").mkdir(parents=True)
    (root / "pipeline_identity.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cohort_id": cohort_id,
                "identity_sha256": _digest("identity"),
            }
        )
    )
    (root / "locateanything/shard-00000.jsonl").write_text("{}\n")
    bundle, manifest = create_recovery_bundle(
        cohort_root=root,
        output_directory=tmp_path / "bundle",
        cohort_id=cohort_id,
    )
    first_hash = sha256(bundle.read_bytes()).hexdigest()
    bundle, manifest = create_recovery_bundle(
        cohort_root=root,
        output_directory=tmp_path / "bundle",
        cohort_id=cohort_id,
    )
    assert sha256(bundle.read_bytes()).hexdigest() == first_hash
    restored = restore_recovery_bundle(
        bundle_path=bundle,
        manifest_path=manifest,
        output_root=tmp_path / "restored",
        cohort_id=cohort_id,
    )
    assert restored["bundle_sha256"] == first_hash
    assert (
        tmp_path / "restored" / cohort_id / "locateanything/shard-00000.jsonl"
    ).read_text() == "{}\n"


def test_incremental_stage_advances_past_verified_shards(tmp_path: Path) -> None:
    identity = FrozenPipelineIdentity(*(_digest(str(index)) for index in range(7)))
    rows = [
        {
            "cohort_id": "toysad8k-official-seen-test-715-v1",
            "opaque_scene_id": "S1000001",
            "view_id": f"V{index:02d}",
            "image_path": f"images/S1000001/V{index:02d}.png",
            "image_sha256": _digest(f"image-{index}"),
        }
        for index in range(5)
    ]

    def worker(row: dict[str, object]) -> dict[str, object]:
        return {**row, "status": "complete", "objects": []}

    covered = []
    processed = []
    for _ in range(3):
        result = run_exact_sharded_stage(
            stage="locateanything",
            input_records=rows,
            output_root=tmp_path,
            identity=identity,
            worker=worker,
            shard_size=2,
            maximum_new_shards=1,
        )
        covered.append(result["covered_input_count"])
        processed.append(result["processed_this_run"])

    assert covered == [2, 4, 5]
    assert processed == [2, 2, 1]


def test_official_wrapper_supplies_the_complete_protected_runner_contract(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def capture(namespace: SimpleNamespace) -> dict[str, object]:
        captured.update(vars(namespace))
        return {"status": "checkpointed"}

    monkeypatch.setattr(official_runner, "assert_evaluator_absent", lambda *args, **kwargs: None)
    monkeypatch.setattr(official_runner.protected_runner, "run", capture)
    arguments = SimpleNamespace(
        seen_model_input=tmp_path / "seen",
        unseen_model_input=None,
        selected_prompt=tmp_path / "prompt.json",
        sam_checkpoint=tmp_path / "sam.pt",
        method_freeze=tmp_path / "method.json",
        application_contract=tmp_path / "application.json",
        policy_freeze=tmp_path / "policy.json",
        lock_result_receipt=tmp_path / "lock-result.json",
        output_root=tmp_path / "output",
        mounted_root=[],
        locate_model_source=None,
        run_mode="incremental",
        device="cuda",
        shard_size=100,
        maximum_new_shards=4,
    )

    result = official_runner._run_role(arguments, "seen")

    assert result == {"status": "checkpointed"}
    assert captured["recovery_bundle"] is None
    assert captured["recovery_manifest"] is None
    assert captured["recovery_output_directory"] is None
    assert captured["prune_after_recovery"] is False
