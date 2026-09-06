"""One-shot evaluation for frozen internal-lock and official predictions.

The workflow verifies every model, policy, prediction, cohort, and evaluator
package digest before parsing evaluator labels. It evaluates the frozen outputs
without changing a score, identity assignment, review rule, or view contract.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
)
from visual_peer_inspector.evaluation.masks import binary_mask_iou
from visual_peer_inspector.evaluation.proposals import box_iou, match_boxes
from visual_peer_inspector.evaluation.ranking import ranking_metrics
from visual_peer_inspector.frozen_resource import (
    publish_frozen_resource,
    verify_frozen_resource,
)
from visual_peer_inspector.models.sam2 import decode_binary_mask
from visual_peer_inspector.notebook_provenance import (
    bind_runtime_code_provenance_to_json,
)
from visual_peer_inspector.qualitative_reporting import (
    EvaluatorObject,
    EvaluatorOverlay,
    EvaluatorView,
    PredictedObject,
    QualitativeResult,
    QualitativeView,
    render_evaluator_overlay_panel,
    save_panel_png,
)
from visual_peer_inspector.workflows.frozen_development_pipeline import (
    verify_prediction_lock,
)

WORKFLOW_SCHEMA = "visual-peer-one-shot-evaluation/v1"
CONTRACT_SCHEMA = "visual-peer-one-shot-evaluation-contract/v1"
EVALUATOR_PACKAGE_SCHEMA = "visual-peer-evaluator-package/v1"
LABEL_SCHEMA = "visual-peer-evaluator-label/v1"
CROSSWALK_SCHEMA = "visual-peer-evaluator-crosswalk/v1"
GEOMETRY_SCHEMA = "visual-peer-evaluator-geometry/v1"
OFFICIAL_EVALUATOR_PACKAGE_SCHEMA = "visual-peer-official-evaluator-manifest/v1"
OFFICIAL_LABEL_SCHEMA = "visual-peer-official-labels/v1"
OFFICIAL_CROSSWALK_SCHEMA = "visual-peer-official-crosswalk/v1"
OFFICIAL_GEOMETRY_SCHEMA = "visual-peer-official-geometry/v1"
VIEW_COUNTS = (2, 3, 5)
EXPECTED_COHORTS = {
    "09d": {
        "toysad8k-internal-lock-400-v1": {
            "role": "internal_lock",
            "split_name": "internal-lock",
            "membership_sha256": (
                "819f21f49705f9398c809002905abe52b8305ad41e4cab349cbbf3a4bd54afe6"
            ),
            "scene_count": 400,
            "image_count": 8_000,
            "asset_group_count": 109,
        }
    },
    "10b": {
        "toysad8k-official-seen-test-715-v1": {
            "role": "official_test",
            "split_name": "seen-test",
            "membership_sha256": "e8ce6470f9439f0548fadde36b720b93a1bfb5cbbec4fb565ed70c5edf83be01",
            "source_membership_sha256": "2ab155c3b520d75c228d8a650323d13c7bc4a42589edbca19c878a3a44247a25",
            "scene_count": 715,
            "image_count": 14_300,
            "asset_group_count": 113,
        },
        "toysad8k-official-unseen-test-2437-v1": {
            "role": "official_test",
            "split_name": "unseen-test",
            "membership_sha256": "1c3abcb450588f6a99c457b6b3ccb80afa008fd5dfbebbc97d725c67a52cfbcb",
            "source_membership_sha256": "356ed4d864701dad16a805d89ef69b1d96dcb191f9f678c97c964746057ec776",
            "scene_count": 2_437,
            "image_count": 48_740,
            "asset_group_count": 316,
        },
    },
}
REQUIRED_PREDICTION_ARTIFACTS = {
    "prediction_manifest.jsonl",
    "global_embeddings.npy",
    "embedding_index.jsonl",
    "multiview_prediction_manifest.jsonl",
}
REQUIRED_EVALUATOR_ARTIFACTS = {
    "crosswalk_private.jsonl",
    "labels_private.jsonl",
    "geometry_private.jsonl",
}
MARGIN_BINS = (0.0, 0.05, 0.10, 0.20, 0.40, 1.0000001)
AGREEMENT_BINS = (0.0, 0.20, 0.40, 0.60, 0.80, 1.0000001)
OFFICIAL_CORRESPONDENCE_MINIMUM_IOU = 0.50
EVALUATION_RULES = {
    "09d": {
        "prediction_reference_mapping": "evaluator_supplied_locked_crosswalk",
    },
    "10b": {
        "prediction_reference_mapping": "maximum_total_iou_one_to_one",
        "minimum_box_iou": OFFICIAL_CORRESPONDENCE_MINIMUM_IOU,
    },
}


def _digest(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(dict(row)) + b"\n")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path.name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"CSV columns differ within {path.name}")
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(
                    f"JSONL row {line_number} is not an object in {path.name}"
                )
            rows.append(value)
    return rows


def _verified_self_hash(value: Mapping[str, Any], field: str) -> None:
    recorded = value.get(field)
    if not _is_sha256(recorded):
        raise ValueError(f"{field} is not a SHA-256 digest")
    unhashed = {key: item for key, item in value.items() if key != field}
    if recorded != _digest(unhashed):
        raise ValueError(f"{field} differs from the record content")


@dataclass(frozen=True)
class CohortExpectation:
    """Exact non-label resources authorised for one evaluation cohort."""

    cohort_id: str
    role: str
    split_name: str
    membership_sha256: str
    source_membership_sha256: str
    scene_count: int
    image_count: int
    asset_group_count: int
    prediction_lock_sha256: str
    prediction_run_record_sha256: str
    evaluator_manifest_sha256: str


@dataclass(frozen=True)
class EvaluationContract:
    """Immutable method, policy, and cohort hashes for one evaluator notebook."""

    stage: str
    method_record_sha256: str
    policy_record_sha256: str
    evaluation_rules: Mapping[str, Any]
    cohorts: tuple[CohortExpectation, ...]
    contract_sha256: str
    internal_lock_result_receipt_sha256: str | None = None
    official_inference_summary_sha256: str | None = None


@dataclass(frozen=True)
class VerifiedBoundary:
    """Paths verified before evaluator labels are parsed."""

    expectation: CohortExpectation
    prediction_root: Path
    prediction_lock: Mapping[str, Any]
    prediction_rows: tuple[Mapping[str, Any], ...]
    multiview_rows: tuple[Mapping[str, Any], ...]
    evaluator_root: Path
    evaluator_manifest: Mapping[str, Any]
    mounted_roots: tuple[Path, ...] = ()


def load_evaluation_contract(path: Path, *, expected_stage: str) -> EvaluationContract:
    """Verify the content-addressed execution contract for 09D or 10B."""

    source = Path(path).expanduser().resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != CONTRACT_SCHEMA
        or value.get("status") != "locked"
    ):
        raise ValueError("Evaluation contract is not a locked one-shot contract")
    if value.get("stage") != expected_stage or expected_stage not in EXPECTED_COHORTS:
        raise ValueError("Evaluation contract stage is invalid")
    if value.get("evaluation_rules") != EVALUATION_RULES[expected_stage]:
        raise ValueError("Evaluation contract has different scoring rules")
    _verified_self_hash(value, "contract_sha256")
    method_hash = str(value.get("method_record_sha256", ""))
    policy_hash = str(value.get("policy_record_sha256", ""))
    if not _is_sha256(method_hash) or not _is_sha256(policy_hash):
        raise ValueError("Method and policy hashes must be explicit SHA-256 digests")
    lock_result_hash = value.get("internal_lock_result_receipt_sha256")
    official_summary_hash = value.get("official_inference_summary_sha256")
    if expected_stage == "10b":
        if not _is_sha256(lock_result_hash) or not _is_sha256(official_summary_hash):
            raise ValueError(
                "Official evaluation requires exact 09D and 10A receipt hashes"
            )
    elif lock_result_hash is not None or official_summary_hash is not None:
        raise ValueError("Internal-lock evaluation cannot depend on official receipts")
    raw_cohorts = value.get("cohorts")
    if not isinstance(raw_cohorts, list):
        raise TypeError("Evaluation contract requires a cohort list")
    expected = EXPECTED_COHORTS[expected_stage]
    observed = {str(item.get("cohort_id")): item for item in raw_cohorts}
    if set(observed) != set(expected):
        raise ValueError("Evaluation contract has the wrong cohort set")
    cohorts = []
    for cohort_id in sorted(expected):
        source_row = observed[cohort_id]
        fixed = expected[cohort_id]
        for field in ("role", "split_name", "scene_count", "image_count"):
            if source_row.get(field) != fixed[field]:
                raise ValueError(
                    f"Evaluation contract differs for {cohort_id}: {field}"
                )
        for field in ("membership_sha256",):
            if field in fixed and source_row.get(field) != fixed[field]:
                raise ValueError(
                    f"Evaluation contract differs for {cohort_id}: {field}"
                )
            if not _is_sha256(source_row.get(field)):
                raise ValueError(f"Evaluation contract is missing exact {field}")
        source_membership = source_row.get(
            "source_membership_sha256", source_row.get("membership_sha256")
        )
        if (
            "source_membership_sha256" in fixed
            and source_membership != fixed["source_membership_sha256"]
        ):
            raise ValueError(
                f"Evaluation contract differs for {cohort_id}: source_membership_sha256"
            )
        if not _is_sha256(source_membership):
            raise ValueError(
                "Evaluation contract requires an exact source membership hash"
            )
        source_row = {**source_row, "source_membership_sha256": source_membership}
        if (
            "asset_group_count" in fixed
            and source_row.get("asset_group_count") != fixed["asset_group_count"]
        ):
            raise ValueError(
                f"Evaluation contract differs for {cohort_id}: asset_group_count"
            )
        if int(source_row.get("asset_group_count", 0)) < 1:
            raise ValueError("Evaluation contract requires the exact asset-group count")
        for field in (
            "prediction_lock_sha256",
            "prediction_run_record_sha256",
            "evaluator_manifest_sha256",
        ):
            if not _is_sha256(source_row.get(field)):
                raise ValueError(f"Evaluation contract is missing exact {field}")
        cohorts.append(
            CohortExpectation(
                **{
                    key: source_row[key]
                    for key in CohortExpectation.__dataclass_fields__
                }
            )
        )
    return EvaluationContract(
        stage=expected_stage,
        method_record_sha256=method_hash,
        policy_record_sha256=policy_hash,
        evaluation_rules=dict(value["evaluation_rules"]),
        cohorts=tuple(cohorts),
        contract_sha256=str(value["contract_sha256"]),
        internal_lock_result_receipt_sha256=None
        if lock_result_hash is None
        else str(lock_result_hash),
        official_inference_summary_sha256=None
        if official_summary_hash is None
        else str(official_summary_hash),
    )


def _find_exact_file(roots: Sequence[Path], name: str, expected_sha256: str) -> Path:
    matches = []
    for root in roots:
        resolved = Path(root).expanduser().resolve(strict=True)
        for path in resolved.rglob(name):
            if path.is_symlink() or not path.is_file():
                continue
            if file_sha256(path) == expected_sha256:
                matches.append(path.resolve())
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise ValueError(
            f"Expected one {name} with hash {expected_sha256}, found {len(unique)}"
        )
    return unique[0]


def _verify_official_prediction_resource(
    lock_path: Path, *, cohort_id: str
) -> Mapping[str, Any]:
    """Verify the complete role-specific resource surrounding an official lock."""

    resource_root = lock_path.parent.parent
    record_path = resource_root / "resource_record.json"
    if not record_path.is_file():
        raise ValueError("Official prediction lock is not inside a frozen resource")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected_files = set(record.get("file_sha256s", {}))
    role = "seen" if "-seen-test-" in cohort_id else "unseen"
    verified = verify_frozen_resource(
        resource_root,
        expected_stage=f"official_{role}_prediction_release",
        expected_files=expected_files,
    )
    if verified.get("bindings", {}).get("cohort_id") != cohort_id:
        raise ValueError("Official prediction resource has a different cohort binding")
    return verified


def _verify_method_and_policy(
    roots: Sequence[Path], contract: EvaluationContract
) -> tuple[Path, Path, Mapping[str, Any], Mapping[str, Any]]:
    method_path = _find_exact_file(
        roots, "method_freeze.json", contract.method_record_sha256
    )
    policy_path = _find_exact_file(
        roots, "review_policy_freeze.json", contract.policy_record_sha256
    )
    verify_frozen_resource(
        method_path.parent,
        expected_stage="cross_view_identity_method_release",
        expected_files={
            "application_output_contract.json",
            "method_freeze.json",
        },
    )
    verify_frozen_resource(
        policy_path.parent,
        expected_stage="calibration_review_policy_release",
        expected_files={"review_policy_freeze.json"},
    )
    method = json.loads(method_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if (
        method.get("schema_version") != "visual-peer-08f-workflow/v1"
        or method.get("status") != "frozen"
    ):
        raise ValueError("Method record is not the frozen Notebook 08F method")
    chosen = method.get("chosen_configuration")
    if not isinstance(chosen, Mapping) or method.get(
        "chosen_configuration_sha256"
    ) != _digest(chosen):
        raise ValueError("Frozen method configuration hash is invalid")
    if policy.get("schema_version") != "visual-peer-09b-review-policy/v1" or policy.get(
        "status"
    ) not in {"frozen", "locked"}:
        raise ValueError("Policy record is not the frozen Notebook 09B policy")
    if (
        policy.get("method_record_sha256", policy.get("method_freeze_sha256"))
        != contract.method_record_sha256
    ):
        raise ValueError("Review policy is linked to a different method")
    recorded_policy_hash = policy.get("policy_record_sha256")
    if recorded_policy_hash is not None:
        _verified_self_hash(policy, "policy_record_sha256")
    if policy.get("selection", {}).get("universal_accuracy_gate") is not None:
        raise ValueError(
            "The frozen policy unexpectedly contains a universal accuracy gate"
        )
    return method_path, policy_path, method, policy


def _verify_evaluator_manifest(
    path: Path, expectation: CohortExpectation
) -> Mapping[str, Any]:
    if file_sha256(path) != expectation.evaluator_manifest_sha256:
        raise ValueError("Evaluator manifest differs from the execution contract")
    value = json.loads(path.read_text(encoding="utf-8"))
    schema = value.get("schema_version")
    if schema not in {
        EVALUATOR_PACKAGE_SCHEMA,
        OFFICIAL_EVALUATOR_PACKAGE_SCHEMA,
    }:
        raise ValueError("Evaluator package manifest is not locked")
    if schema == EVALUATOR_PACKAGE_SCHEMA and value.get("status") != "locked":
        raise ValueError("Evaluator package manifest is not locked")
    if schema == OFFICIAL_EVALUATOR_PACKAGE_SCHEMA and value.get("status") not in {
        None,
        "locked",
    }:
        raise ValueError("Official evaluator manifest has an invalid status")
    checks = {
        "cohort_id": expectation.cohort_id,
        "scene_count": expectation.scene_count,
        "image_count": expectation.image_count,
    }
    membership = value.get(
        "source_membership_sha256",
        value.get(
            "source_scene_membership_sha256",
            value.get("cohort_membership_sha256"),
        ),
    )
    if membership != expectation.source_membership_sha256:
        raise ValueError("Evaluator package differs from the cohort membership hash")
    for key, expected in checks.items():
        if value.get(key) != expected:
            raise ValueError(
                f"Evaluator package differs from the cohort contract: {key}"
            )
    recorded_groups = value.get("asset_group_count")
    if recorded_groups is not None and recorded_groups != expectation.asset_group_count:
        raise ValueError("Evaluator package differs from the asset-group count")
    artifacts = value.get("artifacts", value.get("files"))
    if not isinstance(artifacts, list):
        direct_hashes = {
            "crosswalk_private.jsonl": value.get("crosswalk_sha256"),
            "labels_private.jsonl": value.get("labels_sha256"),
            "geometry_private.jsonl": value.get("geometry_sha256"),
        }
        if not all(_is_sha256(item) for item in direct_hashes.values()):
            raise TypeError("Evaluator manifest has no complete artifact inventory")
        artifacts = [
            {"path": name, "sha256": digest}
            for name, digest in sorted(direct_hashes.items())
        ]
    names = {str(item.get("path")) for item in artifacts}
    if not REQUIRED_EVALUATOR_ARTIFACTS.issubset(names):
        raise ValueError("Evaluator package is missing required private artifacts")
    if value.get("artifact_inventory_sha256") is not None and value.get(
        "artifact_inventory_sha256"
    ) != _digest(artifacts):
        raise ValueError("Evaluator artifact inventory hash is invalid")
    root = path.parent.resolve()
    for item in artifacts:
        artifact = (root / str(item["path"])).resolve()
        if (
            root not in artifact.parents
            or artifact.is_symlink()
            or not artifact.is_file()
        ):
            raise ValueError("Evaluator artifact path is unsafe or missing")
        if file_sha256(artifact) != item.get("sha256"):
            raise ValueError("Evaluator artifact differs from its locked digest")
    return value


def _named_files(roots: Sequence[Path], name: str) -> list[Path]:
    files: set[Path] = set()
    for source in roots:
        root = Path(source).expanduser().resolve(strict=True)
        files.update(
            path.resolve()
            for path in root.rglob(name)
            if path.is_file() and not path.is_symlink()
        )
    return sorted(files)


def _single_record(
    roots: Sequence[Path], name: str, predicate: Any, description: str
) -> tuple[Path, Mapping[str, Any]]:
    matches = []
    for path in _named_files(roots, name):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(value, Mapping) and predicate(value):
            matches.append((path, value))
    if len(matches) != 1:
        raise ValueError(f"Expected one {description}, found {len(matches)}")
    return matches[0]


def create_evaluation_contract(
    *, stage: str, mounted_roots: Sequence[Path], output_path: Path
) -> Mapping[str, Any]:
    """Pin the exact frozen resources before any evaluator label is parsed."""

    if stage not in EXPECTED_COHORTS:
        raise ValueError("Evaluation contract stage is invalid")
    roots = tuple(
        Path(root).expanduser().resolve(strict=True) for root in mounted_roots
    )
    method_path, _method = _single_record(
        roots,
        "method_freeze.json",
        lambda value: (
            value.get("schema_version") == "visual-peer-08f-workflow/v1"
            and value.get("status") == "frozen"
            and isinstance(value.get("chosen_configuration"), Mapping)
            and value.get("chosen_configuration_sha256")
            == _digest(value["chosen_configuration"])
        ),
        "valid frozen method record",
    )
    method_hash = file_sha256(method_path)
    policy_path, policy = _single_record(
        roots,
        "review_policy_freeze.json",
        lambda value: (
            value.get("schema_version") == "visual-peer-09b-review-policy/v1"
            and value.get("status") in {"frozen", "locked"}
            and value.get("method_record_sha256", value.get("method_freeze_sha256"))
            == method_hash
        ),
        "review policy linked to the frozen method",
    )
    _verified_self_hash(policy, "policy_record_sha256")
    if policy.get("selection", {}).get("universal_accuracy_gate") is not None:
        raise ValueError("The review policy contains a universal accuracy gate")
    policy_hash = file_sha256(policy_path)

    verified_predictions = []
    for cohort_id, fixed in sorted(EXPECTED_COHORTS[stage].items()):
        lock_candidates = []
        for path in _named_files(roots, "prediction_lock.json"):
            try:
                preview = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if (
                preview.get("cohort_id") == cohort_id
                and preview.get("cohort_membership_sha256")
                == fixed["membership_sha256"]
                and preview.get("scene_count") == fixed["scene_count"]
                and preview.get("image_count") == fixed["image_count"]
            ):
                lock_candidates.append((path, verify_prediction_lock(path)))
        if len(lock_candidates) != 1:
            raise ValueError(
                f"Expected one verified prediction lock for {cohort_id}, "
                f"found {len(lock_candidates)}"
            )
        lock_path, lock = lock_candidates[0]
        if stage == "10b":
            _verify_official_prediction_resource(lock_path, cohort_id=cohort_id)
        lock_hash = file_sha256(lock_path)
        required_names = {str(item["path"]) for item in lock.get("artifacts", [])}
        if not REQUIRED_PREDICTION_ARTIFACTS.issubset(required_names):
            raise ValueError("Prediction lock is missing final multi-view artifacts")

        def run_matches(
            value: Mapping[str, Any],
            *,
            expected_cohort: str = cohort_id,
            expected_identity: str = str(lock.get("identity_sha256", "")),
            expected_lock: str = lock_hash,
        ) -> bool:
            return (
                value.get("status") == "completed"
                and value.get("cohort_id") == expected_cohort
                and value.get("prediction_only") is True
                and value.get("evaluator_opened") is False
                and value.get("accuracy_metrics_reported") is False
                and value.get("method_record_sha256") == method_hash
                and value.get("policy_record_sha256") == policy_hash
                and value.get("identity_sha256") == expected_identity
                and value.get("prediction_bundle", {}).get("prediction_lock_sha256")
                == expected_lock
            )

        run_path, _run_record = _single_record(
            roots,
            "run_record.json",
            run_matches,
            f"prediction-only run record for {cohort_id}",
        )
        source_membership = str(
            fixed.get("source_membership_sha256", fixed["membership_sha256"])
        )
        verified_predictions.append(
            {
                "cohort_id": cohort_id,
                "fixed": fixed,
                "lock_sha256": lock_hash,
                "run_record_sha256": file_sha256(run_path),
                "source_membership_sha256": source_membership,
            }
        )

    lock_result_hash = None
    official_summary_hash = None
    if stage == "10b":
        lock_result_path, _lock_result = _single_record(
            roots,
            "internal_lock_result_receipt.json",
            lambda value: (
                value.get("schema_version") == "visual-peer-09d-lock-result/v1"
                and value.get("status") == "protocol_complete"
                and value.get("method_record_sha256") == method_hash
                and value.get("policy_record_sha256") == policy_hash
                and value.get("method_or_policy_changed") is False
                and _is_sha256(value.get("internal_lock_prediction_lock_sha256"))
                and _is_sha256(value.get("internal_lock_evaluation_manifest_sha256"))
            ),
            "completed Notebook 09D protocol receipt",
        )
        verify_frozen_resource(
            lock_result_path.parent,
            expected_stage="internal_lock_decision_release",
            expected_files={"internal_lock_result_receipt.json"},
        )
        lock_result_hash = file_sha256(lock_result_path)
        expected_lock_hashes = {
            str(item["cohort_id"]): str(item["lock_sha256"])
            for item in verified_predictions
        }

        def summary_matches(value: Mapping[str, Any]) -> bool:
            cohorts = value.get("cohorts")
            if not isinstance(cohorts, list):
                return False
            observed = {
                str(item.get("cohort_id")): item.get("prediction_lock_sha256")
                for item in cohorts
                if isinstance(item, Mapping)
            }
            return (
                value.get("schema_version") == "visual-peer-10a-official-inference/v1"
                and value.get("status") == "completed"
                and value.get("method_record_sha256") == method_hash
                and value.get("policy_record_sha256") == policy_hash
                and value.get("lock_result_receipt_sha256") == lock_result_hash
                and value.get("prediction_only") is True
                and value.get("evaluator_opened") is False
                and value.get("accuracy_metrics_reported") is False
                and observed == expected_lock_hashes
            )

        summary_path, _summary = _single_record(
            roots,
            "official_inference_summary.json",
            summary_matches,
            "official inference summary linked to both prediction locks",
        )
        verify_frozen_resource(
            summary_path.parent,
            expected_stage="official_prediction_summary_release",
            expected_files={"official_inference_summary.json"},
        )
        official_summary_hash = file_sha256(summary_path)

    cohort_rows = []
    for prediction in verified_predictions:
        cohort_id = str(prediction["cohort_id"])
        fixed = prediction["fixed"]
        source_membership = str(prediction["source_membership_sha256"])

        def evaluator_matches(
            value: Mapping[str, Any],
            *,
            expected_cohort: str = cohort_id,
            expected_scenes: int = int(fixed["scene_count"]),
            expected_images: int = int(fixed["image_count"]),
            expected_source_membership: str = source_membership,
        ) -> bool:
            membership = value.get(
                "source_membership_sha256",
                value.get(
                    "source_scene_membership_sha256",
                    value.get("cohort_membership_sha256"),
                ),
            )
            return (
                value.get("schema_version")
                in {EVALUATOR_PACKAGE_SCHEMA, OFFICIAL_EVALUATOR_PACKAGE_SCHEMA}
                and value.get("cohort_id") == expected_cohort
                and value.get("scene_count") == expected_scenes
                and value.get("image_count") == expected_images
                and membership == expected_source_membership
            )

        evaluator_path, _evaluator = _single_record(
            roots,
            "evaluator_manifest.json",
            evaluator_matches,
            f"evaluator manifest for {cohort_id}",
        )
        evaluator_hash = file_sha256(evaluator_path)
        expectation = CohortExpectation(
            cohort_id=cohort_id,
            role=str(fixed["role"]),
            split_name=str(fixed["split_name"]),
            membership_sha256=str(fixed["membership_sha256"]),
            source_membership_sha256=source_membership,
            scene_count=int(fixed["scene_count"]),
            image_count=int(fixed["image_count"]),
            asset_group_count=int(fixed["asset_group_count"]),
            prediction_lock_sha256=str(prediction["lock_sha256"]),
            prediction_run_record_sha256=str(prediction["run_record_sha256"]),
            evaluator_manifest_sha256=evaluator_hash,
        )
        _verify_evaluator_manifest(evaluator_path, expectation)
        cohort_rows.append(
            {
                field: getattr(expectation, field)
                for field in CohortExpectation.__dataclass_fields__
            }
        )
    record = {
        "schema_version": CONTRACT_SCHEMA,
        "status": "locked",
        "stage": stage,
        "method_record_sha256": method_hash,
        "policy_record_sha256": policy_hash,
        "evaluation_rules": EVALUATION_RULES[stage],
        "cohorts": cohort_rows,
        "label_files_parsed_during_contract_creation": False,
        "method_or_policy_changed": False,
    }
    if stage == "10b":
        record["internal_lock_result_receipt_sha256"] = lock_result_hash
        record["official_inference_summary_sha256"] = official_summary_hash
    record["contract_sha256"] = _digest(record)
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != record:
            raise FileExistsError("Existing evaluation contract differs")
    else:
        _atomic_json(destination, record)
    return record


def verify_boundaries(
    *, contract: EvaluationContract, mounted_roots: Sequence[Path]
) -> tuple[tuple[VerifiedBoundary, ...], Mapping[str, Any]]:
    """Verify method, policy, predictions, and evaluator packages before labels."""

    roots = tuple(
        Path(root).expanduser().resolve(strict=True) for root in mounted_roots
    )
    method_path, policy_path, _method, _policy = _verify_method_and_policy(
        roots, contract
    )
    verified_predictions = []
    for expectation in contract.cohorts:
        lock_path = _find_exact_file(
            roots, "prediction_lock.json", expectation.prediction_lock_sha256
        )
        if contract.stage == "10b":
            _verify_official_prediction_resource(
                lock_path, cohort_id=expectation.cohort_id
            )
        lock = verify_prediction_lock(lock_path)
        required_values = {
            "cohort_id": expectation.cohort_id,
            "cohort_membership_sha256": expectation.membership_sha256,
            "scene_count": expectation.scene_count,
            "image_count": expectation.image_count,
        }
        for field, expected in required_values.items():
            if lock.get(field) != expected:
                raise ValueError(
                    f"Prediction lock differs from the cohort contract: {field}"
                )
        locked_names = {str(item["path"]) for item in lock.get("artifacts", [])}
        if not REQUIRED_PREDICTION_ARTIFACTS.issubset(locked_names):
            raise ValueError("Prediction lock is missing required model outputs")
        run_path = _find_exact_file(
            roots, "run_record.json", expectation.prediction_run_record_sha256
        )
        run_record = json.loads(run_path.read_text(encoding="utf-8"))
        if (
            run_record.get("status") != "completed"
            or run_record.get("cohort_id") != expectation.cohort_id
            or run_record.get("prediction_only") is not True
            or run_record.get("evaluator_opened") is not False
            or run_record.get("accuracy_metrics_reported") is not False
            or run_record.get("method_record_sha256") != contract.method_record_sha256
            or run_record.get("policy_record_sha256") != contract.policy_record_sha256
            or run_record.get("identity_sha256") != lock.get("identity_sha256")
            or run_record.get("prediction_bundle", {}).get("prediction_lock_sha256")
            != expectation.prediction_lock_sha256
        ):
            raise ValueError(
                "Prediction run record does not preserve the frozen boundary"
            )
        prediction_root = lock_path.parent
        prediction_rows = _read_jsonl(prediction_root / "prediction_manifest.jsonl")
        multiview_rows = _read_jsonl(
            prediction_root / "multiview_prediction_manifest.jsonl"
        )
        if len(prediction_rows) != expectation.image_count:
            raise ValueError(
                "Prediction manifest image count differs from the cohort contract"
            )
        image_keys = {
            (str(row.get("opaque_scene_id", "")), str(row.get("view_id", "")))
            for row in prediction_rows
        }
        if len(image_keys) != expectation.image_count or any(
            not all(key) for key in image_keys
        ):
            raise ValueError("Prediction manifest has missing or duplicate image keys")
        decision_keys = {
            (str(row.get("opaque_scene_id", "")), int(row.get("view_count", -1)))
            for row in multiview_rows
        }
        if len(decision_keys) != expectation.scene_count * len(VIEW_COUNTS):
            raise ValueError("Multi-view prediction coverage is incomplete")
        if {count for _scene, count in decision_keys} != set(VIEW_COUNTS):
            raise ValueError(
                "Multi-view predictions do not use the frozen 2, 3 and 5 view contracts"
            )
        verified_predictions.append(
            (
                expectation,
                lock,
                prediction_root,
                tuple(prediction_rows),
                tuple(multiview_rows),
            )
        )
    if contract.stage == "10b":
        receipt_path = _find_exact_file(
            roots,
            "internal_lock_result_receipt.json",
            str(contract.internal_lock_result_receipt_sha256),
        )
        verify_frozen_resource(
            receipt_path.parent,
            expected_stage="internal_lock_decision_release",
            expected_files={"internal_lock_result_receipt.json"},
        )
        receipt_value = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt_value.get("schema_version") != "visual-peer-09d-lock-result/v1"
            or receipt_value.get("status") != "protocol_complete"
            or receipt_value.get("decision_scope")
            != "evaluation_completed_once_without_method_or_policy_change"
            or receipt_value.get("scientific_performance_accepted") is not False
            or receipt_value.get("method_record_sha256")
            != contract.method_record_sha256
            or receipt_value.get("policy_record_sha256")
            != contract.policy_record_sha256
            or receipt_value.get("method_or_policy_changed") is not False
        ):
            raise ValueError("Official evaluation has an invalid 09D result receipt")
        summary_path = _find_exact_file(
            roots,
            "official_inference_summary.json",
            str(contract.official_inference_summary_sha256),
        )
        verify_frozen_resource(
            summary_path.parent,
            expected_stage="official_prediction_summary_release",
            expected_files={"official_inference_summary.json"},
        )
        summary_value = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_locks = {
            item[0].cohort_id: item[0].prediction_lock_sha256
            for item in verified_predictions
        }
        observed_locks = {
            str(item.get("cohort_id")): item.get("prediction_lock_sha256")
            for item in summary_value.get("cohorts", [])
            if isinstance(item, Mapping)
        }
        if (
            summary_value.get("schema_version")
            != "visual-peer-10a-official-inference/v1"
            or summary_value.get("status") != "completed"
            or summary_value.get("method_record_sha256")
            != contract.method_record_sha256
            or summary_value.get("policy_record_sha256")
            != contract.policy_record_sha256
            or summary_value.get("lock_result_receipt_sha256")
            != contract.internal_lock_result_receipt_sha256
            or summary_value.get("prediction_only") is not True
            or summary_value.get("evaluator_opened") is not False
            or summary_value.get("accuracy_metrics_reported") is not False
            or observed_locks != expected_locks
        ):
            raise ValueError(
                "Official inference summary does not link both frozen split locks"
            )
    boundaries = []
    for (
        expectation,
        lock,
        prediction_root,
        prediction_rows,
        multiview_rows,
    ) in verified_predictions:
        manifest_path = _find_exact_file(
            roots, "evaluator_manifest.json", expectation.evaluator_manifest_sha256
        )
        evaluator_manifest = _verify_evaluator_manifest(manifest_path, expectation)
        boundaries.append(
            VerifiedBoundary(
                expectation=expectation,
                prediction_root=prediction_root,
                prediction_lock=lock,
                prediction_rows=prediction_rows,
                multiview_rows=multiview_rows,
                evaluator_root=manifest_path.parent,
                evaluator_manifest=evaluator_manifest,
                mounted_roots=roots,
            )
        )
    receipt = {
        "schema_version": WORKFLOW_SCHEMA,
        "status": "verified_before_label_access",
        "stage": contract.stage,
        "contract_sha256": contract.contract_sha256,
        "method_record": method_path.name,
        "method_record_sha256": contract.method_record_sha256,
        "policy_record": policy_path.name,
        "policy_record_sha256": contract.policy_record_sha256,
        "internal_lock_result_receipt_sha256": (
            contract.internal_lock_result_receipt_sha256
        ),
        "official_inference_summary_sha256": (
            contract.official_inference_summary_sha256
        ),
        "cohorts": [
            {
                "cohort_id": item.expectation.cohort_id,
                "prediction_lock_sha256": item.expectation.prediction_lock_sha256,
                "prediction_run_record_sha256": item.expectation.prediction_run_record_sha256,
                "evaluator_manifest_sha256": item.expectation.evaluator_manifest_sha256,
            }
            for item in boundaries
        ],
        "no_retuning": True,
        "method_changed": False,
        "policy_changed": False,
    }
    return tuple(boundaries), receipt


def _official_geometry_objects(
    view: Mapping[str, Any], object_ids: Sequence[str]
) -> list[tuple[str, Sequence[float]]]:
    boxes = view.get("boxes_xyxy")
    if isinstance(boxes, Mapping):
        return [(str(key), value) for key, value in boxes.items()]
    if not isinstance(boxes, list):
        raise TypeError("Official evaluator view requires boxes_xyxy")
    if boxes and all(isinstance(item, Mapping) for item in boxes):
        return [
            (
                str(item.get("object_id", item.get("reference_object_id", ""))),
                item.get("box_xyxy", item.get("box")),
            )
            for item in boxes
        ]
    if len(boxes) != len(object_ids):
        raise ValueError("Official evaluator boxes do not align with object labels")
    return list(zip(object_ids, boxes, strict=True))


def _normalise_official_evaluator(
    boundary: VerifiedBoundary,
    raw_labels: Sequence[Mapping[str, Any]],
    raw_crosswalk: Sequence[Mapping[str, Any]],
    raw_geometry: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Adapt the official scene records after label access is authorised."""

    crosswalk_by_scene = {str(row["opaque_scene_id"]): row for row in raw_crosswalk}
    geometry_by_scene = {str(row["opaque_scene_id"]): row for row in raw_geometry}
    prediction_by_key = {
        (str(row["opaque_scene_id"]), str(row["view_id"])): row
        for row in boundary.prediction_rows
    }
    labels = []
    geometry = []
    generated_crosswalk = []
    for raw_label in raw_labels:
        scene_id = str(raw_label["opaque_scene_id"])
        bridge = crosswalk_by_scene.get(scene_id)
        scene_geometry = geometry_by_scene.get(scene_id)
        if bridge is None or scene_geometry is None:
            raise ValueError("Official evaluator files have different scene coverage")
        object_map = bridge.get("object_id_map")
        view_map = bridge.get("view_id_map")
        if not isinstance(object_map, Mapping) or not isinstance(view_map, Mapping):
            raise TypeError("Official crosswalk requires object and view mappings")
        object_labels = raw_label.get("object_labels")
        if not isinstance(object_labels, list) or not object_labels:
            raise ValueError("Official labels require an object label list")
        raw_object_ids = [str(item["object_id"]) for item in object_labels]
        reference_ids = list(raw_object_ids)
        if set(object_map) != set(reference_ids) or any(
            int(object_map[object_id]) != index
            for index, object_id in enumerate(reference_ids)
        ):
            raise ValueError("Official object index map differs from the label order")
        anomalies = [
            str(item["object_id"])
            for item in object_labels
            if bool(item.get("is_anomaly"))
        ]
        if len(anomalies) != int(raw_label.get("anomaly_count", -1)) or not anomalies:
            raise ValueError("Official anomaly count differs from object labels")
        labels.append(
            {
                "schema_version": LABEL_SCHEMA,
                "cohort_id": boundary.expectation.cohort_id,
                "opaque_scene_id": scene_id,
                "opaque_asset_group_id": str(bridge["base_asset_id"]),
                "category": str(bridge.get("category", "not_recorded")),
                "anomaly_type": str(raw_label.get("anomaly_type", "not_recorded")),
                "object_count_band": "3"
                if len(reference_ids) == 3
                else "4"
                if len(reference_ids) == 4
                else "5_plus",
                "anomalous_reference_object_ids": anomalies,
            }
        )
        for raw_view in scene_geometry.get("views", []):
            view_id = str(raw_view["view_id"])
            if view_map.get(view_id) != raw_view.get("source_filename"):
                raise ValueError(
                    "Official view filename map differs from the geometry record"
                )
            reference_objects = _official_geometry_objects(raw_view, reference_ids)
            normalized_references = []
            for source_object_id, box in reference_objects:
                reference_id = str(source_object_id)
                row = {
                    "schema_version": GEOMETRY_SCHEMA,
                    "cohort_id": boundary.expectation.cohort_id,
                    "opaque_scene_id": scene_id,
                    "view_id": view_id,
                    "reference_object_id": reference_id,
                    "box_xyxy": [float(value) for value in box],
                    "mask_rle": None,
                    "visible": True,
                    "rgb_relative_path": str(raw_view["source_filename"]),
                }
                geometry.append(row)
                normalized_references.append(row)
            prediction = prediction_by_key.get((scene_id, view_id), {})
            predicted_objects = list(prediction.get("objects", []))
            matches = match_boxes(
                [row["box_xyxy"] for row in normalized_references],
                [item["box_xyxy"] for item in predicted_objects],
                minimum_iou=OFFICIAL_CORRESPONDENCE_MINIMUM_IOU,
            )
            for match in matches:
                generated_crosswalk.append(
                    {
                        "schema_version": CROSSWALK_SCHEMA,
                        "cohort_id": boundary.expectation.cohort_id,
                        "opaque_scene_id": scene_id,
                        "view_id": view_id,
                        "prediction_object_id": str(
                            predicted_objects[match.proposal_index]["object_id"]
                        ),
                        "reference_object_id": str(
                            normalized_references[match.reference_index][
                                "reference_object_id"
                            ]
                        ),
                        "box_iou": match.iou,
                        "correspondence_rule": "maximum_total_iou_one_to_one_at_iou_0_50",
                    }
                )
    return labels, generated_crosswalk, geometry


def _open_evaluator(
    boundary: VerifiedBoundary,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    labels = _read_jsonl(boundary.evaluator_root / "labels_private.jsonl")
    crosswalk = _read_jsonl(boundary.evaluator_root / "crosswalk_private.jsonl")
    geometry = _read_jsonl(boundary.evaluator_root / "geometry_private.jsonl")
    schemas = {
        str(labels[0].get("schema_version", "")) if labels else "",
        str(crosswalk[0].get("schema_version", "")) if crosswalk else "",
        str(geometry[0].get("schema_version", "")) if geometry else "",
    }
    official = {
        OFFICIAL_LABEL_SCHEMA,
        OFFICIAL_CROSSWALK_SCHEMA,
        OFFICIAL_GEOMETRY_SCHEMA,
    }
    if schemas == official:
        labels, crosswalk, geometry = _normalise_official_evaluator(
            boundary, labels, crosswalk, geometry
        )
    expectation = boundary.expectation
    label_ids = set()
    groups = set()
    for row in labels:
        if (
            row.get("schema_version") != LABEL_SCHEMA
            or row.get("cohort_id") != expectation.cohort_id
        ):
            raise ValueError("Private label row has the wrong schema or cohort")
        scene_id = str(row.get("opaque_scene_id", ""))
        anomalies = row.get("anomalous_reference_object_ids")
        if (
            not scene_id
            or scene_id in label_ids
            or not isinstance(anomalies, list)
            or not anomalies
        ):
            raise ValueError("Private labels have an invalid scene or anomaly set")
        label_ids.add(scene_id)
        groups.add(str(row.get("opaque_asset_group_id", "")))
    if (
        len(label_ids) != expectation.scene_count
        or "" in groups
        or len(groups) != expectation.asset_group_count
    ):
        raise ValueError("Private labels do not match the fixed scene and asset counts")
    crosswalk_keys = set()
    for row in crosswalk:
        if (
            row.get("schema_version") != CROSSWALK_SCHEMA
            or row.get("cohort_id") != expectation.cohort_id
        ):
            raise ValueError("Private crosswalk row has the wrong schema or cohort")
        key = (
            str(row.get("opaque_scene_id", "")),
            str(row.get("view_id", "")),
            str(row.get("prediction_object_id", "")),
        )
        if (
            not all(key)
            or key in crosswalk_keys
            or not str(row.get("reference_object_id", ""))
        ):
            raise ValueError(
                "Private crosswalk contains an invalid or duplicate mapping"
            )
        crosswalk_keys.add(key)
    geometry_keys = set()
    for row in geometry:
        if (
            row.get("schema_version") != GEOMETRY_SCHEMA
            or row.get("cohort_id") != expectation.cohort_id
        ):
            raise ValueError("Private geometry row has the wrong schema or cohort")
        key = (
            str(row.get("opaque_scene_id", "")),
            str(row.get("view_id", "")),
            str(row.get("reference_object_id", "")),
        )
        if not all(key) or key in geometry_keys:
            raise ValueError("Private geometry contains an invalid or duplicate object")
        geometry_keys.add(key)
    return labels, crosswalk, geometry


def _track_reference_ids(
    track: Mapping[str, Any],
    mapping: Mapping[tuple[str, str], str],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                mapping[(str(member["view_id"]), str(member["candidate_id"]))]
                for member in track.get("members", [])
                if (str(member["view_id"]), str(member["candidate_id"])) in mapping
            }
        )
    )


def _policy_state(decision: Mapping[str, Any]) -> str:
    signal = str(decision.get("review_signal", "insufficient_evidence"))
    if signal == "automatic":
        return "result_ready"
    if signal in {"review", "identity_review"}:
        return "review_required"
    return "insufficient_evidence"


def build_scene_metrics(
    boundary: VerifiedBoundary,
    labels: Sequence[Mapping[str, Any]],
    crosswalk: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Score every scene and view contract, including terminal failures."""

    label_by_scene = {str(row["opaque_scene_id"]): row for row in labels}
    crosswalk_by_scene: dict[str, dict[tuple[str, str], str]] = defaultdict(dict)
    for row in crosswalk:
        crosswalk_by_scene[str(row["opaque_scene_id"])][
            (str(row["view_id"]), str(row["prediction_object_id"]))
        ] = str(row["reference_object_id"])
    decisions = {
        (str(row["opaque_scene_id"]), int(row["view_count"])): row
        for row in boundary.multiview_rows
    }
    expected_keys = {
        (scene, count) for scene in label_by_scene for count in VIEW_COUNTS
    }
    if set(decisions) != expected_keys:
        raise ValueError("Locked decisions and private labels have different coverage")
    output = []
    for scene_id, view_count in sorted(expected_keys):
        label = label_by_scene[scene_id]
        decision = decisions[(scene_id, view_count)]
        mapping = crosswalk_by_scene.get(scene_id, {})
        anomalies = {str(value) for value in label["anomalous_reference_object_ids"]}
        ranked = sorted(
            (
                track
                for track in decision.get("tracks", [])
                if track.get("fused_unusual_score") is not None
            ),
            key=lambda track: (
                -float(track["fused_unusual_score"]),
                str(track["identity_id"]),
            ),
        )
        relevant = []
        switch_count = unmatched_member_count = ambiguous_track_count = (
            cycle_failure_count
        ) = 0
        for track in decision.get("tracks", []):
            members = list(track.get("members", []))
            mapped_ids = _track_reference_ids(track, mapping)
            mapped_member_count = sum(
                (str(member["view_id"]), str(member["candidate_id"])) in mapping
                for member in members
            )
            unmatched_member_count += len(members) - mapped_member_count
            switch_count += int(len(mapped_ids) > 1)
            ambiguous_track_count += int(
                str(track.get("identity_status", "")) != "matched"
            )
            cycle_failure_count += int(track.get("cycle_consistent") is not True)
            relevant.append(len(mapped_ids) == 1 and mapped_ids[0] in anomalies)
        ranked_relevance = []
        ranked_reference_ids: list[str | None] = []
        credited_anomalies: set[str] = set()
        for track in ranked:
            mapped_ids = _track_reference_ids(track, mapping)
            reference_id = mapped_ids[0] if len(mapped_ids) == 1 else None
            is_new_anomaly = bool(
                reference_id in anomalies and reference_id not in credited_anomalies
            )
            ranked_relevance.append(is_new_anomaly)
            ranked_reference_ids.append(reference_id)
            if is_new_anomaly and reference_id is not None:
                credited_anomalies.add(reference_id)
        ranking = ranking_metrics(
            ranked_relevance,
            positive_total=len(anomalies),
            ranked_object_ids=ranked_reference_ids,
            positive_object_ids=sorted(anomalies),
        )
        first_rank = next(
            (index + 1 for index, item in enumerate(ranked_relevance) if item), None
        )
        available = bool(ranked)
        policy_state = _policy_state(decision)
        top1 = int(ranking.top1)
        top2 = int(any(ranked_relevance[:2]))
        output.append(
            {
                "cohort_id": boundary.expectation.cohort_id,
                "split_name": boundary.expectation.split_name,
                "opaque_scene_id": scene_id,
                "opaque_asset_group_id": str(label["opaque_asset_group_id"]),
                "view_count": view_count,
                "category": str(label.get("category", "not_recorded")),
                "anomaly_type": str(label.get("anomaly_type", "not_recorded")),
                "object_count_band": str(
                    label.get("object_count_band", "not_recorded")
                ),
                "anomaly_count": len(anomalies),
                "ranked_identity_count": len(ranked),
                "coverage": int(available),
                "top1": top1,
                "top2": top2,
                "target_rank": first_rank,
                "reciprocal_rank": ranking.reciprocal_rank,
                "average_precision": ranking.average_precision,
                "recall_at_2": ranking.recall_at_2,
                "exact_set_at_anomaly_count": ranking.exact_set,
                "retrieved_anomaly_count": ranking.retrieved_positive_count,
                "policy_state": policy_state,
                "review_required": int(policy_state != "result_ready"),
                "insufficient_evidence": int(policy_state == "insufficient_evidence"),
                "selective_eligible": int(policy_state == "result_ready"),
                "selective_correct": int(policy_state == "result_ready" and top1 == 1),
                "margin": decision.get("margin"),
                "agreement": decision.get("agreement"),
                "identity_switch_count": switch_count,
                "identity_unmatched_member_count": unmatched_member_count,
                "identity_ambiguous_track_count": ambiguous_track_count,
                "identity_cycle_failure_count": cycle_failure_count,
                "view_failure_count": len(decision.get("view_failures", [])),
            }
        )
    return output


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def summary_metrics(
    rows: Sequence[Mapping[str, Any]], *, include_pooled: bool
) -> list[dict[str, Any]]:
    """Aggregate primary results by split and view count."""

    split_names = sorted({str(row["split_name"]) for row in rows})
    scopes = [(split, count) for split in split_names for count in VIEW_COUNTS]
    if include_pooled:
        scopes.extend(("pooled-descriptive", count) for count in VIEW_COUNTS)
    output = []
    for split, view_count in scopes:
        subset = [
            row
            for row in rows
            if int(row["view_count"]) == view_count
            and (split == "pooled-descriptive" or row["split_name"] == split)
        ]
        automatic = [row for row in subset if int(row["selective_eligible"]) == 1]
        selective_correct = sum(int(row["selective_correct"]) for row in automatic)
        output.append(
            {
                "split_name": split,
                "view_count": view_count,
                "scene_count": len(subset),
                "top1_accuracy": _rate(
                    sum(int(row["top1"]) for row in subset), len(subset)
                ),
                "top2_accuracy": _rate(
                    sum(int(row["top2"]) for row in subset), len(subset)
                ),
                "mean_reciprocal_rank": _rate(
                    sum(float(row["reciprocal_rank"]) for row in subset), len(subset)
                ),
                "mean_average_precision": _rate(
                    sum(float(row["average_precision"]) for row in subset),
                    len(subset),
                ),
                "mean_recall_at_2": _rate(
                    sum(float(row["recall_at_2"]) for row in subset), len(subset)
                ),
                "exact_set_accuracy": _rate(
                    sum(float(row["exact_set_at_anomaly_count"]) for row in subset),
                    len(subset),
                ),
                "target_found_rate": _rate(
                    sum(row["target_rank"] is not None for row in subset), len(subset)
                ),
                "mean_target_rank_when_found": (
                    None
                    if not any(row["target_rank"] is not None for row in subset)
                    else float(
                        np.mean(
                            [
                                int(row["target_rank"])
                                for row in subset
                                if row["target_rank"] is not None
                            ]
                        )
                    )
                ),
                "coverage": _rate(
                    sum(int(row["coverage"]) for row in subset), len(subset)
                ),
                "automatic_count": len(automatic),
                "selective_accuracy": _rate(selective_correct, len(automatic)),
                "selective_risk": None
                if not automatic
                else 1.0 - selective_correct / len(automatic),
                "review_rate": _rate(
                    sum(int(row["review_required"]) for row in subset), len(subset)
                ),
                "insufficient_evidence_rate": _rate(
                    sum(int(row["insufficient_evidence"]) for row in subset),
                    len(subset),
                ),
                "identity_switch_scene_rate": _rate(
                    sum(int(row["identity_switch_count"] > 0) for row in subset),
                    len(subset),
                ),
                "identity_cycle_failure_scene_rate": _rate(
                    sum(int(row["identity_cycle_failure_count"] > 0) for row in subset),
                    len(subset),
                ),
                "claim_scope": "descriptive_only"
                if split == "pooled-descriptive"
                else "primary_split_result",
            }
        )
    return output


def _cluster_interval(
    rows: Sequence[Mapping[str, Any]], metric: str, *, seed: int, repetitions: int
) -> tuple[float, float, float, int]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row["opaque_asset_group_id"])].append(float(row[metric]))
    if not groups:
        raise ValueError("A clustered interval requires at least one asset group")
    names = sorted(groups)
    observed = float(np.mean([value for name in names for value in groups[name]]))
    generator = random.Random(seed)
    samples = []
    for _ in range(repetitions):
        selected = [generator.choice(names) for _name in names]
        samples.append(
            float(np.mean([value for name in selected for value in groups[name]]))
        )
    low, high = np.quantile(samples, (0.025, 0.975))
    return observed, float(low), float(high), len(names)


def clustered_intervals(
    rows: Sequence[Mapping[str, Any]], *, repetitions: int = 10_000, seed: int = 32_040
) -> list[dict[str, Any]]:
    """Return deterministic base-asset-clustered intervals for primary metrics."""

    if repetitions < 100:
        raise ValueError("Clustered intervals require at least 100 repetitions")
    output = []
    for split_index, split in enumerate(
        sorted({str(row["split_name"]) for row in rows})
    ):
        for view_count in VIEW_COUNTS:
            subset = [
                row
                for row in rows
                if row["split_name"] == split and int(row["view_count"]) == view_count
            ]
            metric_sources = {
                "top1_accuracy": (subset, "top1"),
                "top2_accuracy": (subset, "top2"),
                "mean_reciprocal_rank": (subset, "reciprocal_rank"),
                "mean_average_precision": (subset, "average_precision"),
                "mean_recall_at_2": (subset, "recall_at_2"),
                "exact_set_accuracy": (subset, "exact_set_at_anomaly_count"),
                "coverage": (subset, "coverage"),
                "review_rate": (subset, "review_required"),
            }
            automatic = [row for row in subset if int(row["selective_eligible"]) == 1]
            if automatic:
                metric_sources["selective_accuracy"] = (automatic, "selective_correct")
            for metric_index, (metric, (source, field)) in enumerate(
                metric_sources.items()
            ):
                mean, low, high, clusters = _cluster_interval(
                    source,
                    field,
                    seed=seed + split_index * 1_000 + view_count * 10 + metric_index,
                    repetitions=repetitions,
                )
                output.append(
                    {
                        "split_name": split,
                        "view_count": view_count,
                        "metric": metric,
                        "mean": mean,
                        "ci_lower": low,
                        "ci_upper": high,
                        "scene_count": len(source),
                        "asset_group_count": clusters,
                        "repetitions": repetitions,
                        "seed": seed
                        + split_index * 1_000
                        + view_count * 10
                        + metric_index,
                    }
                )
                if metric == "selective_accuracy":
                    output.append(
                        {
                            "split_name": split,
                            "view_count": view_count,
                            "metric": "selective_risk",
                            "mean": 1.0 - mean,
                            "ci_lower": 1.0 - high,
                            "ci_upper": 1.0 - low,
                            "scene_count": len(source),
                            "asset_group_count": clusters,
                            "repetitions": repetitions,
                            "seed": seed
                            + split_index * 1_000
                            + view_count * 10
                            + metric_index,
                        }
                    )
    return output


def reliability_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Describe empirical accuracy in predeclared margin and agreement bins."""

    output = []
    for split in sorted({str(row["split_name"]) for row in rows}):
        for view_count in VIEW_COUNTS:
            base = [
                row
                for row in rows
                if row["split_name"] == split and int(row["view_count"]) == view_count
            ]
            for signal, bins in (
                ("margin", MARGIN_BINS),
                ("agreement", AGREEMENT_BINS),
            ):
                for lower, upper in pairwise(bins):
                    members = [
                        row
                        for row in base
                        if row.get(signal) is not None
                        and lower <= float(row[signal]) < upper
                    ]
                    output.append(
                        {
                            "split_name": split,
                            "view_count": view_count,
                            "signal": signal,
                            "bin_lower": lower,
                            "bin_upper": min(upper, 1.0),
                            "scene_count": len(members),
                            "mean_signal": None
                            if not members
                            else float(
                                np.mean([float(row[signal]) for row in members])
                            ),
                            "top1_accuracy": _rate(
                                sum(int(row["top1"]) for row in members), len(members)
                            ),
                            "automatic_rate": _rate(
                                sum(int(row["selective_eligible"]) for row in members),
                                len(members),
                            ),
                        }
                    )
    return output


def subgroup_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Report fixed subgroup and category results without suppressing failures."""

    output = []
    for split in sorted({str(row["split_name"]) for row in rows}):
        split_rows = [row for row in rows if row["split_name"] == split]
        for field in ("view_count", "category", "anomaly_type", "object_count_band"):
            for value in sorted({str(row[field]) for row in split_rows}):
                members = [row for row in split_rows if str(row[field]) == value]
                output.append(
                    {
                        "split_name": split,
                        "subgroup_field": field,
                        "subgroup_value": value,
                        "scene_contract_count": len(members),
                        "top1_accuracy": _rate(
                            sum(int(row["top1"]) for row in members), len(members)
                        ),
                        "top2_accuracy": _rate(
                            sum(int(row["top2"]) for row in members), len(members)
                        ),
                        "mean_reciprocal_rank": _rate(
                            sum(float(row["reciprocal_rank"]) for row in members),
                            len(members),
                        ),
                        "mean_average_precision": _rate(
                            sum(float(row["average_precision"]) for row in members),
                            len(members),
                        ),
                        "mean_recall_at_2": _rate(
                            sum(float(row["recall_at_2"]) for row in members),
                            len(members),
                        ),
                        "exact_set_accuracy": _rate(
                            sum(
                                float(row["exact_set_at_anomaly_count"])
                                for row in members
                            ),
                            len(members),
                        ),
                        "coverage": _rate(
                            sum(int(row["coverage"]) for row in members), len(members)
                        ),
                        "review_rate": _rate(
                            sum(int(row["review_required"]) for row in members),
                            len(members),
                        ),
                    }
                )
    return output


def failure_and_identity_tables(
    boundaries: Sequence[VerifiedBoundary], scene_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarise terminal stage failures, identity errors, and recorded runtime."""

    failures = []
    runtimes = []
    for boundary in boundaries:
        counts = Counter(
            str(row.get("terminal_failure") or "completed")
            for row in boundary.prediction_rows
        )
        for status, count in sorted(counts.items()):
            failures.append(
                {
                    "split_name": boundary.expectation.split_name,
                    "outcome": status,
                    "image_count": count,
                    "share": count / boundary.expectation.image_count,
                }
            )
        run_candidates = sorted(
            boundary.prediction_root.parent.rglob("runtime_receipt.json")
        )
        runtime = {}
        if run_candidates:
            runtime = json.loads(run_candidates[0].read_text(encoding="utf-8"))
        for stage, values in sorted(runtime.get("stages", {}).items()):
            runtimes.append(
                {
                    "split_name": boundary.expectation.split_name,
                    "stage": stage,
                    "duration_seconds": values.get("duration_seconds"),
                    "processed_this_run": values.get("processed_this_run"),
                    "resumed": values.get("resumed"),
                    "terminal_failures": values.get("terminal_failures"),
                }
            )
    if not runtimes:
        runtimes.append(
            {
                "split_name": "not_available",
                "stage": "not_available",
                "duration_seconds": None,
                "processed_this_run": None,
                "resumed": None,
                "terminal_failures": None,
            }
        )
    identity = []
    for split in sorted({str(row["split_name"]) for row in scene_rows}):
        for view_count in VIEW_COUNTS:
            subset = [
                row
                for row in scene_rows
                if row["split_name"] == split and int(row["view_count"]) == view_count
            ]
            identity.append(
                {
                    "split_name": split,
                    "view_count": view_count,
                    "scene_count": len(subset),
                    "switch_scene_count": sum(
                        int(row["identity_switch_count"] > 0) for row in subset
                    ),
                    "unmatched_member_count": sum(
                        int(row["identity_unmatched_member_count"]) for row in subset
                    ),
                    "ambiguous_track_scene_count": sum(
                        int(row["identity_ambiguous_track_count"] > 0) for row in subset
                    ),
                    "cycle_failure_scene_count": sum(
                        int(row["identity_cycle_failure_count"] > 0) for row in subset
                    ),
                    "view_failure_scene_count": sum(
                        int(row["view_failure_count"] > 0) for row in subset
                    ),
                }
            )
    return failures, identity, runtimes


def attrition_table(
    boundary: VerifiedBoundary,
    geometry: Sequence[Mapping[str, Any]],
    crosswalk: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Measure detection and mask attrition where evaluator geometry is supplied."""

    prediction_by_key = {
        (str(row["opaque_scene_id"]), str(row["view_id"])): row
        for row in boundary.prediction_rows
    }
    crosswalk_by_reference = {
        (
            str(row["opaque_scene_id"]),
            str(row["view_id"]),
            str(row["reference_object_id"]),
        ): str(row["prediction_object_id"])
        for row in crosswalk
    }
    decisions = {
        (str(row["opaque_scene_id"]), int(row["view_count"])): row
        for row in boundary.multiview_rows
    }
    geometry_by_view: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in geometry:
        if bool(row.get("visible", True)):
            geometry_by_view[(str(row["opaque_scene_id"]), str(row["view_id"]))].append(
                row
            )
    output = []
    for view_count in VIEW_COUNTS:
        reference_count = proposal_count = mask_count = identity_count = (
            ranked_count
        ) = 0
        box_ious = []
        mask_ious = []
        for (scene_id, count), decision in decisions.items():
            if count != view_count:
                continue
            track_members = {
                (str(member["view_id"]), str(member["candidate_id"]))
                for track in decision.get("tracks", [])
                for member in track.get("members", [])
            }
            ranked_ids = {
                str(track["identity_id"])
                for track in decision.get("tracks", [])
                if track.get("fused_unusual_score") is not None
            }
            member_track = {
                (str(member["view_id"]), str(member["candidate_id"])): str(
                    track["identity_id"]
                )
                for track in decision.get("tracks", [])
                for member in track.get("members", [])
            }
            for view_id in decision.get("view_ids", []):
                view_id = str(view_id)
                prediction_row = prediction_by_key.get((scene_id, view_id), {})
                predictions = {
                    str(item["object_id"]): item
                    for item in prediction_row.get("objects", [])
                }
                for reference in geometry_by_view.get((scene_id, view_id), []):
                    reference_count += 1
                    reference_id = str(reference["reference_object_id"])
                    prediction_id = crosswalk_by_reference.get(
                        (scene_id, view_id, reference_id)
                    )
                    prediction = predictions.get(prediction_id or "")
                    if prediction is None:
                        continue
                    proposal_count += 1
                    if reference.get("box_xyxy") is not None:
                        box_ious.append(
                            box_iou(reference["box_xyxy"], prediction["box_xyxy"])
                        )
                    if int(prediction.get("foreground_pixels", 0)) > 0:
                        mask_count += 1
                        if reference.get("mask_rle") is not None:
                            mask_ious.append(
                                binary_mask_iou(
                                    decode_binary_mask(reference["mask_rle"]),
                                    decode_binary_mask(prediction["mask_rle"]),
                                )
                            )
                    member_key = (view_id, str(prediction["object_id"]))
                    if member_key in track_members:
                        identity_count += 1
                        if member_track[member_key] in ranked_ids:
                            ranked_count += 1
        output.append(
            {
                "split_name": boundary.expectation.split_name,
                "view_count": view_count,
                "reference_object_view_count": reference_count,
                "proposal_correspondence_count": proposal_count,
                "nonempty_mask_count": mask_count,
                "identity_track_member_count": identity_count,
                "ranked_track_member_count": ranked_count,
                "proposal_retention": _rate(proposal_count, reference_count),
                "mask_retention": _rate(mask_count, reference_count),
                "identity_retention": _rate(identity_count, reference_count),
                "ranking_retention": _rate(ranked_count, reference_count),
                "mean_box_iou": None if not box_ious else float(np.mean(box_ious)),
                "mean_mask_iou": None if not mask_ious else float(np.mean(mask_ious)),
                "geometry_supported": bool(reference_count),
            }
        )
    return output


def error_taxonomy(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Assign each result to one deterministic, mutually exclusive outcome."""

    counts: Counter[tuple[str, int, str]] = Counter()
    for row in rows:
        if int(row["coverage"]) == 0:
            category = "no_ranked_result"
        elif int(row["view_failure_count"]) > 0:
            category = "view_or_stage_failure"
        elif int(row["identity_switch_count"]) > 0:
            category = "identity_switch"
        elif int(row["identity_unmatched_member_count"]) > 0:
            category = "unmatched_identity_member"
        elif int(row["identity_ambiguous_track_count"]) > 0:
            category = "ambiguous_identity"
        elif int(row["identity_cycle_failure_count"]) > 0:
            category = "cycle_inconsistency"
        elif int(row["top1"]) == 0 and row["policy_state"] == "result_ready":
            category = "wrong_automatic_result"
        elif int(row["top1"]) == 0:
            category = "wrong_result_sent_to_review"
        elif row["policy_state"] == "result_ready":
            category = "correct_automatic_result"
        else:
            category = "correct_result_sent_to_review"
        counts[(str(row["split_name"]), int(row["view_count"]), category)] += 1
    return [
        {
            "split_name": split,
            "view_count": view_count,
            "error_category": category,
            "scene_count": count,
        }
        for (split, view_count, category), count in sorted(counts.items())
    ]


def decision_confusion_table(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Cross-tabulate result correctness and the frozen release decision."""

    counts: Counter[tuple[str, int, str, str]] = Counter()
    denominators: Counter[tuple[str, int]] = Counter()
    for row in rows:
        split = str(row["split_name"])
        view_count = int(row["view_count"])
        if int(row["coverage"]) == 0:
            result_outcome = "no_ranked_result"
        elif int(row["top1"]) == 1:
            result_outcome = "correct_top1"
        else:
            result_outcome = "wrong_top1"
        policy_action = str(row["policy_state"])
        counts[(split, view_count, result_outcome, policy_action)] += 1
        denominators[(split, view_count)] += 1
    return [
        {
            "split_name": split,
            "view_count": view_count,
            "result_outcome": result_outcome,
            "policy_action": policy_action,
            "scene_count": count,
            "share": count / denominators[(split, view_count)],
        }
        for (split, view_count, result_outcome, policy_action), count in sorted(
            counts.items()
        )
    ]


def _select_panel_rows(
    rows: Sequence[Mapping[str, Any]], count: int
) -> list[dict[str, Any]]:
    five = [dict(row) for row in rows if int(row["view_count"]) == 5]
    if len(five) < count:
        raise ValueError(
            f"At least {count} five-view scenes are required for qualitative evidence"
        )
    for row in five:
        if int(row["coverage"]) == 0 or int(row["view_failure_count"]) > 0:
            row["panel_stratum"] = "failure"
        elif int(row["top1"]) == 0:
            row["panel_stratum"] = "wrong"
        elif row.get("margin") is not None and float(row["margin"]) >= 0.20:
            row["panel_stratum"] = "best"
        else:
            row["panel_stratum"] = "typical"
    selected = []
    selected_ids = set()
    quota = max(1, count // 4)
    for stratum in ("best", "typical", "wrong", "failure"):
        candidates = [row for row in five if row["panel_stratum"] == stratum]
        candidates.sort(
            key=lambda row: (
                _digest(
                    {
                        "namespace": "one-shot-panels-v1",
                        "scene": row["opaque_scene_id"],
                        "stratum": stratum,
                    }
                ),
                row["opaque_scene_id"],
            )
        )
        for row in candidates[:quota]:
            selected.append(row)
            selected_ids.add(str(row["opaque_scene_id"]))
    remaining = [row for row in five if str(row["opaque_scene_id"]) not in selected_ids]
    remaining.sort(
        key=lambda row: (
            _digest(
                {"namespace": "one-shot-panel-fill-v1", "scene": row["opaque_scene_id"]}
            ),
            row["opaque_scene_id"],
        )
    )
    selected.extend(remaining[: count - len(selected)])
    return selected[:count]


def _geometry_by_scene(
    geometry: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    output: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in geometry:
        output[str(row["opaque_scene_id"])].append(row)
    return output


def _media_roots(roots: Sequence[Path]) -> tuple[Path, ...]:
    candidates: set[Path] = set()
    for source in roots:
        root = Path(source).resolve(strict=True)
        if root.name == "media":
            candidates.add(root)
        candidates.update(
            path.resolve()
            for path in root.rglob("media")
            if path.is_dir() and not path.is_symlink()
        )
    return tuple(sorted(candidates))


def _resolve_panel_rgb(
    *,
    boundary: VerifiedBoundary,
    scene_id: str,
    view_id: str,
    prediction: Mapping[str, Any],
    reference_items: Sequence[Mapping[str, Any]],
    media_roots: Sequence[Path],
) -> Path:
    expected_digest = str(prediction.get("image_sha256", ""))
    if not _is_sha256(expected_digest):
        raise ValueError("A qualitative panel requires the locked RGB digest")
    candidates: set[Path] = set()
    relative_paths = {
        str(item.get("rgb_relative_path", ""))
        for item in reference_items
        if item.get("rgb_relative_path")
    }
    for relative in relative_paths:
        candidate = (boundary.evaluator_root / relative).resolve()
        if (
            boundary.evaluator_root in candidate.parents
            and candidate.is_file()
            and not candidate.is_symlink()
        ):
            candidates.add(candidate)
    for media_root in media_roots:
        candidate = (media_root / scene_id / "rgb" / f"{view_id}.png").resolve()
        if (
            media_root in candidate.parents
            and candidate.is_file()
            and not candidate.is_symlink()
        ):
            candidates.add(candidate)
    matching = sorted(
        path for path in candidates if file_sha256(path) == expected_digest
    )
    if len(matching) != 1:
        raise ValueError(
            "Expected one mounted RGB image matching the locked prediction digest"
        )
    return matching[0]


def render_private_panels(
    *,
    boundary: VerifiedBoundary,
    scene_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    crosswalk: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
    output_root: Path,
    count: int,
) -> list[dict[str, Any]]:
    """Render deterministic evaluator overlays for best, typical, wrong, and failure cases."""

    if not 1 <= count <= 18:
        raise ValueError("A cohort panel allocation must be between 1 and 18")
    selected = _select_panel_rows(scene_rows, count)
    label_by_scene = {str(row["opaque_scene_id"]): row for row in labels}
    geometry_by_scene = _geometry_by_scene(geometry)
    prediction_by_key = {
        (str(row["opaque_scene_id"]), str(row["view_id"])): row
        for row in boundary.prediction_rows
    }
    decision_by_scene = {
        str(row["opaque_scene_id"]): row
        for row in boundary.multiview_rows
        if int(row["view_count"]) == 5
    }
    media_roots = _media_roots(boundary.mounted_roots)
    inventory = []
    for index, metric in enumerate(selected, 1):
        scene_id = str(metric["opaque_scene_id"])
        decision = decision_by_scene[scene_id]
        track_by_member = {
            (str(member["view_id"]), str(member["candidate_id"])): str(
                track["identity_id"]
            )
            for track in decision.get("tracks", [])
            for member in track.get("members", [])
        }
        views = []
        evaluator_views = []
        geometry_rows = geometry_by_scene.get(scene_id, [])
        geometry_by_view: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for item in geometry_rows:
            geometry_by_view[str(item["view_id"])].append(item)
        for view_id in decision.get("view_ids", []):
            view_id = str(view_id)
            prediction = prediction_by_key.get((scene_id, view_id))
            if prediction is None:
                raise ValueError("A panel view is absent from the prediction manifest")
            reference_items = geometry_by_view.get(view_id, [])
            rgb_path = _resolve_panel_rgb(
                boundary=boundary,
                scene_id=scene_id,
                view_id=view_id,
                prediction=prediction,
                reference_items=reference_items,
                media_roots=media_roots,
            )
            objects = []
            heatmap = None
            for item in prediction.get("objects", []):
                prediction_id = str(item["object_id"])
                identity_id = track_by_member.get(
                    (view_id, prediction_id), prediction_id
                )
                objects.append(
                    PredictedObject(
                        object_id=identity_id,
                        box_xyxy=tuple(float(value) for value in item["box_xyxy"]),
                        sam_mask=decode_binary_mask(item["mask_rle"]),
                        p1_score=item.get("p1_score"),
                        p1_rank=item.get("p1_rank"),
                    )
                )
                if (
                    identity_id == decision.get("fused_odd_one_out")
                    and item.get("p1_heatmap") is not None
                ):
                    heatmap = np.asarray(item["p1_heatmap"], dtype=np.float32)
            views.append(
                QualitativeView(
                    view_id=view_id,
                    rgb=rgb_path,
                    objects=tuple(objects),
                    heatmap=heatmap,
                )
            )
            evaluator_views.append(
                EvaluatorView(
                    view_id=view_id,
                    objects=tuple(
                        EvaluatorObject(
                            object_id=str(item["reference_object_id"]),
                            box_xyxy=None
                            if item.get("box_xyxy") is None
                            else tuple(float(value) for value in item["box_xyxy"]),
                            mask=None
                            if item.get("mask_rle") is None
                            else decode_binary_mask(item["mask_rle"]),
                        )
                        for item in reference_items
                    ),
                )
            )
        result = QualitativeResult(
            result_id=f"{boundary.expectation.split_name}-{index:02d}",
            views=tuple(views),
            fused_odd_one_out=decision.get("fused_odd_one_out"),
            agreement=decision.get("agreement"),
            margin=decision.get("margin"),
            review_state=str(metric["policy_state"]),
            publication_metadata={"view_count": 5, "route": "frozen_automatic"},
        )
        label = label_by_scene[scene_id]
        anomalies = [str(item) for item in label["anomalous_reference_object_ids"]]
        overlay = EvaluatorOverlay(
            views=tuple(evaluator_views),
            reference_odd_one_out=anomalies[0]
            if len(anomalies) == 1
            else ", ".join(anomalies),
        )
        panel = render_evaluator_overlay_panel(result, overlay, panel_size=220)
        panel_path = (
            Path(output_root) / f"panel-{index:02d}-{metric['panel_stratum']}.png"
        )
        save_panel_png(panel, panel_path)
        inventory.append(
            {
                "panel_number": index,
                "split_name": boundary.expectation.split_name,
                "panel_stratum": metric["panel_stratum"],
                "opaque_scene_id": scene_id,
                "top1": metric["top1"],
                "policy_state": metric["policy_state"],
                "path": panel_path.name,
                "sha256": file_sha256(panel_path),
            }
        )
    return inventory


def _render_charts(
    summaries: Sequence[Mapping[str, Any]],
    reliability: Sequence[Mapping[str, Any]],
    attrition: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    paths = []
    primary = [row for row in summaries if row["claim_scope"] == "primary_split_result"]
    for filename, fields, title in (
        (
            "ranking_by_view.png",
            (
                "top1_accuracy",
                "mean_average_precision",
                "mean_recall_at_2",
                "exact_set_accuracy",
            ),
            "Single-candidate and multi-anomaly ranking by view count",
        ),
        (
            "coverage_and_review.png",
            ("coverage", "review_rate", "selective_risk"),
            "Coverage, review, and selective risk",
        ),
    ):
        figure, axis = plt.subplots(figsize=(9.2, 4.8))
        x = np.arange(len(primary))
        width = 0.18
        for offset, field in enumerate(fields):
            values = [
                np.nan if row[field] is None else float(row[field]) for row in primary
            ]
            axis.bar(
                x + (offset - (len(fields) - 1) / 2) * width,
                values,
                width,
                label=field.replace("_", " "),
            )
        axis.set_xticks(
            x, [f"{row['split_name']}\n{row['view_count']} views" for row in primary]
        )
        axis.set_ylim(0, 1)
        axis.set_ylabel("Rate")
        axis.set_title(title)
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        path = output_root / filename
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    margin = [
        row
        for row in reliability
        if row["signal"] == "margin" and row["scene_count"] > 0
    ]
    for split in sorted({str(row["split_name"]) for row in margin}):
        subset = [
            row
            for row in margin
            if row["split_name"] == split and int(row["view_count"]) == 5
        ]
        axis.plot(
            [float(row["mean_signal"]) for row in subset],
            [float(row["top1_accuracy"]) for row in subset],
            marker="o",
            label=split,
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean margin within fixed bin")
    axis.set_ylabel("Top-1 accuracy")
    axis.set_title("Five-view margin reliability")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    path = output_root / "margin_reliability.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(attrition))
    for field in (
        "proposal_retention",
        "mask_retention",
        "identity_retention",
        "ranking_retention",
    ):
        axis.plot(
            x,
            [np.nan if row[field] is None else float(row[field]) for row in attrition],
            marker="o",
            label=field.replace("_", " "),
        )
    axis.set_xticks(
        x, [f"{row['split_name']}\n{row['view_count']} views" for row in attrition]
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Reference object retention")
    axis.set_title("Detection, mask, identity, and ranking attrition")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    path = output_root / "pipeline_attrition.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)
    return paths


def _safe_manifest(root: Path) -> Mapping[str, Any]:
    run_record_path = root / "run_record.json"
    if run_record_path.is_file():
        bind_runtime_code_provenance_to_json(run_record_path)
    artifacts = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    manifest = {
        "schema_version": WORKFLOW_SCHEMA,
        "status": "complete",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "artifact_inventory_sha256": _digest(artifacts),
    }
    _atomic_json(root / "artifact_manifest.json", manifest)
    return manifest


def run_one_shot_evaluation(
    *,
    contract_path: Path,
    mounted_roots: Sequence[Path],
    output_root: Path,
    expected_stage: str,
    bootstrap_repetitions: int = 10_000,
    panel_count: int = 16,
) -> Mapping[str, Any]:
    """Execute one sealed evaluation and publish atomic safe and private outputs."""

    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "One-shot evaluation refuses a reused nonempty result root"
        )
    if output.is_symlink():
        raise ValueError("Evaluation result root cannot be a symbolic link")
    staging = output.with_name(output.name + ".incomplete")
    if staging.exists():
        raise FileExistsError("A prior incomplete evaluation staging root exists")
    contract = load_evaluation_contract(contract_path, expected_stage=expected_stage)
    boundaries, boundary_receipt = verify_boundaries(
        contract=contract, mounted_roots=mounted_roots
    )
    staging.mkdir(parents=True)
    _atomic_json(staging / "boundary_verified_before_labels.json", boundary_receipt)

    all_scene_rows = []
    all_attrition = []
    evaluator_receipts = []
    private_inputs = {}
    for boundary in boundaries:
        labels, crosswalk, geometry = _open_evaluator(boundary)
        rows = build_scene_metrics(boundary, labels, crosswalk)
        all_scene_rows.extend(rows)
        all_attrition.extend(attrition_table(boundary, geometry, crosswalk))
        private_inputs[boundary.expectation.cohort_id] = (labels, crosswalk, geometry)
        evaluator_receipts.append(
            {
                "cohort_id": boundary.expectation.cohort_id,
                "status": "opened_after_all_freeze_checks",
                "evaluator_manifest_sha256": boundary.expectation.evaluator_manifest_sha256,
                "label_row_count": len(labels),
                "crosswalk_row_count": len(crosswalk),
                "geometry_row_count": len(geometry),
            }
        )
    _atomic_json(
        staging / "evaluator_access_receipt.json",
        {
            "schema_version": WORKFLOW_SCHEMA,
            "status": "labels_opened_after_locks",
            "cohorts": evaluator_receipts,
        },
    )

    safe = staging / "safe"
    private = staging / "review-only"
    summaries = summary_metrics(all_scene_rows, include_pooled=expected_stage == "10b")
    intervals = clustered_intervals(all_scene_rows, repetitions=bootstrap_repetitions)
    reliability = reliability_table(all_scene_rows)
    subgroups = subgroup_table(all_scene_rows)
    failures, identity, runtimes = failure_and_identity_tables(
        boundaries, all_scene_rows
    )
    taxonomy = error_taxonomy(all_scene_rows)
    tables = {
        "summary_metrics.csv": summaries,
        "clustered_confidence_intervals.csv": intervals,
        "subgroup_and_category_metrics.csv": subgroups,
        "margin_agreement_reliability.csv": reliability,
        "identity_failures.csv": identity,
        "pipeline_attrition.csv": all_attrition,
        "prediction_failures.csv": failures,
        "runtime_summary.csv": runtimes,
        "error_taxonomy.csv": taxonomy,
        "decision_confusion.csv": decision_confusion_table(all_scene_rows),
    }
    for name, rows in tables.items():
        _atomic_csv(safe / name, rows)
    charts = _render_charts(summaries, reliability, all_attrition, safe / "figures")

    if not 12 <= panel_count <= 18:
        raise ValueError("The evaluator report requires between 12 and 18 panels")
    panel_inventory = []
    base_allocation, remainder = divmod(panel_count, len(boundaries))
    for boundary_index, boundary in enumerate(boundaries):
        labels, crosswalk, geometry = private_inputs[boundary.expectation.cohort_id]
        cohort_rows = [
            row
            for row in all_scene_rows
            if row["cohort_id"] == boundary.expectation.cohort_id
        ]
        items = render_private_panels(
            boundary=boundary,
            scene_rows=cohort_rows,
            labels=labels,
            crosswalk=crosswalk,
            geometry=geometry,
            output_root=private / boundary.expectation.split_name,
            count=base_allocation + int(boundary_index < remainder),
        )
        panel_inventory.extend(items)
    _atomic_json(
        private / "qualitative_panel_inventory.json",
        {
            "schema_version": WORKFLOW_SCHEMA,
            "panel_count": len(panel_inventory),
            "items": panel_inventory,
        },
    )
    _atomic_jsonl(private / "scene_level_evaluation.jsonl", all_scene_rows)
    run_record = {
        "schema_version": WORKFLOW_SCHEMA,
        "status": "completed",
        "stage": expected_stage,
        "contract_sha256": contract.contract_sha256,
        "method_record_sha256": contract.method_record_sha256,
        "policy_record_sha256": contract.policy_record_sha256,
        "evaluation_rules": dict(contract.evaluation_rules),
        "cohort_ids": [item.expectation.cohort_id for item in boundaries],
        "split_reporting": "separate_with_pooled_descriptive"
        if expected_stage == "10b"
        else "internal_lock_one_shot",
        "scene_contract_count": len(all_scene_rows),
        "all_failures_in_denominator": True,
        "no_retuning": True,
        "method_changed": False,
        "policy_changed": False,
        "panel_count": len(panel_inventory),
        "safe_table_count": len(tables),
        "safe_figure_count": len(charts),
        "clearml_status": "pending_bounded_post_compute_finalizer",
    }
    _atomic_json(safe / "run_record.json", run_record)
    safe_manifest = _safe_manifest(safe)
    completion = {
        "schema_version": WORKFLOW_SCHEMA,
        "status": "complete",
        "stage": expected_stage,
        "contract_sha256": contract.contract_sha256,
        "evaluation_rules": dict(contract.evaluation_rules),
        "safe_manifest_sha256": file_sha256(safe / "artifact_manifest.json"),
        "safe_artifact_inventory_sha256": safe_manifest["artifact_inventory_sha256"],
        "evaluator_access_receipt_sha256": file_sha256(
            staging / "evaluator_access_receipt.json"
        ),
        "private_panel_inventory_sha256": file_sha256(
            private / "qualitative_panel_inventory.json"
        ),
        "clearml_is_post_compute_only": True,
    }
    if expected_stage == "09d":
        evaluation_manifest = {
            "schema_version": "visual-peer-09d-evaluation-manifest/v1",
            "status": "complete",
            "contract_sha256": contract.contract_sha256,
            "method_record_sha256": contract.method_record_sha256,
            "policy_record_sha256": contract.policy_record_sha256,
            "internal_lock_prediction_lock_sha256": boundaries[
                0
            ].expectation.prediction_lock_sha256,
            "safe_artifact_manifest_sha256": file_sha256(
                safe / "artifact_manifest.json"
            ),
            "summary_metrics_sha256": file_sha256(safe / "summary_metrics.csv"),
            "clustered_confidence_intervals_sha256": file_sha256(
                safe / "clustered_confidence_intervals.csv"
            ),
            "error_taxonomy_sha256": file_sha256(safe / "error_taxonomy.csv"),
            "qualitative_panel_inventory_sha256": file_sha256(
                private / "qualitative_panel_inventory.json"
            ),
            "protocol_decision": "completed_without_retuning",
            "decision_scope": "evaluation_completed_once_without_method_or_policy_change",
            "scientific_performance_accepted": False,
        }
        evaluation_manifest["manifest_sha256"] = _digest(evaluation_manifest)
        _atomic_json(
            staging / "internal_lock_evaluation_manifest.json", evaluation_manifest
        )
        result_receipt = {
            "schema_version": "visual-peer-09d-lock-result/v1",
            "status": "protocol_complete",
            "decision_scope": "evaluation_completed_once_without_method_or_policy_change",
            "scientific_performance_accepted": False,
            "method_record_sha256": contract.method_record_sha256,
            "policy_record_sha256": contract.policy_record_sha256,
            "internal_lock_prediction_lock_sha256": boundaries[
                0
            ].expectation.prediction_lock_sha256,
            "internal_lock_evaluation_manifest_sha256": file_sha256(
                staging / "internal_lock_evaluation_manifest.json"
            ),
            "method_or_policy_changed": False,
        }
        _atomic_json(staging / "internal_lock_result_receipt.json", result_receipt)
        decision_resource_root = staging / "publish" / "internal-lock-decision"
        publish_frozen_resource(
            resource_root=decision_resource_root,
            stage="internal_lock_decision_release",
            files={
                "internal_lock_result_receipt.json": staging
                / "internal_lock_result_receipt.json"
            },
            bindings={
                "method_record_sha256": contract.method_record_sha256,
                "policy_record_sha256": contract.policy_record_sha256,
                "internal_lock_evaluation_manifest_sha256": file_sha256(
                    staging / "internal_lock_evaluation_manifest.json"
                ),
            },
        )
        completion["decision_resource_manifest_sha256"] = file_sha256(
            decision_resource_root / "artifact_manifest.json"
        )
    _atomic_json(staging / "evaluation_complete.json", completion)
    if output.exists():
        output.rmdir()
    staging.replace(output)
    return completion


__all__ = [
    "AGREEMENT_BINS",
    "CONTRACT_SCHEMA",
    "CROSSWALK_SCHEMA",
    "EVALUATOR_PACKAGE_SCHEMA",
    "EXPECTED_COHORTS",
    "GEOMETRY_SCHEMA",
    "LABEL_SCHEMA",
    "MARGIN_BINS",
    "VIEW_COUNTS",
    "WORKFLOW_SCHEMA",
    "EvaluationContract",
    "attrition_table",
    "build_scene_metrics",
    "clustered_intervals",
    "create_evaluation_contract",
    "decision_confusion_table",
    "error_taxonomy",
    "load_evaluation_contract",
    "reliability_table",
    "run_one_shot_evaluation",
    "subgroup_table",
    "summary_metrics",
    "verify_boundaries",
]
