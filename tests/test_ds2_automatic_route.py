from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from visual_peer_inspector.data.cohort_contract import DS2_CONFIRM_CONTRACT
from visual_peer_inspector.models.sam2 import encode_binary_mask
from visual_peer_inspector.models.visual_encoder import ContractVisualEncoder
from visual_peer_inspector.representation.crops import CropConfiguration
from visual_peer_inspector.models.locateanything import LocateAnythingConfiguration
from visual_peer_inspector.workflows.locateanything_confirmation_ds2 import (
    _require_proposal_readiness,
    build_frozen_automatic_proposals,
    load_frozen_ds1_configuration,
    resolve_ds2_operational_policy,
)
from visual_peer_inspector.workflows.multiview_confirmation_ds2 import (
    _proposal_summary,
    _unscoreable_summary,
    fuse_object_ranks,
    prepare_missing_aware_evaluator_scores,
)
from visual_peer_inspector.workflows.representation_confirmation_ds2 import (
    ConfirmationCandidate,
    _score_group,
    _score_scene,
)


def _candidate(
    image_path: Path,
    *,
    index: int,
    mask: np.ndarray,
) -> ConfirmationCandidate:
    return ConfirmationCandidate(
        opaque_scene_id="S000001",
        view_id="V00",
        route="automatic_box",
        peer_key=f"P{index}",
        candidate_id=f"C{index}",
        image_path=image_path,
        image_sha256="a" * 64,
        box_xyxy=(0, 0, 10, 10),
        foreground_pixels=int(mask.sum()),
        predicted_quality_score=0.9,
        mask_rle=encode_binary_mask(mask),
        mask_sha256="b" * 64,
    )


def _automatic_provenance(
    *,
    status: str = "ok",
    proposal_count: int,
    raw_proposal_count: int | None = None,
    removed_proposal_count: int = 0,
    discarded_untrusted_box_count: int = 0,
) -> dict:
    raw_count = proposal_count if raw_proposal_count is None else raw_proposal_count
    return {
        "source_status": status,
        "source_error_type": None if status == "ok" else "TimeoutError",
        "source_attempt": 1 if status == "ok" else 2,
        "carried_as_zero_proposal": status == "error",
        "raw_proposal_count": raw_count,
        "removed_proposal_count": removed_proposal_count,
        "discarded_untrusted_box_count": discarded_untrusted_box_count,
        "proposal_count": proposal_count,
    }


def test_frozen_ds1_configuration_is_loaded_without_ds2_measurements(tmp_path: Path) -> None:
    configuration = LocateAnythingConfiguration(prompt="individual object")
    path = tmp_path / "selected_prompt.json"
    path.write_text(
        json.dumps(
            {
                "stage": "ds1_locateanything_prompt_freeze",
                "development_only": True,
                "requires_unchanged_ds2_confirmation": True,
                "selected": {
                    "prompt": configuration.prompt,
                    "postprocessing_policy": "area_nms_075",
                },
                "generation_configuration": asdict(configuration),
                "generation_configuration_sha256": configuration.sha256,
            }
        ),
        encoding="utf-8",
    )

    loaded, policy, digest = load_frozen_ds1_configuration(path)

    assert loaded == configuration
    assert policy == "area_nms_075"
    assert len(digest) == 64


def test_automatic_proposal_package_contains_no_reference_identity() -> None:
    rows = [
        {
            "status": "ok",
            "opaque_scene_id": "S000001",
            "view_id": "V00",
            "image_sha256": "a" * 64,
            "configuration_sha256": "b" * 64,
            "attempt": 1,
            "error_type": None,
            "proposal_count": 2,
            "boxes_xyxy": [[0, 0, 10, 10], [20, 20, 30, 30]],
        }
    ]

    output = build_frozen_automatic_proposals(
        rows,
        policy_key="raw",
        freeze_sha256="c" * 64,
    )

    assert output[0]["proposal_count"] == 2
    assert output[0]["discarded_untrusted_box_count"] == 0
    assert "opaque_object_id" not in output[0]
    assert "label" not in output[0]


def test_ds2_policy_bounds_tokens_attempts_and_wall_time() -> None:
    frozen = LocateAnythingConfiguration(prompt="individual object")

    effective, policy = resolve_ds2_operational_policy(frozen)

    assert frozen.max_new_tokens == 8192
    assert effective.max_new_tokens == 4096
    assert effective.prompt == frozen.prompt
    assert effective.model_id == frozen.model_id
    assert policy["maximum_attempts_per_job"] == 2
    assert policy["primary"]["attempts"] == [1, 2]
    assert policy["attempt_timeout_seconds"] == 75.0
    assert policy["terminal_failure_disposition"] == "carry_as_zero_proposal"


def test_terminal_failure_becomes_an_explicit_zero_proposal() -> None:
    rows = [
        {
            "status": "error",
            "opaque_scene_id": "S000001",
            "view_id": "V00",
            "image_sha256": "a" * 64,
            "configuration_sha256": "b" * 64,
            "attempt": 2,
            "proposal_count": 0,
            "boxes_xyxy": [],
            "error_type": "TimeoutError",
        }
    ]

    output = build_frozen_automatic_proposals(
        rows,
        policy_key="area_nms_075",
        freeze_sha256="c" * 64,
    )

    assert output == [
        {
            "schema_version": 2,
            "cohort_id": DS2_CONFIRM_CONTRACT.cohort_id,
            "opaque_scene_id": "S000001",
            "view_id": "V00",
            "image_sha256": "a" * 64,
            "configuration_sha256": "b" * 64,
            "ds1_freeze_sha256": "c" * 64,
            "postprocessing_policy": "area_nms_075",
            "raw_proposal_count": 0,
            "removed_proposal_count": 0,
            "discarded_untrusted_box_count": 0,
            "proposal_count": 0,
            "boxes_xyxy": [],
            "source_status": "error",
            "source_error_type": "TimeoutError",
            "source_attempt": 2,
            "carried_as_zero_proposal": True,
        }
    ]


def test_terminal_truncation_discards_parsed_boxes_instead_of_using_them() -> None:
    rows = [
        {
            "status": "error",
            "opaque_scene_id": "S000001",
            "view_id": "V00",
            "image_sha256": "a" * 64,
            "configuration_sha256": "b" * 64,
            "attempt": 2,
            "proposal_count": 2,
            "boxes_xyxy": [[0, 0, 10, 10], [20, 20, 30, 30]],
            "error_type": "PossibleTruncation",
        }
    ]

    output = build_frozen_automatic_proposals(
        rows,
        policy_key="area_nms_075",
        freeze_sha256="c" * 64,
    )

    assert output[0]["raw_proposal_count"] == 2
    assert output[0]["removed_proposal_count"] == 0
    assert output[0]["discarded_untrusted_box_count"] == 2
    assert output[0]["proposal_count"] == 0
    assert output[0]["boxes_xyxy"] == []
    assert output[0]["source_status"] == "error"
    assert output[0]["source_error_type"] == "PossibleTruncation"
    assert output[0]["carried_as_zero_proposal"] is True


def test_terminal_failure_rejects_source_count_mismatch() -> None:
    rows = [
        {
            "status": "error",
            "opaque_scene_id": "S000001",
            "view_id": "V00",
            "image_sha256": "a" * 64,
            "configuration_sha256": "b" * 64,
            "attempt": 2,
            "proposal_count": 1,
            "boxes_xyxy": [],
            "error_type": "PossibleTruncation",
        }
    ]

    with pytest.raises(ValueError, match="count differs"):
        build_frozen_automatic_proposals(
            rows,
            policy_key="area_nms_075",
            freeze_sha256="c" * 64,
        )


def _readiness_row(index: int, *, status: str = "ok", box_x: int = 1) -> dict:
    boxes = [[box_x, 1, box_x + 10, 11]] if status == "ok" else []
    return {
        "status": status,
        "opaque_scene_id": "S000001",
        "view_id": f"V{index:02d}",
        "image_sha256": f"{index + 1:064x}",
        "configuration_sha256": "b" * 64,
        "attempt": 1 if status == "ok" else 2,
        "proposal_count": len(boxes),
        "boxes_xyxy": boxes,
        "error_type": None if status == "ok" else "TimeoutError",
    }


def test_readiness_gates_allow_recorded_pilot_failures_without_quality_cutoff() -> None:
    smoke = [_readiness_row(0)]
    pilot = [_readiness_row(index) for index in range(20)]
    pilot[-1] = _readiness_row(19, status="error")
    repeated = [_readiness_row(index, box_x=2) for index in range(20)]
    repeated[-1] = _readiness_row(19, status="error")
    dispositions = build_frozen_automatic_proposals(
        pilot,
        policy_key="raw",
        freeze_sha256="c" * 64,
    )

    _require_proposal_readiness(
        smoke,
        list(smoke),
        pilot,
        repeated,
        dispositions,
        pilot_history_count_before_resume=21,
        pilot_history_count_after_resume=21,
    )


def test_readiness_smoke_allows_a_valid_zero_proposal_record() -> None:
    smoke = [_readiness_row(0)]
    smoke[0]["proposal_count"] = 0
    smoke[0]["boxes_xyxy"] = []
    pilot = [_readiness_row(index) for index in range(20)]
    dispositions = build_frozen_automatic_proposals(
        pilot,
        policy_key="raw",
        freeze_sha256="c" * 64,
    )

    _require_proposal_readiness(
        smoke,
        list(smoke),
        pilot,
        pilot,
        dispositions,
        pilot_history_count_before_resume=20,
        pilot_history_count_after_resume=20,
    )


def test_readiness_gates_reject_failed_smoke_and_duplicate_pilot_identity() -> None:
    pilot = [_readiness_row(index) for index in range(20)]
    dispositions = build_frozen_automatic_proposals(
        pilot,
        policy_key="raw",
        freeze_sha256="c" * 64,
    )
    with pytest.raises(RuntimeError, match="one-image readiness"):
        _require_proposal_readiness(
            [_readiness_row(0, status="error")],
            [_readiness_row(0, status="error")],
            pilot,
            pilot,
            dispositions,
            pilot_history_count_before_resume=20,
            pilot_history_count_after_resume=20,
        )

    duplicated = list(pilot)
    duplicated[-1] = dict(duplicated[0])
    with pytest.raises(RuntimeError, match="pilot coverage"):
        _require_proposal_readiness(
            [_readiness_row(0)],
            [_readiness_row(0)],
            duplicated,
            pilot,
            dispositions,
            pilot_history_count_before_resume=20,
            pilot_history_count_after_resume=20,
        )


def test_readiness_gates_reject_resume_that_adds_an_attempt() -> None:
    pilot = [_readiness_row(index) for index in range(20)]
    dispositions = build_frozen_automatic_proposals(
        pilot,
        policy_key="raw",
        freeze_sha256="c" * 64,
    )
    with pytest.raises(RuntimeError, match="pilot resume"):
        _require_proposal_readiness(
            [_readiness_row(0)],
            [_readiness_row(0)],
            pilot,
            pilot,
            dispositions,
            pilot_history_count_before_resume=20,
            pilot_history_count_after_resume=21,
        )


def test_two_automatic_candidates_are_recorded_as_unscoreable() -> None:
    candidates = [
        ConfirmationCandidate(
            opaque_scene_id="S000001",
            view_id="V00",
            route="automatic_box",
            peer_key=f"P{index}",
            candidate_id=f"C{index}",
            image_path=Path("unused.png"),
            image_sha256="a" * 64,
            box_xyxy=(0, 0, 10, 10),
            foreground_pixels=100,
            predicted_quality_score=0.9,
            mask_rle={},
            mask_sha256="b" * 64,
        )
        for index in range(2)
    ]
    globals_ = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    patches = [np.asarray([[1.0, 0.0]]), np.asarray([[0.0, 1.0]])]
    valid = [np.ones((1, 1), dtype=bool), np.ones((1, 1), dtype=bool)]

    rows, heatmaps = _score_group(candidates, globals_, patches, valid)

    assert len(rows) == 4
    assert {row["scoring_status"] for row in rows} == {"insufficient_peers"}
    assert all(row["score"] is None and row["rank"] is None for row in rows)
    assert len(heatmaps) == 2


def test_representation_retains_invalid_mask_and_encodes_only_valid_crops(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 32), "white").save(image_path)
    valid_masks = []
    for start in (2, 11, 20):
        mask = np.zeros((32, 32), dtype=bool)
        mask[start : start + 7, start : start + 7] = True
        valid_masks.append(mask)
    invalid_mask = np.zeros((32, 32), dtype=bool)
    invalid_mask[0, 0] = True
    candidates = [
        *[
            _candidate(image_path, index=index, mask=mask)
            for index, mask in enumerate(valid_masks)
        ],
        _candidate(image_path, index=3, mask=invalid_mask),
    ]

    rows, arrays, audit = _score_scene(
        scene_id="S000001",
        scene_groups=[(("S000001", "V00", "automatic_box"), candidates)],
        encoder=ContractVisualEncoder(),
        crop_configuration=CropConfiguration(),
        batch_size=8,
    )

    invalid_rows = [row for row in rows if row["candidate_id"] == "C3"]
    assert len(rows) == 8
    assert {row["method"] for row in invalid_rows} == {"B0", "P1"}
    assert {row["scoring_status"] for row in invalid_rows} == {"invalid_mask"}
    assert all(row["score"] is None and row["rank"] is None for row in invalid_rows)
    assert all(
        row["scoring_reason"] == "foreground_pixels_below_minimum"
        for row in invalid_rows
    )
    assert arrays["candidate_ids"].tolist() == ["C0", "C1", "C2"]
    assert arrays["p1_heatmaps"].shape[0] == 3
    assert audit["candidate_count"] == 4
    assert audit["encoded_candidate_count"] == 3
    assert audit["invalid_mask_candidate_count"] == 1


def test_representation_scene_with_only_invalid_masks_does_not_call_encoder(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 32), "white").save(image_path)
    invalid_mask = np.zeros((32, 32), dtype=bool)
    invalid_mask[0, 0] = True
    candidates = [
        _candidate(image_path, index=index, mask=invalid_mask)
        for index in range(3)
    ]

    rows, arrays, audit = _score_scene(
        scene_id="S000001",
        scene_groups=[(("S000001", "V00", "automatic_box"), candidates)],
        encoder=ContractVisualEncoder(),
        crop_configuration=CropConfiguration(),
        batch_size=8,
    )

    assert len(rows) == 6
    assert {row["scoring_status"] for row in rows} == {"invalid_mask"}
    assert arrays["candidate_ids"].size == 0
    assert arrays["p1_heatmaps"].shape == (0, 0, 0)
    assert audit["encoded_candidate_count"] == 0
    assert audit["invalid_mask_candidate_count"] == 3
    assert audit["encoder_call_count"] == 0


def test_evaluator_keeps_misses_and_extras_after_prediction_lock() -> None:
    references = [
        {
            "opaque_scene_id": "S000001",
            "view_id": "V00",
            "opaque_object_id": object_id,
            "visible": True,
            "box_xyxy": box,
            "foreground_pixels": 100,
        }
        for object_id, box in (
            ("O1", [0, 0, 20, 20]),
            ("O2", [30, 0, 50, 20]),
            ("O3", [60, 0, 80, 20]),
        )
    ]
    automatic_sam = [
        {
            "opaque_scene_id": "S000001",
            "view_id": "V00",
            "routes": {
                "automatic_box": [
                    {"prompt_key": "P1", "box_xyxy": [0, 0, 20, 20]},
                    {"prompt_key": "P2", "box_xyxy": [100, 100, 120, 120]},
                ]
            },
            "automatic_proposal_provenance": _automatic_provenance(
                proposal_count=2
            ),
        }
    ]
    scores = []
    for method in ("B0", "P1"):
        for peer_key in ("P1", "P2"):
            scores.append(
                {
                    "schema_version": 1,
                    "candidate_id": f"{method}-{peer_key}",
                    "opaque_scene_id": "S000001",
                    "view_id": "V00",
                    "route": "automatic_box",
                    "peer_key": peer_key,
                    "encoder_key": "dinov2_small",
                    "method": method,
                    "scoring_status": "insufficient_peers",
                    "score": None,
                    "rank": None,
                    "candidate_count": 2,
                    "foreground_pixels": 100,
                    "predicted_mask_quality": 0.9,
                }
            )

    evaluator_rows, association = prepare_missing_aware_evaluator_scores(
        scores,
        references,
        automatic_sam,
    )

    assert association[0]["matched_count"] == 1
    assert association[0]["miss_count"] == 2
    assert association[0]["extra_count"] == 1
    assert sum(row["association_status"] == "missed_reference" for row in evaluator_rows) == 4
    assert sum(row["association_status"] == "extra_proposal" for row in evaluator_rows) == 2


def test_evaluator_distinguishes_terminal_failure_from_genuine_zero_proposals() -> None:
    references = [
        {
            "opaque_scene_id": "S000001",
            "view_id": "V00",
            "opaque_object_id": f"O{index}",
            "visible": True,
            "box_xyxy": [index * 20, 0, index * 20 + 10, 10],
            "foreground_pixels": 100,
        }
        for index in range(3)
    ]
    failed_sam = [{
        "opaque_scene_id": "S000001",
        "view_id": "V00",
        "routes": {"automatic_box": []},
        "automatic_proposal_provenance": _automatic_provenance(
            status="error",
            proposal_count=0,
            raw_proposal_count=2,
            discarded_untrusted_box_count=2,
        ),
    }]

    evaluator_rows, failed_association = prepare_missing_aware_evaluator_scores(
        [],
        references,
        failed_sam,
    )
    failed_summary = _proposal_summary(failed_association)[0]

    assert len(evaluator_rows) == 6
    assert failed_association[0]["source_status"] == "error"
    assert failed_association[0]["discarded_untrusted_box_count"] == 2
    assert failed_summary["terminal_failed_image_count"] == 1
    assert failed_summary["genuine_zero_proposal_image_count"] == 0
    assert failed_summary["discarded_untrusted_box_count"] == 2

    genuine_sam = [{
        "opaque_scene_id": "S000001",
        "view_id": "V00",
        "routes": {"automatic_box": []},
        "automatic_proposal_provenance": _automatic_provenance(
            status="ok",
            proposal_count=0,
            raw_proposal_count=1,
            removed_proposal_count=1,
        ),
    }]
    _rows, genuine_association = prepare_missing_aware_evaluator_scores(
        [],
        references,
        genuine_sam,
    )
    genuine_summary = _proposal_summary(genuine_association)[0]
    assert genuine_summary["terminal_failed_image_count"] == 0
    assert genuine_summary["genuine_zero_proposal_image_count"] == 1


def test_evaluator_keeps_invalid_mask_as_detected_zero_ranking_evidence() -> None:
    boxes = ([0, 0, 20, 20], [30, 0, 50, 20], [60, 0, 80, 20])
    references = [
        {
            "opaque_scene_id": "S000001",
            "view_id": "V00",
            "opaque_object_id": f"O{index}",
            "visible": True,
            "box_xyxy": box,
            "foreground_pixels": 100,
        }
        for index, box in enumerate(boxes, start=1)
    ]
    automatic_sam = [
        {
            "opaque_scene_id": "S000001",
            "view_id": "V00",
            "routes": {
                "automatic_box": [
                    {"prompt_key": f"P{index}", "box_xyxy": box}
                    for index, box in enumerate(boxes, start=1)
                ]
            },
            "automatic_proposal_provenance": _automatic_provenance(
                proposal_count=3
            ),
        }
    ]
    scores = []
    for method in ("B0", "P1"):
        for index in range(1, 4):
            invalid = index == 3
            scores.append(
                {
                    "schema_version": 1,
                    "candidate_id": f"{method}-P{index}",
                    "opaque_scene_id": "S000001",
                    "view_id": "V00",
                    "route": "automatic_box",
                    "peer_key": f"P{index}",
                    "encoder_key": "dinov2_small",
                    "method": method,
                    "scoring_status": "invalid_mask" if invalid else "scored",
                    "scoring_reason": (
                        "foreground_pixels_below_minimum" if invalid else None
                    ),
                    "score": None if invalid else 1.0 / index,
                    "rank": None if invalid else index,
                    "candidate_count": 3,
                    "foreground_pixels": 1 if invalid else 100,
                    "predicted_mask_quality": 0.9,
                }
            )

    evaluator_rows, _association = prepare_missing_aware_evaluator_scores(
        scores,
        references,
        automatic_sam,
    )
    invalid_rows = [
        row for row in evaluator_rows if row["scoring_status"] == "invalid_mask"
    ]
    assert len(invalid_rows) == 2
    assert all(row["detected"] is True for row in invalid_rows)
    assert all(row["score"] is None and row["rank"] is None for row in invalid_rows)

    p1_rows = [row for row in evaluator_rows if row["method"] == "P1"]
    fused, _ = fuse_object_ranks(
        p1_rows,
        selected_views=("V00",),
        fusion_method="mean_rank",
    )
    invalid_object = next(row for row in fused if row["opaque_object_id"] == "O3")
    assert invalid_object["fused_score"] == 0.0
    assert invalid_object["view_support"] == 1
    assert invalid_object["detected_view_support"] == 1
    assert invalid_object["scored_view_support"] == 0
    assert invalid_object["unscoreable_detected_view_count"] == 1

    summary = _unscoreable_summary(evaluator_rows)
    automatic_p1 = next(
        row
        for row in summary
        if row["route"] == "automatic_box" and row["method"] == "P1"
    )
    assert automatic_p1["unscoreable_candidate_view_count"] == 1
    assert automatic_p1["invalid_mask_candidate_view_count"] == 1


def test_missing_views_reduce_fused_automatic_score() -> None:
    rows = [
        {
            "view_id": "V00",
            "opaque_object_id": "O1",
            "rank": 1,
            "candidate_count": 3,
            "foreground_pixels": 400,
            "detected": True,
        },
        {
            "view_id": "V00",
            "opaque_object_id": "O2",
            "rank": 2,
            "candidate_count": 3,
            "foreground_pixels": 400,
            "detected": True,
        },
        {
            "view_id": "V00",
            "opaque_object_id": "O3",
            "rank": 3,
            "candidate_count": 3,
            "foreground_pixels": 400,
            "detected": True,
        },
    ]

    fused, _ = fuse_object_ranks(
        rows,
        selected_views=("V00", "V10"),
        fusion_method="mean_rank",
    )

    first = next(row for row in fused if row["opaque_object_id"] == "O1")
    assert first["view_support"] == 1
    assert first["detected_view_support"] == 1
    assert first["scored_view_support"] == 1
    assert first["missing_view_count"] == 1
    assert first["unscoreable_detected_view_count"] == 0
    assert first["fused_score"] == 0.5


def test_detected_and_scored_support_are_reported_separately() -> None:
    rows = []
    for object_id, rank in (("O1", 1), ("O2", 2), ("O3", 3)):
        rows.extend(
            [
                {
                    "view_id": "V00",
                    "opaque_object_id": object_id,
                    "rank": rank,
                    "candidate_count": 3,
                    "foreground_pixels": 400,
                    "detected": True,
                    "scoring_status": "scored",
                },
                {
                    "view_id": "V10",
                    "opaque_object_id": object_id,
                    "rank": None,
                    "candidate_count": 3,
                    "foreground_pixels": 1,
                    "detected": True,
                    "scoring_status": "invalid_mask",
                },
            ]
        )

    fused, _ = fuse_object_ranks(
        rows,
        selected_views=("V00", "V10"),
        fusion_method="mean_rank",
    )

    first = next(row for row in fused if row["opaque_object_id"] == "O1")
    assert first["fused_score"] == 0.5
    assert first["detected_view_support"] == 2
    assert first["scored_view_support"] == 1
    assert first["missing_view_count"] == 0
    assert first["unscoreable_detected_view_count"] == 1
