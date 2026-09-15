from __future__ import annotations

import inspect

from visual_peer_inspector.representation.patch_correspondence import PATCH_CONFIGURATIONS
from visual_peer_inspector.workflows import identity_patch_association_study as study


def test_protocol_reuses_frozen_development_boundary() -> None:
    assert study.PROTOCOL["split_change"] is False
    assert study.PROTOCOL["scope"] == "existing nested 100-scene development screen"
    assert study.PROTOCOL["encoder"] == "facebook/dinov2-small"
    assert study.PRIMARY_METHOD == "P1_current"
    assert study.VIEW_SETS == {
        2: ("V00", "V10"),
        3: ("V00", "V07", "V14"),
        5: ("V00", "V04", "V08", "V12", "V16"),
    }


def test_comparison_is_bounded_and_predeclared() -> None:
    assert [item.name for item in PATCH_CONFIGURATIONS] == [
        "reciprocal_hungarian_035",
        "sinkhorn_hungarian_035",
        "sinkhorn_consensus_035",
    ]
    assert all(item.global_weight == 0.50 for item in PATCH_CONFIGURATIONS)
    assert all(item.maximum_cost == 0.35 for item in PATCH_CONFIGURATIONS)


def test_labels_are_not_arguments_to_prediction() -> None:
    signature = inspect.signature(study.predict)
    forbidden = {"evaluator_root", "labels", "truth", "crosswalk"}
    assert forbidden.isdisjoint(signature.parameters)


def test_evaluation_requires_a_sealed_prediction() -> None:
    source = inspect.getsource(study.evaluate)
    assert source.index('verify_seal(prediction_root, name="prediction_lock.json")') < source.index(
        'read_json(evaluator_root / "index.json")'
    )


def test_promotion_rule_contains_fragmentation_guards() -> None:
    source = inspect.getsource(study.evaluate)
    assert "linked_node_fraction >= baseline.linked_node_fraction" in source
    assert "singleton_fraction <= baseline.singleton_fraction" in source
    assert 'float(identity_interval["ci_low"]) > 0.0' in source
