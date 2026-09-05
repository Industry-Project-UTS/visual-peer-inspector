from __future__ import annotations

import pytest

from visual_peer_inspector.workflows.locateanything_prompt_freeze_ds1 import (
    PROMPT_CANDIDATES,
    f_beta,
    prompt_configurations,
    select_prompt_configuration,
)


def _row(key: str, *, precision: float, recall: float, complete: float, proposals: int):
    return {
        "candidate_key": key,
        "prompt_key": key.split("__")[0],
        "prompt": key,
        "postprocessing_policy": "raw",
        "image_count": 2_000,
        "precision_at_0_50": precision,
        "recall_at_0_50": recall,
        "f1_at_0_50": 2 * precision * recall / (precision + recall),
        "f2_at_0_50": f_beta(precision, recall),
        "complete_detection_rate": complete,
        "proposal_count": proposals,
    }


def test_prompt_registry_is_small_generic_and_pinned() -> None:
    assert [key for key, _prompt in PROMPT_CANDIDATES] == [
        "object",
        "individual_object",
    ]
    configurations = prompt_configurations()
    assert set(configurations) == {key for key, _prompt in PROMPT_CANDIDATES}
    assert {configuration.model_revision for configuration in configurations.values()} == {
        "c32291ca5e996f5a7a485845b4f57a233936bba0"
    }


def test_f2_emphasises_recall_without_a_fixed_cutoff() -> None:
    assert f_beta(0.90, 0.94) == pytest.approx(0.9317181)
    assert f_beta(0.90, 0.96) > f_beta(0.90, 0.94)


def test_selection_uses_f2_not_the_historical_point_target() -> None:
    selected = select_prompt_configuration([
        _row("object__raw", precision=0.91, recall=0.949, complete=0.88, proposals=8_000),
        _row("individual_object__raw", precision=0.88, recall=0.960, complete=0.90, proposals=8_400),
    ])
    assert selected.candidate_key == "individual_object__raw"


def test_selection_rejects_incomplete_candidates() -> None:
    row = _row("object__raw", precision=0.9, recall=0.9, complete=0.8, proposals=8_000)
    row["image_count"] = 1_999
    with pytest.raises(ValueError, match="No prompt candidate"):
        select_prompt_configuration([row])
