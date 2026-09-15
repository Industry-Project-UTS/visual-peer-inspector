from __future__ import annotations

from itertools import permutations

import numpy as np

from visual_peer_inspector.workflows.learned_peer_evaluation import (
    RankGroup,
    _rank_metrics,
    fit_linear_ranker,
    score_linear_ranker,
)


def groups() -> list[RankGroup]:
    values = []
    for index in range(12):
        features = np.asarray(
            [
                [0.0, float(index % 3), 0.2],
                [1.0, float(index % 3), 0.3],
                [2.0, float(index % 3), 0.4],
            ],
            dtype=np.float64,
        )
        values.append(
            RankGroup(
                scene=f"S{index:06d}",
                asset_group=f"A{index // 2:02d}",
                route="automatic",
                view_set_id="three-primary",
                view="V00",
                candidate_ids=("C0", "C1", "C2"),
                features=features,
                labels=np.asarray([0, 0, 1], dtype=np.int8),
            )
        )
    return values


def test_all_objectives_rank_the_positive_candidate_first():
    for objective in ("listwise", "ranknet", "hybrid"):
        model = fit_linear_ranker(groups(), objective=objective, regularization=0.1)
        scores = score_linear_ranker(groups()[0], model)
        assert int(np.argmax(scores)) == 2
        assert model["scoreable_training_groups"] == len(groups())


def test_training_is_deterministic():
    first = fit_linear_ranker(groups(), objective="hybrid")
    second = fit_linear_ranker(groups(), objective="hybrid")
    assert first == second


def test_invalid_objective_is_rejected():
    try:
        fit_linear_ranker(groups(), objective="unknown")
    except ValueError as error:
        assert "Unknown" in str(error)
    else:
        raise AssertionError("The invalid objective was accepted")


def test_equal_scores_do_not_gain_credit_from_candidate_id_order():
    metrics = _rank_metrics(
        [
            {"id": "a-negative", "score": 1.0, "reference": "normal"},
            {"id": "z-positive", "score": 1.0, "reference": "anomaly"},
        ],
        {"normal": 0, "anomaly": 1},
    )
    assert metrics["top1"] == 0.0
    assert metrics["top1_tie_expected"] == 0.5
    assert metrics["ap_tie_expected"] == 0.75
    assert metrics["recall_at_2_tie_expected"] == 1.0
    assert metrics["reciprocal_rank_tie_expected"] == 0.75


def test_tie_expected_metrics_are_invariant_to_identifier_order():
    observations = [
        (2.0, "anomaly-a"),
        (2.0, "normal-a"),
        (1.0, "anomaly-b"),
        (1.0, "normal-b"),
    ]
    expected = set()
    deterministic = set()
    for identifiers in permutations(("a", "b", "c", "d")):
        metrics = _rank_metrics(
            [
                {"id": identifier, "score": score, "reference": reference}
                for identifier, (score, reference) in zip(identifiers, observations, strict=True)
            ],
            {"anomaly-a": 1, "normal-a": 0, "anomaly-b": 1, "normal-b": 0},
        )
        expected.add(
            (
                metrics["top1_tie_expected"],
                metrics["ap_tie_expected"],
                metrics["recall_at_2_tie_expected"],
                metrics["reciprocal_rank_tie_expected"],
            )
        )
        deterministic.add(metrics["top1"])
    assert len(expected) == 1
    assert deterministic == {0.0, 1.0}
