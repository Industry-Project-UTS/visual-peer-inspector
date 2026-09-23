"""Frozen 09B review routing uses only model-visible ranking evidence."""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from visual_peer_inspector.web.core import RankedObject
from visual_peer_inspector.web.frozen_policy import (
    POLICY_PATH,
    POLICY_SHA256,
    load_frozen_policy,
)


def ranking(scores, support):
    return tuple(
        RankedObject(f"object_{i}", score, tuple(f"view_{j}" for j in range(support)))
        for i, score in enumerate(scores)
    )


def test_bundled_policy_matches_sealed_calibration_bytes():
    policy = load_frozen_policy()
    assert hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest() == POLICY_SHA256
    assert policy.version == "calibration-selective-v1"
    assert policy.thresholds == {2: .40, 3: .25, 5: .25}


@pytest.mark.parametrize("mutation", [
    lambda value: value.pop("assisted"),
    lambda value: value.update(extra="unexpected"),
    lambda value: value.update(version="changed"),
    lambda value: value["assisted"].update(confidence_threshold_by_view_count={"2": .40}),
])
def test_policy_rejects_missing_extra_or_altered_fields(tmp_path, monkeypatch, mutation):
    value = json.loads(POLICY_PATH.read_text())
    mutation(value)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="integrity hash"):
        load_frozen_policy(path)
    monkeypatch.setattr("visual_peer_inspector.web.frozen_policy.POLICY_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="contract|fields|thresholds"):
        load_frozen_policy(path)


def test_confidence_uses_evaluator_score_gap_and_normalized_softmax_entropy():
    policy = load_frozen_policy()
    ranks = ranking((.90, .40, .10), 3)
    signals = policy.signals(ranks, "assisted", 3)
    weights = [math.exp(score - .90) for score in (.90, .40, .10)]
    probabilities = [weight / sum(weights) for weight in weights]
    entropy = -sum(p * math.log(max(p, 1e-15)) for p in probabilities) / max(math.log(3), 1.0)
    assert signals["score_margin"] == pytest.approx(.50)
    assert signals["normalized_score_entropy"] == pytest.approx(entropy)
    assert signals["confidence"] == pytest.approx(.5 * .50 + .5 * (1 - entropy))
    assert {key: signals[key] for key in ("mode", "view_count", "ranked_count", "top_tie_count", "winner_support")} == {
        "mode": "assisted", "view_count": 3, "ranked_count": 3, "top_tie_count": 1, "winner_support": 3,
    }


def test_three_policy_states_and_automatic_review():
    policy = load_frozen_policy()
    assert policy.decide(policy.signals((), "automatic", 2))[0] == "insufficient_evidence"
    available = policy.signals(ranking((1.0, 0.0), 2), "automatic", 2)
    assert policy.decide(available)[0] == "review_required"
    assert policy.decide({**available, "mode": "assisted"}, correspondence_confirmed=True)[0] == "result_ready"


@pytest.mark.parametrize("count,threshold", [(2, .40), (3, .25), (5, .25)])
def test_assisted_thresholds_are_inclusive(count, threshold):
    policy = load_frozen_policy()
    signals = policy.signals(ranking((1.0, 0.0), count), "assisted", count)
    assert policy.decide({**signals, "confidence": threshold}, correspondence_confirmed=True)[0] == "result_ready"
    assert policy.decide({**signals, "confidence": threshold - 1e-6}, correspondence_confirmed=True)[0] == "review_required"
    assert policy.decide(signals, correspondence_confirmed=False)[0] == "review_required"


def test_uncalibrated_count_tie_and_missing_winner_link_require_review():
    policy = load_frozen_policy()
    for count in (2, 4):
        signals = policy.signals(ranking((1.0, 0.0), count), "assisted", count)
        assert policy.decide({**signals, "view_count": 4}, correspondence_confirmed=True)[0] == "review_required"
        assert policy.decide({**signals, "winner_support": count - 1}, correspondence_confirmed=True)[0] == "review_required"
    tied = policy.signals(ranking((.8, .8), 3), "assisted", 3)
    assert tied["top_tie_count"] == 2
    assert policy.decide(tied, correspondence_confirmed=True)[0] == "review_required"
