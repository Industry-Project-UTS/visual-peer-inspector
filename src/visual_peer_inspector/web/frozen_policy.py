"""The sealed calibration review rule for the two-mode upload application."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .core import RankedObject

POLICY_PATH = Path(__file__).with_name("frozen_review_policy.json")
POLICY_SOURCE = "src/visual_peer_inspector/web/frozen_review_policy.json"
POLICY_SHA256 = "f36e3a61ac993602611ca01fdd5cc0f053c80a632ea3c215f8654931814ca8ff"
POLICY_SCHEMA = "vpi-current-two-mode-review-policy/v1"
POLICY_VERSION = "calibration-selective-v1"


@dataclass(frozen=True)
class FrozenPolicy:
    version: str
    source_sha256: str
    thresholds: dict[int, float]

    def signals(self, ranking: tuple[RankedObject, ...], mode: str, view_count: int) -> dict:
        """Use the same score-gap and softmax entropy calculation as 08M/09B."""
        scores = [float(item.relative_score) for item in ranking]
        top = scores[0] if scores else 0.0
        second = scores[1] if len(scores) > 1 else top
        margin = top - second
        if scores:
            weights = [math.exp(score - top) for score in scores]
            total = sum(weights)
            probabilities = [weight / total for weight in weights]
            entropy = -sum(p * math.log(max(p, 1e-15)) for p in probabilities)
            entropy /= max(math.log(len(scores)), 1.0)
            top_tie_count = sum(math.isclose(score, top, abs_tol=1e-12, rel_tol=0.0) for score in scores)
        else:
            entropy = 1.0
            top_tie_count = 0
        return {
            "mode": mode,
            "view_count": view_count,
            "ranked_count": len(scores),
            "top_tie_count": top_tie_count,
            "confidence": 0.5 * margin + 0.5 * (1.0 - entropy),
            "winner_support": len(ranking[0].view_ids) if ranking else 0,
            "score_margin": margin,
            "normalized_score_entropy": entropy,
        }

    def decide(self, signals: dict, *, correspondence_confirmed: bool = False) -> tuple[str, tuple[str, ...]]:
        if signals["ranked_count"] == 0:
            return "insufficient_evidence", ("No physical object has a usable combined ranking.",)
        if signals["mode"] == "automatic":
            return "review_required", ("Automatic matching requires review before treating the selection as ready.",)
        if signals["mode"] != "assisted":
            raise ValueError("Unknown inspection mode")
        reasons = []
        count = signals["view_count"]
        if not correspondence_confirmed:
            reasons.append("Confirm which physical objects match across views.")
        if count not in self.thresholds:
            reasons.append("This view count has no calibrated readiness threshold.")
        if signals["top_tie_count"] != 1:
            reasons.append("The highest combined scores are tied.")
        if signals["winner_support"] < count:
            reasons.append("The selected object is not linked in every uploaded view.")
        threshold = self.thresholds.get(count)
        if threshold is None or signals["confidence"] < threshold:
            reasons.append("The ranking confidence index is below the calibrated review threshold.")
        return ("review_required", tuple(reasons)) if reasons else ("result_ready", ())


def load_frozen_policy(path: str | Path | None = None) -> FrozenPolicy:
    """The deployable policy must be byte-identical to the sealed 09B artifact."""
    source = Path(path) if path is not None else POLICY_PATH
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != POLICY_SHA256:
        raise ValueError("Review policy integrity hash differs from the sealed calibration artifact.")
    data = json.loads(raw)
    expected = {
        "asset_group_count", "assisted", "automatic", "calibration_role", "confidence_formula",
        "deployment_status", "evaluation_claim", "evaluation_lock_sha256", "independent_confirmation",
        "method_freeze_sha256", "policy_finalizer_sha256", "policy_signal_fields",
        "prediction_lock_sha256", "protocol_lock_sha256", "scene_count", "schema", "selection",
        "states", "version",
    }
    if set(data) != expected or data["schema"] != POLICY_SCHEMA or data["version"] != POLICY_VERSION:
        raise ValueError("Review policy schema or version differs from the frozen contract.")
    if data["policy_signal_fields"] != ["mode", "view_count", "ranked_count", "top_tie_count", "confidence", "winner_support"]:
        raise ValueError("Review policy signal schema differs from the frozen contract.")
    if data["confidence_formula"] != "0.5 * score_margin + 0.5 * (1 - normalized_score_entropy)":
        raise ValueError("Review policy confidence formula differs from the frozen contract.")
    if data["states"] != ["result_ready", "review_required", "insufficient_evidence"]:
        raise ValueError("Review policy states differ from the frozen contract.")
    if data["scene_count"] != 600 or data["asset_group_count"] != 163 or data["calibration_role"] != "calibration" or data["independent_confirmation"] is not False:
        raise ValueError("Review policy calibration provenance differs from the frozen contract.")
    automatic = data["automatic"]
    assisted = data["assisted"]
    if set(automatic) != {"result_ready_enabled", "available_result_state", "reason"} or automatic["result_ready_enabled"] is not False or automatic["available_result_state"] != "review_required":
        raise ValueError("Automatic review rule differs from the frozen contract.")
    if set(assisted) != {"correspondence_requirement", "supported_ready_view_counts", "confidence_threshold_by_view_count", "require_unique_top_score", "require_winner_support_in_every_uploaded_view", "uncalibrated_view_count_state"}:
        raise ValueError("Assisted review rule fields differ from the frozen contract.")
    if assisted["correspondence_requirement"] != "dataset_reference_or_user_confirmed" or assisted["require_unique_top_score"] is not True or assisted["require_winner_support_in_every_uploaded_view"] is not True or assisted["uncalibrated_view_count_state"] != "review_required":
        raise ValueError("Assisted review rule differs from the frozen contract.")
    counts = assisted["supported_ready_view_counts"]
    thresholds = assisted["confidence_threshold_by_view_count"]
    if counts != [2, 3, 5] or set(thresholds) != {"2", "3", "5"}:
        raise ValueError("Assisted view-count thresholds differ from the frozen contract.")
    if any(type(thresholds[str(count)]) not in (int, float) or not math.isfinite(thresholds[str(count)]) or not 0 <= thresholds[str(count)] <= 1 for count in counts):
        raise ValueError("Invalid assisted review threshold.")
    return FrozenPolicy(POLICY_VERSION, digest, {count: float(thresholds[str(count)]) for count in counts})
