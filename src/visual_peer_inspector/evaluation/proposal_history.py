"""Strict resolution of append-only LocateAnything proposal histories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from visual_peer_inspector.data.model_safe_package import file_sha256
from visual_peer_inspector.models.locateanything import ProposalResultStore


@dataclass(frozen=True)
class ResolvedProposalHistory:
    """A validated full history and its single latest row per image."""

    history_rows: tuple[dict[str, Any], ...]
    latest_rows: tuple[dict[str, Any], ...]
    proposal_results_sha256: str


def resolve_committed_proposal_history(
    path: Path,
    *,
    configuration_sha256: str,
    producer_kind: str,
    expected_file_sha256: str,
    expected_result_count: int,
    expected_cohort_id: str,
    expected_input_manifest_sha256: str,
    expected_model_id: str,
    expected_model_revision: str,
) -> ResolvedProposalHistory:
    """Validate every retry row and return one accepted latest row per image."""

    path = path.expanduser().resolve()
    observed_sha256 = file_sha256(path)
    if observed_sha256 != expected_file_sha256:
        raise ValueError("Proposal history differs from its committed file hash")

    store = ProposalResultStore(path, configuration_sha256, producer_kind)
    history_rows = store.history_rows
    latest_rows = store.rows
    if not history_rows:
        raise ValueError("Proposal history is empty")

    previous_by_image: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in history_rows:
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        previous = previous_by_image.get(key)
        if previous is not None and previous.get("status") == "ok":
            raise ValueError("Proposal retry history continues after success")
        previous_by_image[key] = row
        if row.get("cohort_id") != expected_cohort_id:
            raise ValueError("Proposal history identifies another cohort")
        if row.get("input_manifest_sha256") != expected_input_manifest_sha256:
            raise ValueError("Proposal history identifies another model-input manifest")
        if row.get("model_id") != expected_model_id:
            raise ValueError("Proposal history identifies another model")
        if row.get("model_revision") != expected_model_revision:
            raise ValueError("Proposal history identifies another model revision")

    if len(latest_rows) != expected_result_count:
        raise ValueError("Proposal latest-row coverage differs from the committed count")
    if any(
        row.get("status") != "ok" or row.get("producer_kind") != producer_kind
        for row in latest_rows
    ):
        raise ValueError("Proposal latest rows are not accepted model outputs")

    return ResolvedProposalHistory(
        history_rows=tuple(history_rows),
        latest_rows=tuple(latest_rows),
        proposal_results_sha256=observed_sha256,
    )


__all__ = ["ResolvedProposalHistory", "resolve_committed_proposal_history"]
