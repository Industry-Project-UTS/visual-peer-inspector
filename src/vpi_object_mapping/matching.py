"""Joint, partial physical-object assignment using fixed DINO global features.

Each feasible track is a clique of cross-view observations at cosine distance
at most 0.35. A binary set partition covers every observation exactly once.
Its objective is the sum of (0.35 - distance) over links inside chosen tracks.
This rewards supported links and consequently favours longer compatible tracks;
it is not a likelihood, and indistinguishable objects remain ambiguous.

The best three partitions are found using no-good cuts. They are alternatives
under this appearance model, not calibrated posterior samples. The primary map
must be proved optimal; otherwise all observations receive explicit singleton
fallback tracks. No object scores, reference identities or evaluator fields are
accepted. Solver deadlines are cooperative HiGHS limits, with separate finite
enumeration caps; they are not an operating-system hard process timeout.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
import math
import re
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, vstack


SCHEMA = "vpi-joint-partial-assignment/v1"
DISTANCE_THRESHOLD = 0.35
FEATURE_DIMENSION = 384
MAX_OBSERVATIONS = 64
MAX_SELECTED_VIEWS = 5
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class _BudgetExceeded(Exception):
    pass


def _token(value, name):
    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        raise ValueError(f"{name} must be an opaque token")
    return value


def _positive_integer(value, name, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds the frozen bound")


def _inputs(observations, selected_views):
    if isinstance(selected_views, (str, bytes)):
        raise ValueError("selected_views must be an explicit sequence")
    views = [_token(view, "view_id") for view in selected_views]
    if not 1 <= len(views) <= MAX_SELECTED_VIEWS or len(set(views)) != len(views):
        raise ValueError("Select one to five distinct views")
    rows, keys = [], set()
    for observation in observations:
        if not isinstance(observation, Mapping) or set(observation) != {"view_id", "candidate_id", "embedding"}:
            raise ValueError("Observations accept only view_id, candidate_id and embedding")
        view = _token(observation["view_id"], "view_id")
        candidate = _token(observation["candidate_id"], "candidate_id")
        if view not in views:
            raise ValueError("Observation is outside selected views")
        if (view, candidate) in keys:
            raise ValueError("Duplicate observation in the same view")
        keys.add((view, candidate))
        vector = np.asarray(observation["embedding"], dtype=np.float64)
        if vector.shape != (FEATURE_DIMENSION,) or not np.isfinite(vector).all():
            raise ValueError("Expected one finite 384-dimensional DINO global embedding")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("DINO embedding must have finite nonzero norm")
        rows.append({"view_id": view, "candidate_id": candidate, "vector": vector / norm})
    rows.sort(key=lambda row: (row["view_id"], row["candidate_id"]))
    return rows, sorted(views)


def _enumerate_tracks(compatible, max_tracks, deadline):
    """Enumerate cliques once, retaining singletons and never crossing a view."""
    tracks = []

    def extend(prefix, available):
        if time.monotonic() >= deadline:
            raise _BudgetExceeded("enumeration_time_limit")
        if len(tracks) >= max_tracks:
            raise _BudgetExceeded("candidate_track_limit")
        tracks.append(prefix)
        for position, node in enumerate(available):
            remaining = tuple(other for other in available[position + 1:] if compatible[node, other])
            extend(prefix + (node,), remaining)

    for node in range(len(compatible)):
        extend((node,), tuple(np.flatnonzero(compatible[node] & (np.arange(len(compatible)) > node))))
    return sorted(tracks)


def _partition(indices, tracks, rows, benefits, *, optimal, source, rank=None, best=None):
    chosen = sorted((tracks[index] for index in indices), key=lambda track: tuple(track))
    objects = []
    for number, track in enumerate(chosen, 1):
        objects.append({
            "identity_id": f"joint-{number:04d}",
            "members": [{"view_id": rows[index]["view_id"], "candidate_id": rows[index]["candidate_id"]} for index in track],
            "status": "linked" if len(track) > 1 else "unmatched",
        })
    objective = float(sum(benefits[index] for index in indices))
    return {"tracks": objects, "objective": objective, "optimal": bool(optimal), "source": source,
            "rank": rank, "objective_gap_from_best": None if best is None else float(max(0.0, best - objective))}


def _solution_indices(solution, incidence, count):
    """Reject a malformed incumbent rather than turning fractional values into a map."""
    if solution is None:
        return None
    values = np.asarray(solution, dtype=float)
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError("Solver returned an invalid solution vector")
    rounded = np.rint(values)
    if np.any(np.abs(values - rounded) > 1e-6) or np.any((rounded != 0) & (rounded != 1)):
        raise ValueError("Solver returned a nonbinary assignment")
    if not np.array_equal(np.asarray(incidence @ rounded).reshape(-1), np.ones(incidence.shape[0])):
        raise ValueError("Solver assignment does not cover every observation exactly once")
    return tuple(int(index) for index in np.flatnonzero(rounded))


def _finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def match_scene(observations, *, selected_views, time_limit_seconds=5.0, max_hypotheses=3, max_tracks=50000):
    """Return a complete best map and up to three globally ranked alternatives.

    ``observations`` contains opaque ``view_id``, ``candidate_id`` and fixed
    384-dimensional ``embedding`` records from exactly the selected images.
    Empty selected views are permitted. All output tracks, including singleton
    hypotheses, have IDs; ``status='unmatched'`` exposes unresolved association.

    A bounded failure returns ``status='fallback'`` and a singleton ``primary``.
    Any unproved feasible incumbent is separate and must not be used as a proven
    optimum. An alternative-search timeout preserves an already proved primary.
    """
    if isinstance(time_limit_seconds, bool) or not isinstance(time_limit_seconds, (int, float)) or not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive and finite")
    _positive_integer(max_hypotheses, "max_hypotheses", 3)
    _positive_integer(max_tracks, "max_tracks", 50000)
    started = time.monotonic()
    deadline = started + float(time_limit_seconds)
    rows, views = _inputs(observations, selected_views)
    n = len(rows)
    singletons = [(index,) for index in range(n)]
    fallback = _partition(tuple(range(n)), singletons, rows, np.zeros(n), optimal=False, source="singleton_fallback")
    output = {
        "schema": SCHEMA, "selected_views": views,
        "missing_views": [view for view in views if not any(row["view_id"] == view for row in rows)],
        "status": "fallback", "primary": fallback, "hypotheses": [], "unproven_incumbent": None,
        "alternative_search_status": "not_started", "fallback_reason": None,
        "config": {"distance_threshold": DISTANCE_THRESHOLD, "feature_dimension": FEATURE_DIMENSION,
                   "objective": "sum_compatible_pair_margins", "max_hypotheses": max_hypotheses,
                   "time_limit_seconds": float(time_limit_seconds), "max_tracks": max_tracks,
                   "max_observations": MAX_OBSERVATIONS},
        "counts": {"observations": n, "candidate_tracks": 0, "compatible_pairs": 0},
        "solver_runs": [],
    }

    def finish(reason=None):
        output["fallback_reason"] = reason
        output["elapsed_seconds"] = float(time.monotonic() - started)
        return output

    if n > MAX_OBSERVATIONS:
        return finish("observation_limit")
    if n == 0:
        empty = _partition((), [], rows, [], optimal=True, source="joint_milp", rank=1, best=0.0)
        output.update(status="optimal", primary=empty, hypotheses=[empty], alternative_search_status="exhausted")
        return finish()
    vectors = np.stack([row["vector"] for row in rows])
    distances = np.clip(1.0 - vectors @ vectors.T, 0.0, 2.0)
    different_view = np.asarray([[left["view_id"] != right["view_id"] for right in rows] for left in rows])
    compatible = (distances <= DISTANCE_THRESHOLD) & different_view
    output["counts"]["compatible_pairs"] = int(np.triu(compatible, 1).sum())
    try:
        tracks = _enumerate_tracks(compatible, max_tracks, deadline)
    except _BudgetExceeded as error:
        return finish(str(error))
    output["counts"]["candidate_tracks"] = len(tracks)
    benefits = np.asarray([sum(DISTANCE_THRESHOLD - distances[left, right] for left, right in combinations(track, 2)) for track in tracks])
    row_indices, columns = [], []
    for column, track in enumerate(tracks):
        row_indices.extend(track)
        columns.extend([column] * len(track))
    incidence = csc_matrix((np.ones(len(columns)), (row_indices, columns)), shape=(n, len(tracks)))
    constraints = incidence
    lower, upper = np.ones(n), np.ones(n)
    seen = set()
    for rank in range(1, max_hypotheses + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            output["alternative_search_status"] = "time_limit" if output["hypotheses"] else "not_started"
            return finish(None if output["hypotheses"] else "solver_time_limit")
        try:
            solved = milp(
                c=-benefits, integrality=np.ones(len(tracks), dtype=np.uint8),
                bounds=Bounds(np.zeros(len(tracks)), np.ones(len(tracks))),
                constraints=LinearConstraint(constraints, lower, upper),
                options={"time_limit": remaining, "mip_rel_gap": 0.0, "presolve": True},
            )
        except RuntimeError as error:
            output["solver_runs"].append({"rank": rank, "status": "error", "error_type": type(error).__name__})
            output["alternative_search_status"] = "solver_error"
            return finish(None if output["hypotheses"] else "solver_error")
        status = int(solved.status)
        dual_bound = _finite(getattr(solved, "mip_dual_bound", None))
        output["solver_runs"].append({"rank": rank, "status": status,
                                      "mip_gap": _finite(getattr(solved, "mip_gap", None)),
                                      "objective_upper_bound": None if dual_bound is None else -dual_bound})
        indices = _solution_indices(getattr(solved, "x", None), incidence, len(tracks))
        if status != 0:
            if indices is not None:
                output["unproven_incumbent"] = _partition(indices, tracks, rows, benefits, optimal=False, source="unproven_solver_incumbent")
            output["alternative_search_status"] = {1: "time_limit", 2: "exhausted", 3: "solver_unbounded", 4: "solver_error"}.get(status, "solver_error")
            return finish(None if output["hypotheses"] else f"solver_status_{status}")
        if indices is None or indices in seen:
            raise ValueError("Optimal solver result is absent or repeats an excluded partition")
        seen.add(indices)
        best = output["hypotheses"][0]["objective"] if output["hypotheses"] else float(sum(benefits[index] for index in indices))
        hypothesis = _partition(indices, tracks, rows, benefits, optimal=True, source="joint_milp", rank=rank, best=best)
        output["hypotheses"].append(hypothesis)
        if rank == 1:
            output.update(status="optimal", primary=hypothesis)
        # With exact coverage, any different complete partition must omit a
        # selected track; this excludes exactly the current partition.
        cut = csc_matrix((np.ones(len(indices)), (np.zeros(len(indices), dtype=int), indices)), shape=(1, len(tracks)))
        constraints = vstack((constraints, cut), format="csc")
        lower = np.append(lower, -np.inf)
        upper = np.append(upper, len(indices) - 1)
    output["alternative_search_status"] = "limit_reached"
    return finish()


__all__ = ["SCHEMA", "DISTANCE_THRESHOLD", "FEATURE_DIMENSION", "match_scene"]
