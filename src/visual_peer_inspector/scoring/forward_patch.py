"""Directed patch scoring for the development-only comparison."""

import numpy as np

from visual_peer_inspector.scoring.peer import PeerScoreResult, _unit_rows


def forward_patch_scores(features, masks, *, top_fraction):
    """Take the lower median across peers, then the worst patch tail.

    Floor rounding and the lower median reproduce the proposed scorer's
    reduction order. This is different from the current symmetric P1 score.
    """
    if len(features) != len(masks) or not 0 < top_fraction <= 1:
        raise ValueError("Invalid patch inputs or top fraction")
    if len(features) < 3:
        return PeerScoreResult("forward", "insufficient_peers", np.full(len(features), np.nan))
    vectors = [_unit_rows(x) for x in features]
    valid = [np.asarray(x, dtype=bool) for x in masks]
    for x, m in zip(vectors, valid, strict=True):
        if m.ndim != 2 or m.size != len(x) or not m.any():
            raise ValueError("Every object needs a non-empty valid patch grid")
    scores, maps = [], []
    for i, target in enumerate(vectors):
        distances = [
            np.min(1 - target[valid[i].ravel()] @ peer[valid[j].ravel()].T, axis=1)
            for j, peer in enumerate(vectors) if i != j
        ]
        # NumPy's median averages the two middle values; the proposal uses the lower one.
        middle = (len(distances) - 1) // 2
        patch_distance = np.partition(np.stack(distances), middle, axis=0)[middle]
        tail_count = max(1, int(len(patch_distance) * top_fraction))
        scores.append(float(np.partition(patch_distance, len(patch_distance) - tail_count)[-tail_count:].mean()))
        heatmap = np.full(valid[i].shape, np.nan)
        heatmap[valid[i]] = patch_distance
        maps.append(heatmap)
    return PeerScoreResult("forward", "scored", np.asarray(scores), tuple(maps))


def patch_policy(crop, grid_size, coverage, *, fallback):
    """Select patches and record whether the declared fallback was needed."""
    valid = crop.patch_validity(grid_size, minimum_fraction=coverage)
    used_fallback = not valid.any()
    if used_fallback:
        if fallback == "all":
            valid[:] = True
        elif fallback == "largest":
            step = crop.foreground.shape[0] // grid_size
            amount = crop.foreground.reshape(grid_size, step, grid_size, step).mean(axis=(1, 3))
            valid.flat[int(np.argmax(amount))] = True
        else:
            raise ValueError("Unknown patch fallback")
    return valid, used_fallback
