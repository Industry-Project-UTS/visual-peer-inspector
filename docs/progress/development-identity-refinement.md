# Development identity refinement

**Status:** Complete development evidence. No method was promoted.

## Why this experiment was run

Notebooks 08G and 08H showed that DINOv2-small and the scorer family perform
better when object identity is reliable. The automatic route was weaker
because it did not link the same physical object consistently across views.

Notebook 08I tested one bounded change: foreground patch descriptors for
cross-view identity. LocateAnything, SAM 2.1, the encoder revision, peer
scorers, view sets, and the original 2,000-scene protocol stayed fixed. The run
used the same nested 100-scene development screen as the earlier scoring
comparison.

## Work completed

- reused the sealed Notebook 08G global descriptors and score records
- calculated foreground patch descriptors only where they were missing
- compared global, local, and hybrid descriptors at three coverage levels
- tested unmatched costs of 0.20, 0.35, and 0.50
- evaluated two, three, and five views with the existing scorer set
- measured identity precision, recall, F1, and track purity together with
  scene average precision, top-1 accuracy, recall at two, and AUROC
- used asset-grouped paired intervals
- reviewed four improvements, four regressions, and four fixed examples in
  evaluator-only evidence
- saved aggregate tables and figures in GitHub and ClearML

## Result

In the main automatic three-view comparison, the foreground hybrid candidate
raised top-1 accuracy from 0.24 to 0.31. Scene average precision rose from
0.339835 to 0.375897, but its paired interval included no change. Identity-link
F1 fell from 0.188789 to 0.139612.

The candidate produced purer but less complete tracks. This can help it choose
the correct unusual object in some scenes, but it does not solve cross-view
identity. The private gallery also contains both improvements and regressions.

## Decision

Do not promote the foreground hybrid candidate. Keep it as development
evidence and retain the current global descriptor as the stable default.

The next bounded experiment should reuse the sealed 08I outputs and change
only track association. Mutual matching, cycle consistency, and support across
views are the next useful candidates. A method must improve identity quality
and odd-one-out ranking together before it replaces the current method.

The calibration and internal-lock roles remain closed. The original split and
model stack remain unchanged.
