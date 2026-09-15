# Physical-object mapping after the object-bank scorer

## Fixed question

Can a joint assignment of detections across uploaded views preserve the anomaly-ranking gain measured with supplied correspondence? This comparison changes only the association step. LocateAnything, SAM 2.1, DINOv2-small and the selected `object_bank_top2` scorer remain fixed.

The 08J pairwise association and 08K patch-matching alternatives did not jointly improve physical identity and ranking. The current comparison introduces a single globally consistent partition, rather than another pairwise threshold or descriptor search.

## Representation and objective

A node is one detection in one selected image. A track is a proposed physical object. Feasible tracks contain at most one node from each selected image and require every pair to pass the existing cosine-distance limit of 0.35. Singleton tracks preserve unmatched detections.

The solver chooses tracks that cover every node exactly once. It maximises:

`sum over chosen tracks T [sum over pairs i,j in T (0.35 - cosine_distance(i,j))]`

This objective rewards directly supported cross-view links and favours longer mutually compatible tracks. It is an appearance objective, not a probability of physical identity. Similar objects can remain indistinguishable even when the optimisation is exact. No anomaly scores, supplied identities, camera positions or absolute image coordinates enter the matcher.

Up to three best complete partitions are retained using exclusion constraints. The best proven partition drives the unchanged scorer. Alternative rankings and gaps are uncertainty diagnostics; they do not determine which map is selected, and are not averaged into a new scoring formula.

## Scope and evaluation

Use the same 100 development scenes and 27 asset groups. The nineteen existing schedules contain one, two, three or five views. Each call receives only the selected observations and features. The three-view primary schedule remains the selection setting; two and three views are the intended demonstration range, with five supported.

Four methods have identical scene denominators:

1. Existing all-candidate association with frozen bank scoring.
2. Joint partial assignment with the same scoring.
3. Previous winner consensus, reused exactly from the sealed results.
4. Predicted masks with supplied correspondence and frozen scoring, clearly labelled assisted.

Promotion requires a positive lower 95% paired asset-bootstrap top-1 gain over winner consensus at primary three views, no AP or Recall@2 regression, improved link F1 without extra fragmentation against existing association, and positive mean gain across the three-view schedules. Undefined identity metrics cannot silently remove scenes from this gate. A failed gate ends this matching/scoring search.

## Runtime and persistence

No database is required. JSON tracks refer to candidate IDs, NPZ contains cached patch arrays, and CSV/JSON contains metrics. Scientific predictions, evaluation, rendering and ClearML publication are separately sealed. The matching budget is five seconds per setting with finite enumeration bounds; HiGHS limits are cooperative, not an operating-system hard timeout. An unproved primary produces an explicit singleton fallback, never a silently accepted optimality claim.

The source worktree is separate from `main`. Private configuration points to the retained 08N/08O cache in the working project. Clean and executed notebooks are separate; aggregate publication excludes raw images and private scene records.

## Research basis

[Multi-way matching by permutation synchronization](https://proceedings.neurips.cc/paper/2013/hash/3df1d4b96d8976ff5986393e8767f5b2-Abstract.html) motivates consistent maps across multiple sets. The small set-partition objective above is our adaptation, not an implementation or reproduced result from that paper. [SciPy MILP documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html) specifies the solver's objective, constraints, time limits and termination statuses.

## Application boundary

Dataset-reference correspondence and human-confirmed correspondence have different provenance. The existing app supports identity editing but still uses older P1 scoring. Its identity-only correction path also repeats neural work. Integration must add the frozen scorer, retain compatible features for CPU-only reranking, and represent tied rankings as no unique selection. Neither the assisted development accuracy nor an optimal appearance assignment proves reliable automatic highlighting on arbitrary uploads.
