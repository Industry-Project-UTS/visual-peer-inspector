# Multi-view object-bank evaluation

## Scope and decisions

Work remains in this development folder. The submission checkout is read-only. No commits, pushes or PRs are part of this stage.

No database is required. JSON object records hold identities and observations; NumPy NPZ files retain lossless patch features; CSV/JSON provide evaluation results. Hash manifests and atomic checkpoints bind them to the exact inputs, model, code and configuration. A future shared web service may need a job database, but that is independent of this experiment.

## Notebook 08N: Object registry and reusable patch preparation

1. Verify the exact existing model/scoring packages and extract neutral correspondence from already label-free sources.
2. Use the fixed DINOv2-small revision and existing masks/crop policy. Retain the missing raw patch tensors with per-view checkpoints and original-image transforms.
3. Run a real local runtime pilot, then choose local extraction if practical or Kaggle for GPU work. No LocateAnything/SAM reruns.
4. Seal model-facing features, registries and deterministic predictions before opening anomaly labels.

The decision inputs contain at most five views. Existing predefined schedules use different subsets of the available twenty cached views; extracting their union does not permit any setting to access excluded images.

## Notebook 08O: Bounded scoring evaluation

Compare the unchanged cached forward anchor, a forward baseline regenerated from the same patches, cross-view peer banks, and one bank-size control. Use mean and strongest-two rank fusion; do not search weights. Supplied-mask/supplied-ID, predicted-mask/supplied-ID and existing predicted-ID routes remain separate. Report each schedule and scene; missing/unrankable cases remain in the denominator.

The bank-size control averages nearest-patch distances across a peer's views before reducing across physical peers. It controls the extra minimum over views but may penalize different visible surfaces. It is a diagnostic, not a claim of improved performance.

Report asset-group intervals, candidate/identity/ranking losses, anomaly-count and normal-majority breakdowns. Preserve success, uncertainty and failure examples with boxes, masks, black-background objects, per-view scores and final ranks. Heatmaps are coarse patch evidence, not validated defect boundaries.

## Decision and stopping rule

The protocol is fixed before full evaluation. Use the declared supplied-reference scoring gate before any new automatic association development. A failed gate ends this direction; it does not trigger another scorer search. A positive result permits one bounded mapping check and later freeze/app integration. This stage does not reopen protected evaluation or claim an untouched cohort.

## Persistence and publication

- Compute first; preserve verified output independently of plotting/tracking.
- Resume exact checkpoints only. Corruption or changed bindings stop the run.
- ClearML publication is aggregate-only and time bounded. A failure yields a retryable receipt without rerunning computation.
- Preserve raw executed notebooks privately; use the existing sanitiser for repository-review copies.
- Prepare repository-files/ and review-only/ only after evidence is verified. Reused files already on main are not new changes.
