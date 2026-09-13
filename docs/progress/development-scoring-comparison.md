# Development scoring comparison

**Status:** Complete development evidence. No protected calibration or
internal-lock labels were opened.

## Why this comparison was needed

Notebook 08F showed that the application could calculate DINOv2-small features,
but its label-free cross-view object links were weak. Notebooks 08G and 08H
checked whether a different peer-scoring rule could improve the result without
changing the encoder, the fixed 2,000-scene protocol, LocateAnything, or SAM.

The comparison used the existing 100-scene development screen. This is a
nested development subset, not a new split. All methods used the same cached
features and the same view selections so that the comparison was fair.

## Work completed

- reused the fixed DINOv2-small revision and existing object regions
- processed 100 scenes and 2,000 RGB views across supplied and automatic routes
- compared B0, B1, P1, two directional patch scorers, mutual missing evidence,
  geometry, P2 consensus, and an asset-grouped out-of-fold context ranker
- evaluated two, three, and five view application settings
- kept 20-view results as a reference-identity capability diagnostic only
- calculated scene average precision, top-1 accuracy, recall at two, AUROC,
  tie behaviour, winner support, and winner purity
- used asset-grouped bootstrap intervals for paired method differences
- saved aggregate tables, one comparison graph, 12 label-free previews, and 24
  evaluator-only success and failure galleries
- verified both completed Kaggle runs and their ClearML artifact lists

## Main result

The primary application setting is the automatic mask route with predicted
identity and three views. The best average precision was 0.34165 from
`forward_10_cov0`. The best top-1 accuracy was 0.24, shared by
`forward_02_cov70` and the out-of-fold context ranker. P1 reached 0.31998
average precision and 0.23 top-1 accuracy.

The scorer-only capability was much stronger when identity was reliable. On
the supplied-mask, reference-identity diagnostic, top-1 accuracy reached 0.66.
On automatic masks with reference identity, the best three-view method reached
0.70160 average precision and 0.51 top-1 accuracy.

This gap is the important finding. DINOv2-small features and the candidate
scorers can separate unusual objects when the same physical object is linked
correctly across views. The automatic route loses most of that performance
when predicted identity is used.

## Decision

Keep DINOv2-small. Keep P1 as the frozen comparator. Retain
`forward_02_cov70`, `forward_10_cov0`, and the out-of-fold context ranker as
development candidates, but do not replace P1 from this used development
screen alone.

The next experiment should improve label-free cross-view identity and then
re-evaluate the frozen scorer set on the same development boundary. The
calibration and internal-lock roles must remain closed until that decision is
made. A new split and a DINO version change are not needed.

## Run evidence

- Notebook 08G Kaggle Version 2 completed the 100-scene feature and prediction
  run. ClearML task: `1bcbb232e3f342d8b9fa7b78c146bc01`.
- Notebook 08H Kaggle Version 2 completed the evaluation. ClearML task:
  `d8b38d6277b04afabb24c46710eb7060`.
- All 55 focused scoring, recovery, mount, and publication tests passed.
- The clean notebook sources have no saved execution output. Repository-safe
  executed copies remove private scene visuals and protected identifiers.

## Limits

These results are development evidence. They are not untouched confirmation
and do not support a final production accuracy claim. The optional human
correction route remains useful when box or identity evidence is incomplete,
but it is not counted as automatic-route performance.
