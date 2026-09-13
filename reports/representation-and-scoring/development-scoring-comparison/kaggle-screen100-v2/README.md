# Development scoring comparison, Kaggle Versions 2

This folder contains GitHub-safe aggregate evidence from Notebooks 08G and
08H. The runs reused the existing 100-scene development screen with 2,000 RGB
views. LocateAnything and SAM were not rerun.

Notebook 08G generated the DINOv2-small feature and scorer records for the
supplied and automatic mask routes. Notebook 08H compared nine methods across
two, three, and five views, with 20-view reference-identity results retained as
a capability diagnostic.

## Primary automatic result

| Method | Scene AP | Top-1 | Recall at 2 | AUROC |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.32320 | 0.23 | 0.295 | 0.30945 |
| P1 | 0.31998 | 0.23 | 0.295 | 0.30546 |
| Forward 2%, 70% coverage | 0.33984 | 0.24 | 0.355 | 0.33557 |
| Forward 10%, no coverage filter | 0.34165 | 0.23 | 0.325 | 0.34731 |
| Out-of-fold context ranker | 0.33817 | 0.24 | 0.335 | 0.33483 |

The primary setting is automatic masks, predicted identity, and three views.
No method passed an automatic promotion rule. The paired average-precision gain
of the context ranker over P1 was positive, but its top-1 interval included no
change. The directional method gains were also uncertain after grouping by
asset.

## Interpretation

Reference-identity diagnostics were much stronger. The best automatic-mask,
reference-identity three-view result reached 0.70160 scene average precision
and 0.51 top-1 accuracy. With supplied masks and reference identity, top-1
accuracy reached 0.66. The main limitation is therefore cross-view identity,
not a lack of DINO feature vectors.

P1 remains the frozen comparator. The two directional methods and the
out-of-fold context ranker remain candidates for the next identity-aware
development comparison. No calibration or internal-lock result is included.

## Contents

- `08g/` contains runtime evidence and the ClearML receipt.
- `08h/` contains aggregate rankings, subgroup tables, paired intervals, the
  route and view comparison chart, and the ClearML receipt.
- `execution/` contains repository-safe saved-output notebooks, sanitized logs,
  and publication receipts.

Private scene galleries, per-scene metrics, raw Kaggle logs, feature caches,
prediction shards, RGB inputs, masks, evaluator labels, and credentials are
excluded from this repository folder.
