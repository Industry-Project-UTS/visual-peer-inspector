# Result and next step

## Decision

The cross-view patch-bank scorer passed the predefined development gate. Keep `object_bank_top2` fixed for the next correspondence check. This is a confirmed **development candidate with supplied correspondence**, not a confirmed automatic application method.

The selected rule compares foreground target patches with every other physical object's selected-view patch bank, takes the lower median across peers and the upper 2% patch tail, normalises scores to average per-view ranks, then averages the two strongest available observations. Every observation of the target identity is excluded from its references. Unknown identities remain explicit and unscored by the bank method.

## Fixed selected method: primary schedules

All values are tie-aware scene top-1 on the same 100 development scenes and 27 underlying asset groups.

| Region and correspondence route | 2 views | 3 views | 5 views |
|---|---:|---:|---:|
| Supplied masks + supplied correspondence | 57.6% | **64.3%** | 69.5% |
| Predicted masks + supplied correspondence | 57.7% | **62.8%** | 69.0% |
| Predicted masks + existing predicted correspondence | 25.3% | 20.4% | 18.7% |
| Previous automatic winner consensus | 39.0% | 37.7% | 18.9% |

Do not describe either assisted row as ordinary RGB-only upload accuracy. In the predicted-mask assisted route, unknown candidates remain in the results but cannot enter trusted peer banks.

## Uncertainty and schedule sensitivity

For the primary three-view supplied-mask route, the selected scorer improved over regenerated forward mean by **13.1 percentage points**, with an asset-grouped paired 95% interval of **+4.7 to +21.6 points**. Its absolute top-1 interval is **53.4–75.2%**.

With predicted masks and supplied correspondence, the gain is **10.0 points**, with a paired interval of **+1.7 to +18.1 points**. Its absolute top-1 interval is **52.2–73.1%**.

Across all five predefined three-view schedules, the selected scorer averages **65.1%** with supplied masks and **62.5%** with predicted masks, still with supplied correspondence. The existing automatic winner consensus averages **30.1%**. Each scene is averaged across schedules before asset-group resampling; repeated views are not additional independent test scenes.

Both mean and strongest-two patch-bank fusion passed the scoring gate. The registered three-view criterion selected strongest-two fusion. The peer-view-mean bank-size control was reported as a diagnostic and was not eligible for selection. Do not combine whichever fusion wins at each view count into an untested headline configuration.

## Checks and boundaries

- Reproduced the three historical anchor values exactly: 37.7% automatic winner consensus, 51.2% supplied-mask forward at three views and 61.0% supplied-mask forward at five views.
- Reproduced the winner failure decomposition: all 100 scenes contain an anomaly candidate somewhere in the primary three views; 19 lose it through winner pruning; 10 through mixed association; 71 remain rankable.
- The cohort contains 125 anomalous and 260 normal objects. Fifteen scenes do not have a strict normal majority. Anomaly-count and majority-status tables preserve those scenes rather than removing them.
- These data were already used in development. Neither this cohort nor the remaining development data should be called globally untouched. Audit prior asset exposure before selecting any confirmation cohort.
- The result supports a 60% development point estimate under supplied-correspondence conditions. Its uncertainty interval does not establish a population accuracy of at least 60%.

## Execution and storage

Everything ran on the local CPU. Missing DINO features took about 371 seconds to prepare; model-facing ranking took about 52 seconds; evaluation took about 8 seconds. No LocateAnything or SAM inference was repeated. The 4,000 view/region checkpoints retain 15,294 candidate observations across the nineteen schedules' input union. The neural feature cache occupies about 5.62 GB. Peak process memory was not instrumented.

Both clean and executed notebooks are saved, along with the exact protocol, manifests, scene checkpoints, original-image transforms, 47,500 scene/method/schedule metric rows, aggregate tables, private galleries and failure receipts. Local file-backed records are sufficient; no database or Kaggle run was needed.

The final suite passed 124 tests. It includes a real local Jupyter kernel, candidate/identity/tie cases, interruptions and exact resumes, corruption and provenance checks, crop/heatmap coordinates, and publication failures. The later selected-method notebook gallery addition passed the 18 non-kernel publication/notebook checks and was exercised through the executed notebook.

## ClearML

Eleven approved aggregate artifacts were uploaded and the remote artifact inventory verified. Private galleries, scene metrics, raw inputs, model weights and patch caches were not uploaded.

[Verified ClearML task](https://app.clear.ml/projects/62a09a229f184cb7bbe50da59b4226e7/experiments/acd73b155c49417bbb00e6d17b324a6e/output/log)

## Next bounded step

1. Freeze the selected scorer and its extraction/normalisation rules.
2. Compare the existing all-candidate association against one global, partial-assignment approach retaining a small number of plausible maps. Matching evidence must be label-free and confined to uploaded views.
3. Measure whether that mapping preserves the assisted scoring gain, including identity fragmentation, mixed tracks, unmatched observations and uncertainty in highlighting.
4. If mapping does not help, retain the stronger current automatic baseline and an explicitly correspondence-confirmed mode. Do not start another scorer or backbone search.
5. Reconcile evaluation exposure and the assisted-input claim before confirmation and app integration.

The working result is now an object-level multi-view anomaly decision with useful evidence under reliable correspondence. Automatic mapping is the unresolved step.
