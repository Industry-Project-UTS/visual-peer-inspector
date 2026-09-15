# Development peer-ranking result

## What was tested

The established vision pipeline was not changed. LocateAnything supplied object proposals, SAM 2.1 supplied object masks, and DINOv2-small supplied visual features. The experiment reused the sealed outputs from the existing 100-scene development screen.

Notebook 08L prepared label-free candidate and cross-view features. Notebook 08M opened the development labels only after the feature package was locked. It compared 14 scoring methods using five grouped folds. Scenes from the same source asset stayed in the same fold.

The comparison included the existing B0, P1, P2 and forward-patch scores, robust score fusion, Borda ranking, view voting, three small learning-to-rank models, and two winner-clustering methods. The new winner-consensus method groups the object that wins within each view and ranks the resulting cross-view groups. View support is the main signal. A bounded margin and feature-coherence term only breaks equal-support ties.

## Main result

The primary result uses the automatic proposal and mask route with views V00, V07 and V14.

| Method | Tie-aware top-1 | Tie-aware average precision | Tie-aware recall at 2 | Tie-aware MRR |
| --- | ---: | ---: | ---: | ---: |
| Winner consensus | 0.377 | 0.477 | 0.526 | 0.528 |
| Winner forward cluster | 0.303 | 0.424 | 0.506 | 0.480 |
| Forward-patch baseline | 0.205 | 0.321 | 0.334 | 0.347 |
| Current P1 | 0.210 | 0.312 | 0.294 | 0.339 |
| Current P2 | 0.195 | 0.307 | 0.278 | 0.333 |
| B0 | 0.193 | 0.306 | 0.288 | 0.331 |

The winner-consensus improvement over the forward-patch baseline was 0.172 for tie-aware top-1. The paired asset-group bootstrap interval was 0.096 to 0.248. All five grouped folds improved.

The result is weaker when every predefined three-view schedule is included. Winner consensus averaged 0.301 tie-aware top-1 across the five schedules, compared with 0.204 for the forward-patch baseline. The gap between the primary schedule and the schedule average shows that view choice and cross-view identity remain important.

## What the result does not prove

- This is a 100-scene development comparison, not the protected final result.
- The selected method has not been confirmed on a separate frozen development cohort.
- Confidence is not calibrated. It is only a review diagnostic.
- Twenty-nine scenes did not have a rankable anomaly candidate on the automatic route. These scenes remained in the denominator.
- A higher deterministic top-1 value is also reported for audit purposes, but it is not used for selection because equal scores can be ordered by candidate ID. The scientific result averages over every equal-score ordering.

## Qualitative evidence

The private gallery contains 12 automatically selected cases from distinct asset groups:

- four correct cases with the strongest cross-view support
- four uncertain cases
- four definite failures

Each case shows the selected box, SAM overlay, object on a black background and coarse anomaly evidence for three views. Blank panels explicitly state when the selected identity was not linked in that view.

## Decision

Winner consensus is selected for frozen development confirmation. It is not yet the final web-application scorer. LocateAnything, SAM 2.1 and DINOv2-small remain fixed for this comparison.

## Next experiment

Apply the exact locked winner-consensus formula to a separate development confirmation cohort. Do not change the split, prompt, mask route, encoder, feature schema, score formula or view schedules after the evaluator is opened.

The confirmation notebook should report the same tie-aware metrics, the automatic-route failure waterfall, all fixed 2, 3 and 5-view schedules, asset-group intervals, and the same qualitative gallery layout. It should issue one of three outcomes: confirmed, confirmed with review, or not confirmed.

Protected calibration and the internal lock remain unopened until the development confirmation is complete.
