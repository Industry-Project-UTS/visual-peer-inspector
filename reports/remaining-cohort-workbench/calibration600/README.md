# 600-scene calibration evaluation

This directory contains the aggregate evidence from the frozen-method calibration evaluation. The cohort contains 600 scenes in 163 asset groups. Automatic predicted correspondence and assisted dataset correspondence are reported separately.

> Claim boundary: this is an exposure-qualified calibration result. It is not independent confirmation.

## Headline result

| Route | Views | Tie-aware top-1 | Recall@2 | AP | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Automatic predicted correspondence | 1 | 37.65% | — | — | — |
| Automatic predicted correspondence | 2 | 30.31% | — | — | — |
| Automatic predicted correspondence | 3 | 24.70% | — | — | — |
| Automatic predicted correspondence | 5 | 21.32% | 36.27% | 37.25% | 39.45% |
| Assisted dataset correspondence | 1 | 38.40% | — | — | — |
| Assisted dataset correspondence | 2 | 47.06% | — | — | — |
| Assisted dataset correspondence | 3 | 53.79% | — | — | — |
| Assisted dataset correspondence | 5 | 57.23% | 76.56% | 73.62% | 74.55% |

The assisted route improved as views were added, while the automatic route degraded. The complete-cohort assisted result did not reach 60% on calibration.

## Frozen review policy

- Automatic results require review whenever a ranking is available; an empty ranking is insufficient evidence.
- Assisted results are ready only when the winner is unique, linked in every uploaded view, and reaches the frozen confidence threshold.
- The thresholds are 0.40 for two views and 0.25 for three or five views.
- Four-view uploads remain review-required because four views were not evaluated as a frozen contract.

Selective assisted accuracy reached 67.35% at 8.17% coverage for two views, 75.48% at 14.00% coverage for three views, and 96.47% at 3.54% coverage for five views. These calibration-fitted selective values do not replace complete-cohort accuracy.

## Files

- `tables/`: complete aggregate metric, schedule, failure, identity and subgroup summaries.
- `figures/`: nine aggregate report figures.
- `review-policy/`: the frozen mode-specific policy, metrics and receipt.
- `qualitative/`: six unique examples with boxes, masks, combined rankings and per-view scores; the duplicate seventh source entry has a deduplication receipt.
- Run bindings, stage/report receipts and local verification records provide provenance.

Complete per-scene metrics, evaluator truth, caches, tensors, raw masks, model assets and credentials remain in the verified local evidence store.
