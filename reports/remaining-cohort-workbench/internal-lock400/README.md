# 400-scene internal-lock multi-view evaluation

This directory contains repository-safe aggregate evidence from the one-shot 09D evaluation. It covers all 400 internal-lock scenes in 109 asset groups and reports the frozen automatic and assisted routes separately.

> Claim boundary: this is an exposure-qualified fixed-method evaluation. Prior asset access is incomplete and all 109 asset groups are potentially exposed, so this is not independent confirmation.

## Headline result

The table averages each metric across the predefined schedules for that view count. Confidence intervals use asset-group bootstrap resampling, while all 400 scenes remain in the denominator.

| Route | Views | Tie-aware top-1 | 95% asset-group interval | Recall@2 | AP | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Automatic predicted correspondence | 1 | 41.32% | 37.73% to 44.97% | 37.96% | 37.55% | 41.80% |
| Automatic predicted correspondence | 2 | 32.98% | 30.02% to 36.13% | 50.48% | 40.80% | 44.27% |
| Automatic predicted correspondence | 3 | 25.88% | 23.04% to 29.00% | 46.09% | 39.83% | 42.69% |
| Automatic predicted correspondence | 5 | 22.00% | 18.87% to 25.26% | 36.13% | 37.99% | 40.19% |
| Assisted dataset correspondence | 1 | 42.38% | 39.03% to 45.84% | 62.51% | 62.00% | 62.98% |
| Assisted dataset correspondence | 2 | 52.34% | 48.40% to 56.01% | 71.51% | 70.01% | 71.15% |
| Assisted dataset correspondence | 3 | 57.90% | 53.45% to 62.26% | 75.59% | 73.82% | 74.92% |
| Assisted dataset correspondence | 5 | 61.72% | 57.10% to 66.14% | 79.33% | 76.50% | 77.70% |

The assisted five-view route exceeds the 60% project target on this cohort. It uses dataset-supplied object correspondence and must be described as assisted. Its predefined primary five-view schedule reaches 62.77% tie-aware top-1. The fully automatic route degrades as views are added, which confirms that automatic cross-view association and fusion remain the main application limitation.

## Fixed methods

- Automatic: `winner_cluster_consensus` with predicted correspondence.
- Assisted: `object_bank_top2` with dataset-supplied correspondence.
- Core application settings: 2, 3 and 5 views; one view is diagnostic.
- No method, score, schedule or threshold was retuned after calibration.
- The 09D ranking evaluator did not fit a new readiness threshold.

## Files

- `tables/metric_summary.csv`: primary metrics and asset-group intervals.
- `tables/view_schedule_summary.csv`: predefined schedule sensitivity, including the primary five-view schedule.
- `tables/failure_summary.csv`: candidate coverage and loss decomposition.
- `tables/identity_summary.csv`: identity purity, fragmentation and pair metrics.
- `tables/majority_summary.csv` and `tables/anomaly_count_summary.csv`: controlled subgroup analyses.
- `figures/`: nine locked report figures.
- `qualitative/`: six unique examples with boxes, masks, combined rankings and per-view scores; the duplicate seventh source entry has a deduplication receipt.
- `evaluation_run_binding.json`, `report_run_binding.json`, receipts and `local_verification_summary.json`: provenance and integrity evidence.

## Privacy boundary

Complete per-scene metrics, evaluator truth, caches, tensors, raw masks, checkpoints, model assets and credentials remain in the local evidence store.
