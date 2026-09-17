# 1,000-scene development multi-view evaluation

This directory contains the repository-safe aggregate evidence from the fixed-method development evaluation. The evaluation includes 1,000 scenes in 273 asset groups. It reports the automatic and assisted correspondence routes separately.

> Claim boundary: this is an exposure-qualified development result. All 273 asset groups are potentially exposed, so it is not independent confirmation.

## Headline result

| Route | Views | Tie-aware top-1 | 95% asset-group interval | Recall@2 | AP | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Automatic predicted correspondence | 1 | 40.33% | 38.10% to 42.58% | 31.97% | 31.59% | 40.86% |
| Automatic predicted correspondence | 2 | 31.29% | 29.41% to 33.10% | 42.94% | 35.11% | 42.18% |
| Automatic predicted correspondence | 3 | 26.80% | 24.92% to 28.59% | 41.85% | 36.98% | 43.53% |
| Automatic predicted correspondence | 5 | 22.24% | 20.13% to 24.39% | 34.34% | 36.05% | 41.15% |
| Assisted dataset correspondence | 1 | 44.19% | 41.88% to 46.49% | 58.85% | 61.31% | 63.70% |
| Assisted dataset correspondence | 2 | 52.17% | 49.68% to 54.65% | 66.53% | 68.49% | 70.70% |
| Assisted dataset correspondence | 3 | 58.51% | 55.81% to 61.24% | 70.92% | 72.90% | 74.90% |
| Assisted dataset correspondence | 5 | 62.42% | 59.57% to 65.16% | 74.59% | 75.68% | 77.61% |

The assisted five-view route exceeded the 60% project target at 62.42% tie-aware top-1. It uses dataset-supplied object correspondence and must be described as assisted. The fully automatic route performed best at one view in this aggregate and degraded as views were added, which confirms that predicted association and fusion remain the main application limitation.

## Fixed methods

- Automatic: `winner_cluster_consensus` with predicted correspondence.
- Assisted: `object_bank_top2` with dataset-supplied correspondence.
- Main application settings: 2, 3, and 5 views; one view is a diagnostic.
- All scenes stay in the denominator, including missing and unrankable outcomes.

## Files

- `tables/metric_summary.csv`: primary metrics and asset-group intervals.
- `tables/view_schedule_summary.csv`: predefined schedule sensitivity.
- `tables/failure_summary.csv`: candidate coverage and loss decomposition.
- `tables/identity_summary.csv`: identity purity, fragmentation, and pair metrics.
- `tables/majority_summary.csv` and `tables/anomaly_count_summary.csv`: controlled subgroup analyses.
- `figures/`: nine aggregate, report-ready figures.
- `run_binding.json`, receipts, and local verification: provenance and integrity evidence.
- `tracking_receipt.json`: ClearML publication is pending with `MissingConfigError`; the scientific results are complete and sealed.

## Privacy boundary

Private scene metrics, evaluator truth, scene galleries, per-object scores, caches, tensors, masks, checkpoints, model assets, and credentials are intentionally excluded. They remain in the verified local evidence store.
