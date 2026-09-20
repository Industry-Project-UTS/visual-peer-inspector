# Protected 2,000-scene multi-view evaluation

The fixed multi-view pipeline has completed its protected 2,000-scene programme: 1,000 development scenes, 600 calibration scenes and 400 internal-lock scenes. Cohorts remain separate because they have different experimental roles and exposure histories.

## Five-view results

| Cohort | Scenes | Asset groups | Automatic top-1 | Assisted top-1 |
| --- | ---: | ---: | ---: | ---: |
| Development | 1,000 | 273 | 22.24% | 62.42% |
| Calibration | 600 | 163 | 21.32% | 57.23% |
| Internal lock | 400 | 109 | 22.00% | 61.72% |

The internal-lock primary assisted five-view schedule reached 62.77%. The assisted route uses dataset-supplied physical-object correspondence and is not an automatic RGB-only result. The automatic route degrades as views are added, so the application must show supporting evidence and require review. The stronger assisted mode lets the user confirm object correspondence before producing the combined ranking.

## Frozen application contract

- Automatic method: `winner_cluster_consensus` with predicted correspondence.
- Assisted method: `object_bank_top2` with confirmed or dataset correspondence.
- Supported evaluated view counts: 2, 3 and 5; one view is diagnostic.
- Every scene remains in the denominator, including missing and unrankable outcomes.
- Calibration-fitted readiness thresholds are applied without retuning on internal-lock data.

## Evidence

- `../development1000/`: development aggregates and figures.
- `../calibration600/`: calibration aggregates, figures and frozen review policy.
- `../internal-lock400/`: one-shot internal-lock aggregates and figures.
- `../../../notebooks/executed/protected-evaluation/`: complete numbered 09-series milestone chain and evaluator-conversion notebooks.

These results are exposure-qualified because earlier experiment access may include related assets. They are suitable for the project report with that limitation. Official seen/unseen evaluation remains a separate 10-series gate and has not been run.
