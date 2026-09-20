# Protected 2,000-scene multi-view evaluation

The frozen multi-view method completed development, calibration and one-shot internal-lock evaluation without changing the vision models or ranking methods between protected stages.

## Completed cohorts

| Cohort | Scenes | Role | Five-view automatic | Five-view assisted |
| --- | ---: | --- | ---: | ---: |
| Development | 1,000 | Method diagnosis and fixed-method measurement | 22.24% | 62.42% |
| Calibration | 600 | Review-policy fitting | 21.32% | 57.23% |
| Internal lock | 400 | One-shot protected evaluation | 22.00% | 61.72% |

All values are tie-aware top-1 accuracy. The primary internal-lock assisted five-view schedule reached 62.77%. The assisted route depends on supplied or user-confirmed object correspondence. The automatic route remains substantially weaker and requires review.

## Decision

The scientific 09-series exit gate is complete. The web application should expose both routes, keep per-view scores and object links visible, and highlight the selected physical object in every usable matched view. Automatic results remain review-required. Assisted results may be marked ready only under the frozen calibration policy.

The official seen/unseen 10-series remains unopened. ClearML publication of the sealed 09B and 09D aggregate allowlists is operational follow-up and must not rerun evaluation.

## Evidence

- `reports/remaining-cohort-workbench/development1000/`
- `reports/remaining-cohort-workbench/calibration600/`
- `reports/remaining-cohort-workbench/internal-lock400/`
- `reports/remaining-cohort-workbench/protected2000/`
- `notebooks/executed/protected-evaluation/`
