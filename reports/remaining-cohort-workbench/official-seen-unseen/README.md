# Official seen and unseen fixed-method results

10B evaluated the frozen multi-view methods on separate official roles. These are **exposure-qualified results**, not independent confirmation: the access audit records incomplete prior history and potentially exposed assets. Assisted uses supplied dataset correspondence; automatic uses predicted correspondence. One view is diagnostic, while 2, 3 and 5 views are application settings.

Tie-aware top-1 for each predefined primary view schedule (95% asset-grouped intervals):

| Role | Route | Views | Top-1 | Interval | Scenes |
| --- | --- | ---: | ---: | ---: | ---: |
| seen | automatic | 2 | 21.43% | 18.97–24.16% | 715 |
| seen | automatic | 3 | 13.08% | 10.64–15.76% | 715 |
| seen | automatic | 5 | 11.39% | 9.11–13.86% | 715 |
| seen | assisted | 2 | 24.32% | 21.18–27.36% | 715 |
| seen | assisted | 3 | 22.13% | 18.85–25.77% | 715 |
| seen | assisted | 5 | 22.93% | 19.94–26.14% | 715 |
| unseen | automatic | 2 | 19.14% | 17.58–20.74% | 2437 |
| unseen | automatic | 3 | 13.49% | 12.24–14.71% | 2437 |
| unseen | automatic | 5 | 9.26% | 8.21–10.27% | 2437 |
| unseen | assisted | 2 | 24.86% | 23.15–26.50% | 2437 |
| unseen | assisted | 3 | 23.59% | 21.70–25.41% | 2437 |
| unseen | assisted | 5 | 23.35% | 21.80–25.00% | 2437 |

The 60% target was not achieved on these official roles. Accuracy decreases with more views for the automatic route. Keep the original scene denominators and distinguish these numbers from development and internal-lock results. The complete schedule, Recall@2, average precision, reciprocal rank, candidate coverage, identity, and failure summaries are in `tables/`; nine aggregate figures are in `figures/`.

The selected qualitative examples were the first scene of each role and were chosen without checking outcomes. Their raw images, predicted boxes, masks, per-view scores and galleries are retained privately in the working directory. Object boxes are visual evidence and are not validated defect boundaries.

The Kaggle 10B job completed evaluation and report generation with no notebook error. The downloaded aggregate files match the scientific seal; all 85 report files match the report seal. The 6,305 private per-scene scientific files were not downloaded locally and remain on Kaggle. A separate publication job uploaded 16 sealed aggregate artifacts to [ClearML](https://app.clear.ml/projects/62a09a229f184cb7bbe50da59b4226e7/experiments/99f06937b3054fe5a2a3658b4a09e8dd/output/log) without rerunning the scientific computation; `clearml_publication_receipt.json` records the verified artifacts and lock hashes.
