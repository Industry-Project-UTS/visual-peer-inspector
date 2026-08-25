# Progress record

These notes record completed work in chronological order. A dated note may
describe a next step that has since been completed. Use the current
[notebook sequence](../../notebooks/README.md) for the active plan.

| Period | Record | Outcome |
| --- | --- | --- |
| Week 3 | [Data and full-dataset EDA](week-03-data-and-eda.md) | Source structure, integrity, labels, and leakage risks checked |
| Week 4 | [Selection and tracking](week-04-selection-and-tracking.md) | Primary 2,000-scene experiment and role split prepared |
| Week 4 | [Opaque-package smoke test](week-04-opaque-package-smoke.md) | Model-facing DS0 package verified without protected labels |
| Week 4 | [LocateAnything DS0](week-04-locateanything-ds0.md) | 400-image proposal run completed |
| Week 5 | [SAM 2.1 DS0](week-05-sam21-ds0.md) | Reference-box and automatic-box masks completed |
| Week 5 | [Proposal and mask evaluation](week-05-proposal-mask-evaluation.md) | Eight evaluation gates passed; DS1 supported |
| Week 6 | [Representation and peer scores](week-06-representation-peer-scores.md) | B0, B1, P1 and bounded encoder screen completed on DS0 |
| Week 6 | [DS1 detection and segmentation](week-06-ds1-detection-and-segmentation.md) | Reference-box SAM passed; the automatic proposal route needs refinement |

## Current position

Notebooks 01 to 07 are complete on DS0. Notebooks 03.1 to 06.1 are also
complete on the 100-scene DS1 development screen using clean source snapshots.
The next experiment is the DS1 representation and encoder screen. Notebook 08
begins after the encoder and scorer are frozen.

The automatic deployment route is not frozen. The DS1 evaluator supports the
reference-box SAM 2.1 route but requires LocateAnything to be refined or
replaced. The supplied-mask and reference-box routes can continue into
representation testing while the automatic proposal route remains
experimental.
