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
| Week 7 | [DS1 proposal refinement](week-07-proposal-refinement.md) | Duplicate proposals reduced; missed objects remained |
| Week 7 | [DS1 representation screen](week-07-ds1-representation-screen.md) | DINOv2-small and P1 selected for confirmation |
| Week 8 | [DS2 confirmation](week-08-ds2-confirmation-preparation.md) | Frozen 300-scene model run and protected multi-view evaluation completed |

## Current position

Notebooks 01 to 08C are complete as development evidence. The DS1 screen
selected DINOv2-small with P1 for confirmation, with B0 retained as the
baseline. The bounded LocateAnything prompt comparison was verified on Kaggle
Version 4. The 300-scene DS2 release, frozen model run, and protected
multi-view evaluation have now completed.

On DS2, LocateAnything reached 0.9383 recall and 0.9350 precision at box IoU
0.50. It achieved complete object detection in 87.9 percent of images. SAM 2.1
reached 0.9478 mean mask IoU with reference boxes and 0.8834 with automatic
boxes. P1 improved the supplied-mask 20-view scene AP over B0 by 0.0505. The
declared three-view automatic route reached 0.6531 scene AP.

The automatic deployment route remains a candidate, not a perfect detector.
The assisted route keeps user box confirmation before SAM 2.1. The next work
is calibration of review behaviour, label-free identity matching, and one
locked evaluation after the settings are saved.
