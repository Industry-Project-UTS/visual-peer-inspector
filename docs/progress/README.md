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
| Development freeze | [Full development results](development-freeze-completion.md) | 1,000 scenes and 20,000 views locked; cross-view comparison complete, with low identity accuracy |
| Development scoring | [Scoring comparison](development-scoring-comparison.md) | Fixed DINOv2-small features reused to compare nine scorers; identity association remains the main limitation |
| Identity refinement | [Foreground patch comparison](development-identity-refinement.md) | Same 100-scene screen reused; top-1 improved but identity-link F1 fell, so no method was promoted |
| Identity association | [Association study](development-identity-association.md) | Reciprocal and cycle-pruned alternatives were rejected because ranking gains came with weak or fragmented identity tracks |
| Patch correspondence | [Patch correspondence study](development-patch-correspondence.md) | Foreground patch and Sinkhorn alternatives did not pass the joint promotion rule |
| Learned peer ranking | [Learned peer-view ranker](development-learned-peer-view-ranker.md) | Winner consensus improved tie-aware top-1 on the development screen and was selected for frozen confirmation |
| Candidate retention and fusion | [Multi-view object bank](development-multiview-object-bank.md) | 08N/08O retained candidates and selected `object_bank_top2` for assisted correspondence |
| Physical-object mapping | [Mapping and stopping decision](development-physical-object-mapping.md) | 08P rejected the joint automatic mapper and ended the method search |
| Protected evaluation | [2,000-scene multi-view evaluation](protected-2000-multiview-evaluation.md) | Development, calibration and one-shot internal lock complete through 09D |

## Current position

The 09-series exit gate is complete: 1,000 development, 600 calibration and 400 internal-lock scenes were evaluated with the frozen automatic and assisted methods. Internal-lock five-view tie-aware top-1 accuracy was 22.00% automatic and 61.72% assisted. The assisted route uses confirmed correspondence. The official 10-series remains pending.

Notebooks 01 to 09D are complete through the protected-evaluation exit gate. The historical development detail below records how the frozen methods were selected. The DS1 screen
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
User box correction remains an optional assisted feature; it does not replace
the automatic route's results. The corrected 08E.1 V4 reference release and
08F V1 comparison are complete. Embedding-only matching was retained, but
five-view pairwise identity accuracy is only 33.04%. This is not odd-one-out
accuracy. In 413 of 1,000 five-view results, the chosen identity appears in
one view only. The code and output contract are reproducible, but reliable
cross-view identity is not solved. Review these development limitations
before starting protected calibration. Any method change needs a new
development-only comparison and method lock.

Notebooks 08G and 08H reused the existing 100-scene development screen. They
did not change the 2,000-scene split and did not rerun LocateAnything or SAM.
The best automatic three-view scorer reached 0.34165 scene average precision
and 0.24 top-1 accuracy. On the supplied-mask, reference-identity diagnostic,
top-1 accuracy reached 0.66. The gap confirms that predicted cross-view
identity is the main next development problem. P1 remains the frozen comparator
until an identity-aware development rerun supports a replacement.

Notebook 08I completed that bounded development rerun with foreground patch
and hybrid descriptors. The main candidate improved automatic three-view
top-1 accuracy from 0.24 to 0.31, but identity-link F1 fell from 0.188789 to
0.139612 and the average-precision interval crossed zero. No method was
promoted. The current global descriptor remains the stable default while the
next development study focuses only on track association.

Notebooks 08J and 08K completed the bounded association studies. Reciprocal,
cycle-pruned, foreground-patch and Sinkhorn candidates did not improve identity
quality and ranking together, so the existing Hungarian association was
retained. Notebook 08M then compared fixed and learned peer-ranking rules using
the locked 08L features and asset-group folds. Winner consensus reached 0.377
tie-aware top-1 on the primary automatic three-view schedule, compared with
0.205 for the previous forward-patch baseline. Across all five declared
three-view schedules it averaged 0.301. This is a development selection for a
separate frozen confirmation, not a final accuracy claim.
