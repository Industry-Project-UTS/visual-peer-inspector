# Week 6 progress: DS1 detection and segmentation

**Status:** Complete. SAM 2.1 is supported with reliable boxes. LocateAnything
needs refinement or replacement before the automatic route can be promoted.

## Work completed

- verified the label-free 100-scene DS1 package with all 20 views
- ran LocateAnything-3B on all 2,000 images with bounded recovery and resume
- ran SAM 2.1 for supplied boxes and LocateAnything boxes
- evaluated proposal and mask quality inside the protected evaluator
- calculated clustered bootstrap intervals and supported subgroup checks
- reviewed representative, worst and outlier mask panels privately
- stored the approved aggregate evidence in ClearML and GitHub-safe reports

## Result

The reference-box SAM route reached 0.99234 recall at mask IoU 0.75 and
0.95267 detection-aware mean mask IoU. Both main gates passed.

LocateAnything reached 0.78668 proposal precision and 0.94494 proposal recall
at box IoU 0.50. The automatic-box SAM route reached 0.92481 mask recall and
0.89388 detection-aware mean mask IoU. Proposal precision, proposal recall and
automatic-route mask IoU missed their declared gates.

The decision is to keep SAM 2.1 and refine or replace LocateAnything. The
supplied-mask and reference-box routes remain valid for the DS1 representation
screen. The automatic route must stay experimental.

## Next step

Run the DS1 representation and encoder screen on the supplied-mask and
reference-box routes. Keep the automatic route as a diagnostic comparison.
In parallel, test a bounded proposal improvement such as filtering, count
control, prompt refinement or a replacement detector before another automatic
route acceptance test.
