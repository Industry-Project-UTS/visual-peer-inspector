# Week 5 progress: DS0 proposal and mask evaluation

## Work completed

- evaluated the frozen LocateAnything proposals against supplied geometry
- compared supplied masks with supplied-box and automatic-box SAM routes
- measured mask IoU, Dice, boundary F1, missed objects and extra predictions
- used 10,000 paired bootstrap repetitions grouped by base asset
- added strong, typical and challenging mask examples for private review
- created a label-free mask-route package for the ranking stage
- stored approved aggregate metrics, tables and charts in ClearML

## Result

All eight scientific gates passed. The supplied-box route reached 0.9963
recall at mask IoU 0.75 and 0.9591 mean matched mask IoU. The automatic route
reached 0.9396 recall and 0.9485 mean matched mask IoU.

The result supports the 100-scene DS1 screen. It does not approve the automatic
route as the final web application default because wider evaluation and
downstream ranking checks are still pending.

## Next step

Notebook 07 will calculate B0, B1 and P1 with the supplied-mask,
supplied-box and automatic-box routes under one frozen scoring setup.
