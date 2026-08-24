# ToysAD-8K DS0 proposal and mask evaluation

The `kaggle-v6-candidate` folder contains safe aggregate evidence from the
completed Notebook 06 run.

The evaluator covered 20 scenes, 20 views per scene, 1,640 supplied masks,
1,640 supplied-box SAM masks and 1,616 automatic-box SAM masks. All eight
scientific gates passed.

The automatic route reached 0.9396 recall at mask IoU 0.75 and 0.9485 mean
matched mask IoU. The supplied-box route reached 0.9963 recall and 0.9591 mean
matched mask IoU.

ClearML task `ffe3eb049191483da876e3d1305a87cf` stores the seven aggregate
tables, scalar series and nine charts. Representative and worst-case RGB and
mask panels remain outside GitHub and ClearML for private review and report use.

This result supports the DS1 screen. It does not claim final route acceptance.
The run is a candidate because its recorded source snapshot was dirty.
