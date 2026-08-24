# ToysAD-8K DS0 SAM 2.1 evidence

The `kaggle-v5-candidate` folder contains safe aggregate evidence from the
completed Notebook 05 run.

The run covered 20 scenes, 20 views per scene and 400 images. It produced
3,256 masks from 1,640 supplied-box prompts and 1,616 automatic-box prompts.
No image failed and no empty mask was produced. The repeated 20-image check
matched all 169 mask hashes and quality scores.

ClearML task `ab86003feefa423fa42192ba57895286` stores the approved aggregate
tables, charts, metrics and run records. The six-view RGB and mask comparison
panel is kept outside GitHub and ClearML for private review and report use.

This evidence is a candidate because the recorded source snapshot was dirty.
Notebook 06 provides the ground-truth mask evaluation.
