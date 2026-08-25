# ToysAD-8K DS1 SAM 2.1 segmentation

The `kaggle-v2` folder contains the safe evidence from Notebook 05.1. The
notebook ran the pinned SAM 2.1 Hiera Large model on the 100-scene DS1 screen.

The run produced 7,700 masks from supplied reference boxes and 9,249 masks
from LocateAnything boxes. It completed all 2,000 images without an inference
failure. The 20-image repeat covered 460 prompts and reproduced every mask hash
and confidence score.

This notebook checks mask generation and repeatability. Mask quality is
measured against protected references in Notebook 06.1.

ClearML task `db0e28ec737f447c9ea0f5b7fda232d5` stores the aggregate tables,
figures, runtime and run identity.

Encoded masks, prompt boxes, scene identifiers and private overlay panels are
not stored in this repository.
