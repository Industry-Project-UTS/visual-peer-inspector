# ToysAD-8K DS1 LocateAnything screen

The `kaggle-v4` folder contains the safe evidence from Notebook 04.1. The
notebook ran the pinned NVIDIA LocateAnything-3B model on all 2,000 images in
the 100-scene DS1 development screen.

All images completed. The final result contains 9,249 object proposals. The
run recorded eight recoverable truncation attempts and one successful slow-mode
fallback. Repeating the 20-image pilot gave identical boxes.

This notebook checks inference reliability and proposal generation. Proposal
accuracy is measured in Notebook 06.1.

ClearML task `04a27581efab440f9aa0ada2b2efc15e` stores the aggregate tables,
figures, runtime and run identity.

Raw model responses, bounding boxes, scene identifiers and private example
panels are not stored in this repository.
