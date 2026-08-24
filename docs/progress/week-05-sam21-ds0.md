# Week 5 progress: SAM 2.1 DS0 segmentation

## Work completed

- prepared model-safe prompts for the supplied-box and automatic-box routes
- ran the pinned SAM 2.1 model on all 400 DS0 images
- produced 1,640 supplied-box masks and 1,616 automatic-box masks
- repeated 20 images and confirmed identical mask hashes and quality scores
- recorded runtime, mask count, mask area, predicted score and failure evidence
- added six direct visual comparisons across both prompt routes
- stored the approved aggregate evidence in ClearML

## Result

All 400 images completed and all operational gates passed. The run produced
3,256 masks with no failed images and no empty masks. The median processing
time was 0.2482 seconds per image and the p95 time was 0.2625 seconds.

This result confirms that the SAM stage is complete and repeatable on DS0. It
does not claim that every mask is accurate.

## Next step

Notebook 06 compares both SAM routes with the supplied masks. It measures mask
IoU, Dice, boundary quality, missed objects and extra predictions.
