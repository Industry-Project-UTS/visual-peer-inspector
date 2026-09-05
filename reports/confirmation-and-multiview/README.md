# DS2 confirmation evidence

This folder contains aggregate evidence from the frozen 300-scene DS2
development confirmation. The RGB images, masks, per-image predictions,
evaluator labels, and private example panels stay outside GitHub.

## Run sequence

1. Notebook 04.3 froze the LocateAnything prompt and post-processing policy on
   DS1.
2. Notebook 08A released the separated DS2 inputs.
3. Notebook 08B ran LocateAnything, both SAM 2.1 routes, and DINOv2-small with
   B0 and P1.
4. Notebook 08C opened evaluator data after prediction lock and measured the
   frozen routes, view counts, fusion rules, subgroups, and failures.

## Main findings

- LocateAnything recall was 0.9383 at box IoU 0.50. Complete object detection
  was achieved in 87.9 percent of images.
- SAM 2.1 mean mask IoU was 0.9478 with reference boxes and 0.8834 with
  automatic boxes.
- P1 improved supplied-mask 20-view scene average precision over B0 by 0.0505.
  The paired 95 percent interval was 0.0249 to 0.0777.
- The declared automatic three-view route reached 0.6531 scene average
  precision with mean-rank fusion.

The working pipeline remains LocateAnything, SAM 2.1 Hiera-large,
DINOv2-small, and P1. The automatic route keeps a user-confirmed box fallback.
These results do not replace calibration or the internal-lock evaluation.
