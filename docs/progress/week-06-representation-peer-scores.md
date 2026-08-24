# Week 6 progress: representation and peer scores

**Status:** Complete DS0 candidate. Final encoder and scorer selection remains
open until DS1.

## Work completed

- prepared the same 224 by 224 masked object crops for every visual encoder
- compared DINOv2-small, DINOv2-small with registers and DINOv2-base
- retained ResNet50 as a non-DINO control
- calculated B0, B1 and P1 without exposing labels to the feature pipeline
- compared supplied-mask, reference-box and automatic-box routes
- reported one-anomaly and two-anomaly results separately
- evaluated separate shape and layout branches
- added paired confidence intervals grouped by base asset
- generated aggregate charts, crop checks, PCA plots and coarse P1 panels
- recorded runtime, GPU memory, repeatability, failures and completion gates
- stored approved aggregate evidence in ClearML

## Result

DINOv2-small with registers produced the highest B0 scene AP at 0.6709.
DINOv2-small produced 0.6560 with B0 and 0.6724 with P1. The paired intervals
for both improvements crossed zero, so DS0 does not support a final model or
scorer change.

The automatic-box route produced 0.6383 B0 scene AP. Eleven of its 400 views
had fewer than three peer proposals and were reported as insufficient rather
than forced into the ranking results.

DINOv3 could not be evaluated because gated Hugging Face access was not
available. DINOv2-base used more memory and did not improve B0.

## Next step

Run the supplied-mask encoder screen on the 100-scene DS1 cohort. Compare
DINOv2-small, DINOv2-small with registers and the licensed DINOv3 small
variants. Freeze the encoder and scorer only after the paired promotion gates
pass. Notebook 08 can then test view subsets, identity association, fusion and
calibration.

See the [notebook sequence](../../notebooks/README.md) for the full dependency
flow and acceptance rules.
