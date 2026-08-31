# Week 7 progress: DS1 representation screen

**Status:** Complete candidate evidence. DINOv2-small remains the encoder and
P1 is selected for the next development stage.

## Work completed

- processed all 100 DS1 scenes and all 2,000 views
- compared DINOv2-small, DINOv2-small with registers, DINOv2-base and a
  ResNet-50 control
- recorded both DINOv3 candidates as unavailable because gated model access
  was not available in Kaggle
- calculated B0, B1 and P1 ranking results
- compared supplied-mask, reference-box and automatic-box routes
- reported one-anomaly and two-anomaly scene results separately
- used confidence intervals grouped by the 27 base assets
- generated eight aggregate figures and report-ready tables
- verified the downloaded Kaggle output independently
- stored the safe metrics, tables and figures in ClearML

## Result

DINOv2-small reached a mean scene average precision of 0.64074 with B0 on the
supplied-mask route. DINOv2-small with registers reached 0.64658 and
DINOv2-base reached 0.64129. Neither improvement passed the fixed promotion
rule. The ResNet-50 control reached 0.62350.

With DINOv2-small, P1 improved mean scene average precision from 0.64074 to
0.66721. The paired 95 percent interval for this gain was 0.01208 to 0.04295.
B1 reached 0.63768 and was not promoted.

The automatic-box route remained weaker. Its P1 average precision was 0.63400,
compared with 0.65809 for reference boxes and 0.66721 for supplied masks. This
supports the earlier decision not to accept the current automatic proposal
route.

## Decision

Freeze DINOv2-small with P1 for the next development stage. Keep the supplied
mask route as the controlled anchor and the reference-box route as the practical
comparison. Keep automatic boxes as diagnostic evidence until proposal recall
improves.

## Next step

Use the frozen DINOv2-small and P1 configuration in Notebook 08. Evaluate view
subsets, cross-view identity association, score fusion, calibration and the
review rule without changing the encoder or scorer from the DS1 evidence.
