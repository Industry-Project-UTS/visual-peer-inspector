# ToysAD-8K DS0 representation and peer scores

The `kaggle-v5-candidate` folder contains safe aggregate evidence from the
completed Notebook 07 run. The run covered 20 scenes and all 400 RGB views.

The notebook compared DINOv2-small, DINOv2-small with registers, DINOv2-base
and a ResNet50 control. It also evaluated B0, B1 and P1, the three mask routes,
separate geometry scores, one-anomaly and two-anomaly scenes, runtime, memory
and rankable coverage.

DINOv2-small with registers reached 0.6709 scene AP with B0. DINOv2-small
reached 0.6560 with B0 and 0.6724 with P1. The paired confidence intervals
crossed zero, so these differences are not treated as final. DINOv2-small
remains the anchor until the 100-scene DS1 screen.

DINOv3 was recorded as unavailable because gated model access was not present.
This is an access result, not a model-quality result.

ClearML task `172ae6c5953c4f769177833ad17642ea` stores 22 aggregate artifacts
and eight plots. Feature caches, object-level scores, private crop panels, P1
panels, labels and credentials remain outside GitHub and ClearML.

The evidence remains a candidate because the Kaggle source snapshot recorded
uncommitted Notebook 07 work. The original run record is retained unchanged.
CSV line endings were normalised to LF for GitHub. The values are unchanged,
and `artifact_manifest.json` records the repository copy.
