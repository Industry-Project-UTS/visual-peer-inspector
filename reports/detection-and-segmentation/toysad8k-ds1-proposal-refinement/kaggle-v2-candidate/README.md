# DS1 proposal refinement, Kaggle Version 2

This folder contains the GitHub-safe aggregate output from Notebook 04.2.
The run covered all 100 DS1 scenes and all 2,000 images.

The selected diagnostic policy was area filtering followed by NMS at 0.75.
It increased precision from 0.78668 to 0.90995 while preserving recall at
0.94494. The proposal route is not accepted because recall remains below the
0.95 gate.

The files in this folder contain only aggregate tables, charts, configuration,
the decision record, completion gates and a ClearML receipt. Per-image
evaluator rows, proposal overlays, RGB images, masks, model caches and
credentials are excluded.

This is candidate evidence because the Kaggle code snapshot recorded an
uncommitted local source state. The result is suitable for implementation
review. A clean-snapshot rerun can replace it before final evaluation.
