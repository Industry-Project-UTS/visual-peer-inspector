# DS1 representation and peer scores, Kaggle Version 3

This folder contains the GitHub-safe aggregate output from Notebook 07.1. The
run covered all 100 DS1 scenes and all 2,000 views.

DINOv2-small remains the selected encoder. P1 improved mean scene average
precision from 0.64074 with B0 to 0.66721 on the supplied-mask route. The
asset-grouped confidence interval for the gain was 0.01208 to 0.04295.

DINOv2-small with registers and DINOv2-base did not pass the fixed promotion
rule. DINOv3 access was unavailable in Kaggle and this outcome was recorded
without substituting a different model under the DINOv3 name.

The files in this folder contain aggregate tables, charts, configuration,
completion gates, the promotion decision and a ClearML receipt. Feature caches,
per-scene score shards, RGB images, masks, model inputs and credentials are
excluded.

This is candidate evidence because the Kaggle code snapshot came from an
uncommitted local source state. It is suitable for implementation review. A
clean-snapshot rerun can replace it before final evaluation.
