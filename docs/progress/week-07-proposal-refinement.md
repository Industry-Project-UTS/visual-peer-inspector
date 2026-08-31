# Week 7 progress: DS1 proposal refinement

**Status:** Complete candidate evidence. The refinement improves precision, but
the automatic proposal route is still not accepted.

## Work completed

- evaluated five fixed, label-free proposal policies on all 100 DS1 scenes
- processed all 2,000 images and retained the original LocateAnything output
- tested area filtering and non-maximum suppression without using evaluator
  labels to choose proposals
- calculated precision, recall, F1, missed objects and extra proposals
- used paired confidence intervals grouped by the 27 base assets
- reviewed private before-and-after proposal examples
- verified the Kaggle output independently
- stored aggregate metrics, tables and figures in ClearML

## Result

The original proposal set reached 0.78668 precision, 0.94494 recall and
0.85858 F1. The selected diagnostic policy, area filtering followed by NMS at
0.75, reached 0.90995 precision, 0.94494 recall and 0.92712 F1.

The F1 gain was 0.06854. Its 95 percent asset-grouped interval was 0.01611 to
0.11977. Extra proposals fell from 1,973 to 720, but the number of missed
objects stayed at 424. The recall gate of 0.95 therefore still failed.

## Decision

Keep this refinement as a useful diagnostic and optional cleanup step. Do not
promote the current automatic proposal route. Filtering can remove duplicate
or extreme boxes, but it cannot recover an object that LocateAnything did not
propose.

## Next step

Continue the DS1 representation screen with supplied masks as the anchor and
reference-box masks as the practical segmentation route. Keep the automatic
route as a diagnostic comparison. In parallel, test a different detector or a
revised object prompt before repeating the automatic route acceptance test.
