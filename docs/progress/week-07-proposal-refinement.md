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
objects stayed at 424. The earlier 0.95 recall value was an internal target,
not an external acceptance rule. The missed objects and visual box quality are
the reasons the automatic route still needs review.

## Decision

Keep this refinement as a useful diagnostic and optional cleanup step. Do not
promote the current automatic proposal route. Filtering can remove duplicate
or extreme boxes, but it cannot recover an object that LocateAnything did not
propose.

## Bounded prompt follow-up

Notebook 04.3 compares the original `object` prompt with `individual object`.
A third wording was rejected by the one-image gate because the model returned
no boxes. It was not allowed to consume the full screen.

The real `individual object` run completed all 2,000 images with no unresolved
inference failures. Evaluation of the locked predictions selected
`area_nms_075_cap8`. This policy applies area filtering and NMS, then retains
at most the eight largest remaining proposals. The cap was derived from the
label-free DS1 proposal-count distribution, where 99.5 percent of cleaned
images had no more than eight proposals. It prevents rare responses with
dozens of background strips from dominating the automatic route.

The selected development result reached 0.97332 precision, 0.96195 recall,
0.96760 F1 and 0.96420 F2 at IoU 0.50. It missed 293 of 7,700 visible objects,
and 164 of 2,000 images still contained at least one missed object. Complete
detection was 0.918. These values were first recovered locally after the first
Kaggle notebook stopped on the rejected third prompt. The cleaned two-prompt
Notebook 04.3 Version 4 then reproduced and verified the selection on Kaggle.

## Decision

Use `individual object` with `area_nms_075_cap8` as the DS2 confirmation
candidate. Do not describe LocateAnything as fully automatic or perfect.
Keep the user-confirmed box route because postprocessing removes bad proposals
but cannot recover objects that the model did not find.

## Next step

Notebook 08B applied the selected model revision, prompt and post-processing to
the separate 300-scene DS2 role. Its operational timeout and retry policy was
versioned separately, and terminal generation failures remained visible as
zero-proposal results. Notebook 08C then measured supplied, assisted and
automatic routes separately without using DS2 results to retune the models.

The DS2 proposal result reached 0.9350 precision and 0.9383 recall at box IoU
0.50. It also confirmed the earlier limitation: only 87.9 percent of images had
complete object detection. The automatic route therefore continues with a
user-confirmed fallback.
