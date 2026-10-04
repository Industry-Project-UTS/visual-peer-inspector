# Hosted app display and workflow validation

The SageMaker app was exercised through Chrome using fresh three-view inputs,
confirmed correspondence, saved cases and operator actions. The selected scoring
methods, model assets and calibrated review policy are unchanged.

## Repairs

- Heatmaps now use a blue/cyan/yellow/red scale with one foreground blend instead
  of a faint, twice-blended orange overlay. Each object is scaled separately.
  Coordinates still come from the existing crop-to-image projection. These maps
  show coarse mismatch, not validated defect boundaries or probabilities.
- Saved cases retain their original maps and receive the appropriate legacy
  legend. A missing foreground mask displays the original image and an explicit
  caption instead of an unexplained black image.
- Expected Decision form validation appears inline. A required operator field
  is available in Decision; failed validation preserves the entered finding and
  machine-result cards. Human decisions remain separate from model predictions.
- Review's preview no longer receives stale updates from programmatic dropdown
  changes. Display callbacks run on user input; completion resets the layer to
  Boxes. Score-table view labels follow upload order.

## Verification

The focused web/SageMaker suite passed **155 tests and 12 subtests**, with one
private-cache test skipped. Hosted checks covered all five evidence layers,
single-object filtering, gallery navigation, box addition/cancellation/removal,
identity conflict validation, all four human decision actions, report creation,
saved-case reopening, live-session preservation and clean follow-up creation.
Earlier 2/3/5-view and cancellation checks are retained in the previous readiness
receipt; they are not presented as repeated during this display repair.

The final **151-file** source manifest matched the deployed Space with zero
mismatches. The server restarted successfully. Restart health alone does not
establish inference accuracy; the fresh inference and UI checks precede it.

### Additional curated examples

| Scene | Route | Selected-object support | Outcome |
|---|---|---:|---|
| Dinosaurs, three views | Automatic | 1/3 | Fragmented match; retain as review example |
| Fish, three views | Automatic | 2/3 | Unusual fish selected in two views; review required |
| Fish, three views | Correctly confirmed Assisted | 3/3 | Fish without dorsal fin selected; frozen Result ready state |

The valid assisted fish run reused nine candidates and encoded none. Its relative
score was 1.000 and ranking-confidence index 0.273. An initial manual mapping
swapped two physical objects in View 2; it was corrected after inspecting the
original photos and is excluded from valid assisted conclusions. These curated
demonstrations do not change evaluation accuracy or its denominator.

## Remaining external gates

The free ngrok domain, official agent and private token file are prepared.
The account owner still needs to enter the new demo login password. Public
forwarding, external login/upload/download and same-address restart remain
unverified. The private Studio app is available. See the
[restart and sharing instructions](../../apps/web/sagemaker/README.md).

Chrome blocked the two attempted new ZIP downloads with an organisation-policy
message. Browser security was not bypassed. The ZIP/HTML/JSON files remain on
the Space; three case archives reopened there and rendered all five layers
without inference. Earlier local portable cases also remain available.

See the [current validation receipt](../../reports/web-integration/safe/sagemaker-feature-validation.json)
and [earlier demo readiness receipt](../../reports/web-integration/safe/sagemaker-readiness.json).
Private screenshots, original photos, case archives and credentials are excluded
from this source change. Maximum-size sustained concurrent load is outside this
bounded validation.
