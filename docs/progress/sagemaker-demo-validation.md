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

## Public startup and responsiveness verification

The protected free ngrok address now serves the app. A real Space stop/start
was verified: the persistent Jupyter extension started the app and tunnel without
manual terminal launch. A private status receipt verified authentication and new
process IDs. AWS provisioning still takes time; the address is offline while the
Space is stopped. See the [startup instructions](../../apps/web/sagemaker/README.md).

The final 153-file application package matched SageMaker with no hash mismatches.
The updated suite passed **164 tests and 12 subtests**, with one private-cache test
skipped. The earlier feature checks above remain distinct from this latest run.

- Progress and cancellation sit below the active tab's actions. The display
  shows stage detail, elapsed time and a count of completed stages. Its percentage
  is a stage counter, not a prediction of remaining time.
- Page initialisation bypasses the inspection queue. Cancellation and browser
  refresh were tested against active hosted work; the worker stopped and the GPU
  returned to idle. Refresh starts a new inspection session rather than resuming
  unsaved results.
- The 390-pixel form has no horizontal overflow; mobile help opens as a dialog.
- The Decision tab now explicitly describes the human record as optional. Mask
  overlays are coloured; Objects on black uses the same masks to remove background.

### Measured performance

DINO now loads once per inspection, preserving CPU FP32 and the original four-crop
forward batches. In one controlled fresh three-view fish benchmark (nine objects),
backend time fell from **39.152 to 28.602 seconds**, and DINO time from **15.660 to
5.493 seconds**. The selected object, scores and supporting views were identical.
A fresh public-browser run displayed **29 seconds**. The user's three-view bottle
run displayed **50 seconds**. The first detector checkpoint load after the real
Space restart took about **57 seconds** by itself. These individual measurements
are not a latency guarantee or an accuracy estimate. Rehearse once after boot.

See the [performance and startup receipt](../../reports/web-integration/safe/sagemaker-performance-startup.json).

## Remaining limits

Chrome blocked the two attempted new ZIP downloads with an organisation-policy
message. Browser security was not bypassed. The ZIP/HTML/JSON files remain on
the Space; three case archives reopened there and rendered all five layers
without inference. Earlier local portable cases also remain available.

See the [current validation receipt](../../reports/web-integration/safe/sagemaker-feature-validation.json)
and [earlier demo readiness receipt](../../reports/web-integration/safe/sagemaker-readiness.json).
Private screenshots, original photos, case archives and credentials are excluded
from this source change. Maximum-size sustained concurrent load is outside this
bounded validation.
