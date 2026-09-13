# Visual Peer Inspector web app

A separate Gradio application in the existing implementation repository. The
app runs without GPUs, model weights, datasets or recorded examples.

## Run locally

From the repository root, using Python 3.11 or newer:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r apps/web/requirements.txt
.venv/bin/python apps/web/app.py
```

Open <http://127.0.0.1:7860>. To use a different port, add `--port 7861`.
The app binds to localhost by default and does not create a public share link.
Use its own environment; do not add UI dependencies to running notebook jobs.

## What works now

- Upload, validate, preview and remove 2–5 real images, and start a new scene.
- Reject duplicate decoded images, unsupported formats, animated images,
  corrupt files and excessive image dimensions/sizes.
- Prepare the inspection/result/review/report flow independently of inference.
- Open the optional **How it works** guide for the planned pipeline and model
  roles. It starts closed, opens alongside the app on desktop, and uses a modal
  overlay on smaller screens. Close it with its button, Escape, or the backdrop.
- Render actual backend rankings, supporting-view counts, boxes, masks and
  heatmaps once an adapter returns them.
- After inference, edit boxes and cross-view identities, or add a box by
  clicking two corners. Reruns are explicitly labelled assisted.
- Export an HTML report with embedded evidence and a JSON result. HTML can be
  printed to PDF from a browser.

The default app says **Inspection not connected** and disables inspection.
Uploading images never generates substitute scores or replays another scene.
The results and report stay empty until actual inference completes.

## GPU inference adapter

The built-in adapter now connects uploaded images to LocateAnything, SAM 2.1,
DINOv2-small, P1 scoring, embedding-only identity matching and mean normalized
rank fusion. It runs in a separate Python process and environment. See
[`space/README.md`](space/README.md) for the Docker Spaces setup.

The local preview remains model-free by default. Neither imports, packaging,
CPU tests nor the deployment build download weights. On deployed hardware,
`VPI_ENABLE_INFERENCE=1` selects the built-in adapter and
`VPI_WORKER_PYTHON=/opt/inference/bin/python` selects its environment.
`VPI_ALLOW_MODEL_DOWNLOADS=1` explicitly allows pinned checkpoints to be
downloaded inside that deployment. With `0`, only existing caches are used.

The two environments are necessary: Gradio 6.27 requires huggingface-hub >=1.16,
while the selected Transformers 4.57.1 requires huggingface-hub <1.0. The UI does
not import the inference libraries. Requests and results cross the boundary as
temporary JSON/PNG files; there is no pickle or exposed inference endpoint.
One inspection runs at a time. Cancel and the configurable wall-time limit stop
the worker process group before deleting its temporary files.

Every stage uses real inference in deployment. The model stubs used by CPU
tests are confined to `tests/`, which is excluded from the deployment package.

### Build the deployment folder without running models

```bash
.venv/bin/python apps/web/check_deployment.py
.venv/bin/python apps/web/package_space.py --output .local/huggingface-space
```

The output directory must be empty. The package has a root Spaces README,
Dockerfile, dependency files, application source, and a SHA-256 manifest.
It excludes notebooks, research outputs, datasets, credentials, model weights,
tests and caches. Creating the folder does not deploy, upload, or build Docker.

### Optional alternative backend

Set `VPI_BACKEND=your_package.adapter:create_backend` when launching. The factory
must return an object implementing `InspectionBackend` from
[`core.py`](../../src/visual_peer_inspector/web/core.py). It is operator-side
configuration, not an input field in the web app. The factory loads once.

```python
class PipelineBackend:
    def inspect(self, request, progress):
        # request.views: opaque ID, metadata-free RGB pixels, content hash
        # request.corrections: None for automatic; corrected boxes for assisted
        progress("Finding objects")
        # Call the existing frozen pipeline here. Do not reimplement it in UI.
        # Return a validated InspectionResult with actual observations/ranking.
```

Supported progress labels are `Finding objects`, `Creating masks`,
`Extracting features`, `Comparing peers`, and `Combining views`. Report stages when they actually run.
The UI processes one queued operation at a time (maximum queue size eight).

The adapter must preserve the [frozen application requirements](../../reports/development-freeze/08f-kaggle-v1/safe/application_output_contract.json):
2–5 views, relative ranking, per-view evidence and explicit review states. The
Python dataclasses describe a UI transport boundary; they do not replace or
modify the frozen research artifacts. The adapter must map the internal policy
state `review_required` to the public contract state `review_recommended`.

Return the actual method and review-policy versions. Do not invent a calibrated
policy while calibration is pending. An adapter can identify a provisional
policy explicitly. Scores must be finite and ranked from most to least unusual.
Support counts must match observations. Masks and RGB heatmaps must be in the
original, EXIF-corrected image dimensions. Missing evidence is shown explicitly.

### Upload preprocessing and review policy

The separately versioned manifest is
`src/visual_peer_inspector/web/pipeline_config.json`. Aspect-preserving
letterboxing maps uploads into the 256×256 model canvas; masks, boxes and
heatmaps are restored to original image coordinates. The selected crop and
P1 settings are retained. Invalid masks have no fabricated score and remain
available for correction. Scoring requires at least three usable peers in a
view; fused winners need scored observations in at least two views. These
deployment adaptations are not a claim of unchanged research accuracy.

All usable results default to `review_recommended`, with the explicit
`provisional-review-all-v1` policy version. To supply an approved policy, set
`VPI_REVIEW_POLICY` to a deployment-local JSON file with exactly these fields:

```json
{
  "schema_version": "vpi-web-review-policy/v1",
  "pipeline_sha256": "<fingerprint printed by check_deployment.py>",
  "policy_configuration": "<the complete ReviewPolicy object from the approved policy>"
}
```

`policy_configuration` must be a JSON object, not the explanatory string above.
Its exact fields are defined by `ReviewPolicy` in `src/visual_peer_inspector/review_policy.py`.
The app rejects a partial policy or a fingerprint mismatch. Do not re-label a
research policy as validated for resized arbitrary uploads without confirming
its applicability. Reports include the policy hash and upload-pipeline identity.

The code and CPU contracts can be verified locally. A fresh Linux container
build and real checkpoint/GPU runs are still required later to verify package
compatibility, memory, latency, model quality and the complete hosted journey.

## Temporary data

The app strips EXIF and other source metadata from inference inputs. Original
filenames and source paths are not part of `InspectionRequest`. Gradio retains
temporary upload/preview/download files and removes files older than 30 minutes
on a one-minute cleanup interval. Reset clears session results and the app's
export directory; Gradio's copies remain until its cleanup interval. Session
state expires after 30 minutes. Nothing is written to the research dataset or
experiment outputs. A downloaded report intentionally contains the user's
inspection images.

Public hosting, access controls and GPU deployment are separate integration
steps. This local workspace has no authentication and should not be exposed to
the internet as-is.

## Verification

```bash
PYTHONPATH=src GRADIO_ANALYTICS_ENABLED=False .venv/bin/python -m pytest tests/test_webapp.py tests/test_web_pipeline.py -q
```

Tests create small synthetic images in temporary directories. They test upload
validation, backend boundaries, result invalidation, assisted labels, HTML
escaping, report cleanup, real CPU P1/matching orchestration, image transforms,
worker cancellation/timeouts and deployment packaging. They are not
model-accuracy tests or app examples. Test packages require pytest, NumPy and
SciPy in addition to the UI requirements; they do not require torch or SAM.
