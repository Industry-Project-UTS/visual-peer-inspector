# Visual Peer Inspector web app

Upload 2–5 views of the same arrangement and obtain one combined object ranking.
The existing **Images**, **Results**, **Review** and **Report** tabs support two
explicit modes. Three views are the main application setting; two and five are
also supported. More views do not guarantee a better result.

## Decision pipeline

```text
Uploaded RGB views → LocateAnything boxes → SAM 2.1 masks → DINOv2-small
    → Automatic: per-view forward/P1/P2 consensus → compatible winner clusters
    → Assisted: confirm neutral object IDs → cross-view peer patch banks
    → combined ranking → selected object on every supported view
```

Automatic uses the frozen `winner_cluster_consensus` adapter. Every valid
candidate remains available for review, although only per-view winners enter
this method's ranking. Other observations are explicitly unlinked. Assisted
uses `object_bank_top2`: exclude the target physical object from peer banks,
score its foreground patches, normalize scores within each view, then average
its two strongest available view ranks. With one usable observation, use that
one and disclose the limited support. Identity confirmation does not choose the
anomaly; the scorer still ranks the objects.

Results contain consistent IDs where supported, all-candidate scores,
forward/P1/P2 components, fusion contributors, boxes, SAM masks, objects on
black, and coarse patch evidence. Tied leaders do not receive a unique selected
box. A missing link is not proof that an object is absent. Scores are relative
comparisons, not calibrated probabilities. Every usable result currently
recommends review.

## Local interface

Use Python 3.12 and a separate app environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r apps/web/requirements.txt
.venv/bin/python apps/web/app.py
```

Open `http://127.0.0.1:7860`; add `--port 7861` to change the port. This default
starts an interface preview with inspection disabled. It does not manufacture
scores or use a dataset lookup. The app binds to localhost without a public
share link. Do not modify a running notebook environment to add UI packages.

## Inference and runtime separation

The real adapter runs uploaded images in a bounded worker subprocess. Set
`VPI_ENABLE_INFERENCE=1` and point `VPI_WORKER_PYTHON` to its environment. The
[Docker setup](space/README.md) defines three environments:

| Environment | Responsibility |
| --- | --- |
| Web | Gradio, image validation, review and export |
| Inference | LocateAnything/SAM on CUDA; CPU matching and score fusion |
| DINO | Fixed DINOv2-small CPU FP32 runtime matching the executed feature cache |

LocateAnything's Transformers version differs from the DINO experiment's
version. `VPI_DINO_PYTHON` selects the isolated DINO interpreter; do not upgrade
LocateAnything's runtime to make the dependency conflict disappear. The DINO
adapter rejects incompatible versions. Requests use JSON/PNG and tensor
boundaries use NPZ with pickle disabled. Only configured model downloads are
allowed (`VPI_ALLOW_MODEL_DOWNLOADS=1`); the default uses cached pinned weights.

GPU detection and segmentation on new uploads still require suitable hardware.
CPU tests and recorded-stage replay establish integration behavior; they do not
establish GPU latency, Linux container compatibility or accuracy on new photos.

## Session data and corrections

A database is unnecessary for this demo. Each session has an in-memory object
map and private JSON/NPZ candidate files. Cache access uses an opaque session
capability. Entries are bound to image pixels, boxes, preprocessing, source
versions and tensor checksums. Reports never contain cache handles or embeddings.

Changing only identities reuses masks and DINO features. Changed boxes/images
invalidate the affected candidates. Unknown, expired, corrupt or incompatible
cache handles produce an error; they do not silently launch expensive inference.
Reset, cancellation and session expiry release cache data. Gradio independently
cleans temporary uploads and downloads after 30 minutes. A downloaded report
intentionally contains the user's inspection images.

The Review tab presents proposed IDs across all views. The user must confirm
the identity map before assisted scoring; editing rows clears that confirmation.
Predicted suggestions are not treated as confirmed identities by themselves.
HTML exports embed images and score tables; JSON preserves all public evidence
and method/provenance fields without model tensors or server cache secrets.

## Build and verify

```bash
.venv/bin/python -m pip install -r apps/web/space/requirements-scoring.txt pytest==8.4.2
PYTHONPATH=src GRADIO_ANALYTICS_ENABLED=False .venv/bin/python -m pytest tests/test_webapp.py tests/test_web_pipeline.py tests/test_web_two_mode.py tests/test_web_presentation.py tests/test_web_packaged_decision.py -q
.venv/bin/python apps/web/check_deployment.py
.venv/bin/python apps/web/package_space.py --output .local/huggingface-space
```

The destination must be empty. The source package contains the complete selected
scoring dependency tree and a SHA-256 manifest. It excludes datasets, model
weights, notebooks, results, credentials, caches, QA replay scripts and tests.
Packaging does not upload, deploy or run Docker.

For local recorded-development QA, use `scripts/verify_web_cached_replay.py`
with an explicit private artifact configuration. It verifies RGB/crop/checkpoint
hashes, exercises 2/3/5 views, exports evidence, and asserts that identity-only
reranking performs zero additional model-stage calls. Dataset correspondence is
labelled as such. This QA tool is not part of the deployed app.

See [verification evidence](VERIFICATION.md) for what was actually checked.

## Evaluation boundary

The selected methods come from a 100-scene development cohort with 27 asset
groups. Automatic and supplied-correspondence results have separate denominators
and labels in the report. Supplied correspondence is not evidence of reliable
human identity confirmation, and a selected gallery is not evaluation accuracy.
The original 1000/600/400 role assignment remains unchanged. The remaining
notebooks require an exposure audit and compatible current-method adapters
before protected results can be reported.

`pipeline_config.json` records fixed model revisions and method settings. Old external P1 review-policy envelopes are rejected because their thresholds
do not apply to these two scoring modes. The current route uses the provisional
review-all policy; a calibrated two-mode policy is not enabled. The UI/transport dataclasses are not a replacement for sealed research
artifacts. An operator may set `VPI_BACKEND=module:factory` for another explicit
adapter; web users cannot select arbitrary Python modules.
