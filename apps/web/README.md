# Visual Peer Inspector web app

Upload 2–5 views of the same arrangement and obtain one combined object ranking.
The **Images**, **Results**, **Review** and **Report** tabs support two explicit
modes. Three views are the main application setting; two and five are also
supported. More views do not guarantee a better result.

The app is organised as a small-batch inspection and rework desk: an operator
opens a case for 3–5 similar items, photographs them from several angles,
investigates the unusual item, records a physical finding in **Decision**, saves
a portable case file, and can reopen it later in **Saved cases** or start a
linked follow-up inspection. See [Inspection cases](#inspection-cases).

The first screen keeps capture and mode selection together. Reference details
are optional and the case ID is generated automatically. Results show the
combined ranking and image evidence first; score components remain available
in a collapsed section.

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
comparisons, not correctness probabilities. The frozen calibration policy keeps
automatic results in **Review required**. Assisted results can be **Result ready**
only with confirmed object matches, a unique leader, support in every uploaded
view, and a ranking confidence index meeting the fixed threshold for 2, 3 or 5
views. Four-view inspections remain reviewable but cannot claim readiness.

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

### Optional Groq explanation

Set `GROQ_API_KEY` in the app process environment before starting the server to
enable the **Explain the ranking** button. The app uses Groq's text-only
`openai/gpt-oss-20b` chat completion endpoint with an eight-second timeout.
Clicking the button sends only the view count, review state, relative scores and
number of views supporting up to five ranked candidates. It does not send
photos, object IDs, case fields, decisions, embeddings or cache handles. The
generated text is temporary, is not included in the saved case, and cannot
change the model result. The app continues normally if Groq is unconfigured or
unavailable. Do not put the API key into source files or a case ZIP.

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
cleans temporary uploads and downloads after 30 minutes. A downloaded report or
case file intentionally contains the user's inspection images and the names typed
into the case and decision fields.

The Review tab presents proposed IDs across all views. The user must confirm
the identity map before assisted scoring; editing rows clears that confirmation.
Predicted suggestions are not treated as confirmed identities by themselves.
HTML exports embed images and score tables; JSON preserves all public evidence
and method/provenance fields without model tensors or server cache secrets.

## Inspection cases

Each session works on one inspection case: a case ID (generated, editable), an
item or product description, an optional batch or reference, the operator's name
and the creation time (UTC). New photos start a new capture within the same case;
**Start new case** begins a fresh one. A short timeline records photo loads,
inspections, match confirmations, operator decisions and saves.

**Results** first shows the model result and an object board: one row per
physical object, with its crop from every uploaded view, its rank and combined
score, tied leaders marked as tied, and an explicit "Not linked" slot where a
view has no matching observation. Observations that were never linked across
views are listed separately. The evidence gallery follows, and the ranking and
component score tables sit in a collapsed section.

**Decision** keeps the human record apart from the model. The operator chooses
the object the finding refers to (or the whole peer group), an action (inspect
further, hold, correct / rework, no issue found), a finding, a short reason and
an optional responsible person. Nothing is preselected. Recording a decision does
not change scores, the selected object or the review state, and a decision naming
one object says nothing about the others. **Result ready** and **Review required**
remain model review-routing states; the case status (open, on hold, rework
required, closed) comes only from the operator's decision on the latest result.
A new inspection result needs a new decision; earlier decisions stay in history.

**Report → Save case and report** writes three files:

| File | Contents |
| --- | --- |
| `<case>-case.zip` | `case.json`, lossless views, object masks and heatmap evidence |
| `<case>-report.html` | Case details, model result, operator decision, object board, tables, evidence, match corrections, decision history, timeline |
| `<case>-result.json` | The same machine and case record without images |

`case.json` holds the public machine evidence and provenance used by the JSON
report, the confirmed identity/box corrections (with counts of boxes added,
edited or removed relative to the model proposals), the operator decisions and
the timeline. It never contains session IDs, cache handles, embeddings, patch
tensors or local file paths. Heatmap pixels are stored only inside each object's
mask and box, which is all the viewer draws.

**Saved cases** reopens a case file read-only. It never calls a backend, runs a
model or creates an inference cache. Import is strict: only expected file names,
bounded sizes and counts, strict JSON without NaN, exact field sets, view pixels
that reproduce their recorded SHA-256 and view ID, overlays of the right size,
a model result that passes the same checks as live results and reproduces its
saved summary, and a case status consistent with its decisions. The file is not
signed; these checks detect damage and inconsistency, not a deliberate forgery.

**Start a follow-up inspection** opens a new case linked to the saved one (its
ID, file fingerprint, model outcome and operator decisions). The follow-up is a
new capture of the full peer group. Both records are shown side by side, and
scores are never compared across cases: a lower score after rework does not show
that the repair worked.

The launcher accepts case uploads up to 100 MiB. Save rejects a larger archive
before showing success. Each photo is still limited to 10 MiB and 12 MP.

## Build and verify

```bash
.venv/bin/python -m pip install -r apps/web/space/requirements-scoring.txt pytest==8.4.2
PYTHONPATH=src GRADIO_ANALYTICS_ENABLED=False .venv/bin/python -m pytest tests/test_webapp.py tests/test_web_pipeline.py tests/test_web_two_mode.py tests/test_web_presentation.py tests/test_web_packaged_decision.py tests/test_web_frozen_policy.py tests/test_web_case.py -q
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
The original 1000/600/400 role assignment remains unchanged. Development,
calibration and internal-lock results have been preserved with their exposure
qualifications; they are not an accuracy guarantee for arbitrary uploads.

`pipeline_config.json` records fixed model revisions and method settings. The
deployable package includes a byte-verified copy of the sealed 09B two-mode
review policy. The confidence index is a ranking-separation signal for review
routing, not a probability that the selected object is correct. Old P1 policy
envelopes are rejected. The UI/transport dataclasses are not a replacement for
sealed research artifacts. An operator may set `VPI_BACKEND=module:factory` for another explicit
adapter; web users cannot select arbitrary Python modules.
