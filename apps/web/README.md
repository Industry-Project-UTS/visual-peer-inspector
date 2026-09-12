# Visual Peer Inspector web app

A separate Gradio application in the implementation repository. It starts
without GPU libraries, model weights, datasets or recorded examples.

## Run locally

From the repository root, using Python 3.11 or newer:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r apps/web/requirements.txt
.venv/bin/python apps/web/app.py
```

Open <http://127.0.0.1:7860>. Add `--port 7861` for another port. The server binds
to localhost and does not create a public share link.

## Interface

- Upload, validate, preview and remove 2–5 JPEG, PNG or WebP images.
- Use Scene, Results, Review and Report to move through an inspection.
- Open the optional How it works panel for model roles. It starts closed,
  opens alongside the desktop app and uses a modal on smaller screens.
- Display backend rankings, supporting views, boxes, masks and heatmaps.
- Correct boxes and cross-view identities; reruns are labelled assisted.
- Export HTML with embedded evidence and JSON with method/policy provenance.
- Start over to clear the current scene, results and app export files.

The default interface displays **Inspection not connected** and disables
inspection. Uploads do not create predictions. Results and reports require a
backend response. The built-in GPU worker and Hugging Face Spaces packaging
are delivered in the subsequent inference/deployment change.

## Backend contract

Set `VPI_BACKEND=your_package.adapter:create_backend` to load an operator-supplied
adapter implementing `InspectionBackend` in
[`core.py`](../../src/visual_peer_inspector/web/core.py). The UI has no GPU imports.
The adapter receives metadata-free RGB views and optional corrected boxes,
and returns a validated `InspectionResult`. It must report actual method and
policy versions, accurate support counts and evidence in original image sizes.
Cancellation is optional; the UI offers it when the adapter implements it.

The transport contract supports request IDs, nullable scores for unusable
observations, cancellation and provenance so the next PR can add inference
without changing the interface contract. It preserves the existing
[application output requirements](../../reports/development-freeze/08f-kaggle-v1/safe/application_output_contract.json).

## Temporary data and access

Inference inputs exclude original filenames and source metadata. Session
state expires after 30 minutes. Gradio checks every minute for temporary files
older than 30 minutes. Reset clears app exports; Gradio copies expire on its
cleanup interval. Downloaded reports contain the inspection images.

Set both `VPI_AUTH_USERNAME` and `VPI_AUTH_PASSWORD` to enable optional basic
authentication. Keep credentials outside Git. Public hosting and real GPU
acceptance are subsequent deployment steps.

## Verify without models

```bash
.venv/bin/python -m pip install pytest
PYTHONPATH=src GRADIO_ANALYTICS_ENABLED=False .venv/bin/python -m pytest tests/test_webapp.py -q
git diff --check
```

Tests generate small synthetic images for upload validation, state clearing,
backend/result contracts, assisted mode, evidence rendering, report escaping
and cleanup. Test fixtures are not product examples or evidence of model quality.
