# Deployment preparation verification — 12 September 2026

The application and deployment code were originally prepared on
`codex/inspection-webapp`. The interface is now merged through
[PR #17](https://github.com/Industry-Project-UTS/visual-peer-inspector/pull/17), at main commit
`bf85c1bc6b066e7d26cc42a7b2281455e77a16fe`. The inference/deployment code is the subsequent change.
No model weights were downloaded, no Kaggle jobs were started, no Docker image
was pulled or built, and no hosted deployment was performed during preparation.

## Verified locally

- 42 tests and 9 subtests passed across `test_webapp.py` and `test_web_pipeline.py`.
- Ruff checks and formatting, Git whitespace checks, and the model-free
  deployment preflight passed.
- CPU tests execute the actual crop preparation, P1 scoring, identity matching,
  normalized-rank fusion, review decisions, overlay restoration and report code.
  GPU-stage outputs are injected by fixtures confined to `tests/`.
- Subprocess tests cover JSON/PNG transport, successful results, worker failures,
  cancellation scoped to a request, timeout, process cleanup and temporary files.
- Tests cover empty masks, fewer than three peers, missing views, assisted IDs,
  non-square image transforms, policy fingerprint mismatch and path rejection.
- Runtime guard tests check lazy imports, offline cache access, CUDA checks before
  checkpoint access, pinned revisions and truncated detection responses.
- Browser checks against a temporary test-only worker covered real file uploads,
  automatic results, mask/heatmap controls, assisted reruns, HTML/JSON downloads,
  exported provenance, empty state in a fresh session and mobile cancellation.
  The temporary QA server was then stopped. The normal preview remains model-free.
- PyPI dependency metadata confirms that Gradio and Transformers need separate
  Hugging Face Hub versions. The Dockerfile creates separate environments.
- Registry metadata confirmed the pinned Python base image and Linux amd64
  platform. Only metadata was read; no image layers were downloaded.

## Checks reserved for deployed GPU hardware

- Build the Linux container and verify the complete resolved dependency graph.
- Fetch and load the real pinned checkpoints in the Space.
- Verify GPU memory, cold-start time, normal inspection time and cancellation
  while actual model kernels/downloads are active.
- Run new images through the complete hosted journey and inspect model outputs.
- Confirm or supply the review policy appropriate for this upload adapter.

These pending checks are deliberately not represented as passing. A packaged
application and passing CPU tests do not establish checkpoint compatibility,
GPU performance, calibrated accuracy or production acceptance.
