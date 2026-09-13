---
title: Visual Peer Inspector
emoji: 🔎
colorFrom: gray
colorTo: gray
sdk: docker
app_port: 7860
fullWidth: true
models:
  - nvidia/LocateAnything-3B
  - facebook/sam2.1-hiera-large
  - facebook/dinov2-small
short_description: Compare similar objects across multiple views of a scene.
---

# Visual Peer Inspector

Upload 2–5 views of the same scene, compare peer objects, review the evidence,
correct boxes or identities, and export an inspection report.

This package contains application code. It does not include model weights,
datasets, training labels, recorded predictions or credentials.

## Deployment settings

1. Choose a CUDA GPU in Space settings. Exact memory and latency still need
   verification on the chosen hardware; CPU Basic cannot execute this route.
2. Set the Space variable `VPI_ALLOW_MODEL_DOWNLOADS=1`. This enables downloads
   of the pinned checkpoints **inside the Space**, on the first inspection.
   The default is `0`, which permits existing cached weights only.
3. Use a private Space while validating. For an additional app login, set both
   `VPI_AUTH_USERNAME` and `VPI_AUTH_PASSWORD` as Space secrets. Never put tokens
   or passwords in source files. `HF_TOKEN` can be supplied as a secret if needed.
4. Check the build logs, then complete the verification steps below before
   sharing the app. A successful build alone does not verify GPU inference.

The Space starts its web server before loading models. Loading happens in an
isolated worker, one model stage at a time, to limit peak GPU memory. Downloads
are cached under `HF_HOME`; ephemeral cache contents may disappear on a Space
restart. No prediction cache is used. No images leave the Space for inference.

## Runtime controls

| Variable | Default | Purpose |
| --- | --- | --- |
| `VPI_ENABLE_INFERENCE` | `1` in this image | Set `0` for an interface-only preview |
| `VPI_ALLOW_MODEL_DOWNLOADS` | `0` | Set `1` to fetch pinned weights in the deployment |
| `VPI_WORKER_PYTHON` | `/opt/inference/bin/python` | Isolated model environment |
| `VPI_INSPECTION_TIMEOUT` | `1800` | Maximum seconds per inspection, including downloads |
| `VPI_REVIEW_POLICY` | unset | Optional policy JSON bound to this pipeline |
| `HF_HOME` | `/home/appuser/.cache/huggingface` | Model cache; use a mounted persistent path if configured |

Cancel inspection stops the worker process and removes its temporary files.
The app runs one inspection at a time, with a bounded queue. Uploads and reports
expire after 30 minutes; downloaded reports intentionally contain their images.

## Method and review status

The manifest in `src/visual_peer_inspector/web/pipeline_config.json` records
every model revision and the app's method settings. The selected prompt, SAM,
crop, DINOv2, P1 and embedding-only identity code are reused from the research
implementation. Uploads are letterboxed to 256×256 while preserving aspect
ratio, and evidence is mapped back to original image coordinates. This upload
adapter is separately versioned; research benchmark accuracy is not a claim
about arbitrary uploaded photos.

P1 needs at least three usable peers in a view. Unscoreable objects stay in the
review screen with a missing score. Mean normalized ranks combine scored
observations; a winner needs scoring support in at least two views. No normal
reference library or dataset metadata is required at inference time.

Until a matching review policy is supplied, all usable results are marked
**Review recommended**, with a provisional policy version. This is not a
calibrated defect detector. The optional policy format is documented in
`apps/web/README.md`. HTML/JSON reports include actual method and policy versions;
JSON also contains model revisions, runtime package versions and score meaning.

## Verification after deployment

- Upload new real images and run the full automatic pipeline. Confirm masks,
  heatmaps and boxes align with the original landscape/portrait images.
- Correct a box and identity, rerun, and confirm the report says **assisted**.
- Export HTML and JSON and check versions, scores and supporting views.
- Exercise fewer than three peers, no proposals, empty masks, uncertain matches,
  cancellation, timeout and a second browser session. Errors must not fabricate
  scores or expose another session's images.
- Record cold-start time, warm inspection time and peak GPU memory for 2–5 views.

Code and CPU contract checks are available before deployment. Live checkpoint
compatibility, GPU memory and end-to-end model quality require this later test.

Platform reference: [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker).
