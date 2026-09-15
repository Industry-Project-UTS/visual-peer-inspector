---
title: Visual Peer Inspector
emoji: 🔎
colorFrom: teal
colorTo: indigo
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

Upload 2–5 views of the same scene, inspect a combined ranking, confirm object
identities when needed, and export the visual evidence. Automatic and Assisted
are separate modes. Both keep the Images, Results, Review and Report workflow.

The package contains code only. It excludes datasets, labels, recorded
predictions, model weights, credentials and local QA adapters.

## Deployment

1. Select a CUDA GPU for LocateAnything and SAM. Memory and latency require a
   smoke test on the chosen hardware; CPU Basic cannot run the full new-image route.
2. Set `VPI_ALLOW_MODEL_DOWNLOADS=1` to fetch the pinned checkpoints inside the
   deployment. Default `0` permits existing model caches only.
3. Validate in a private Space. Optional `VPI_AUTH_USERNAME` and
   `VPI_AUTH_PASSWORD` belong in Space secrets, as does `HF_TOKEN` if needed.
4. Check the build and full-image inference before sharing the application.

The Dockerfile separates Gradio, the detector/SAM worker and the exact DINO CPU
runtime. LocateAnything retains its established Transformers environment;
DINOv2-small uses the executed experiment's torch/Transformers/NumPy/Pillow
versions in `/opt/dino`. The DINO worker refuses incompatible versions or model
bytes. CPU scoring uses the pinned NumPy/SciPy/Pandas stack. No model is loaded
at import or during packaging. CUDA checkpoints run sequentially to limit peak
memory. This build recipe is not evidence that a Linux GPU deployment has run.

| Variable | Default in this image | Purpose |
| --- | --- | --- |
| `VPI_ENABLE_INFERENCE` | `1` | `0` starts an interface preview |
| `VPI_ALLOW_MODEL_DOWNLOADS` | `0` | Permit pinned model downloads explicitly |
| `VPI_WORKER_PYTHON` | `/opt/inference/bin/python` | Detector/SAM and CPU decision worker |
| `VPI_DINO_PYTHON` | `/opt/dino/bin/python` | Exact DINO CPU feature worker |
| `VPI_INSPECTION_TIMEOUT` | `1800` | Maximum inspection time, including downloads |
| `HF_HOME` | `/home/appuser/.cache/huggingface` | Downloaded model cache |

Requests use temporary JSON/PNG files and feature tensors use NPZ without
pickle. The app runs one inspection at a time with a bounded queue. Cancel
terminates the process group. No user image is sent to another inference
service. Model caches may disappear on an ephemeral Space restart.

## Decisions and evidence

Automatic uses forward-patch/P1/P2 winner consensus and compatible winner
clusters. Assisted requires neutral identity confirmation and uses the selected
`object_bank_top2` patch-bank scorer. The scorer chooses the anomaly after
matching is confirmed. All candidates remain reviewable, including candidates
that automatic winner selection did not retain.

Results show per-view observation scores, component scores, normalized ranks,
fusion contributors, masks, boxes and coarse patch evidence. Only a unique
leader is highlighted, wherever its identity has a supported observation.
Ties and missing links are explicit. Relative scores are not probabilities,
and results use a provisional review-all policy. Old P1 review thresholds are
not valid for these two modes.

Masks and crop heatmaps map back to the original EXIF-corrected image. A cache
binds each candidate to pixels, boxes, preprocessing, model settings, source
hashes and tensor checksums. Identity-only edits reuse its features; changing a
box or image recomputes affected candidates. Unknown or corrupt cache handles
fail visibly instead of silently repeating neural inference.

Session caches and temporary uploads expire after 30 minutes. Reset and
cancellation clear the app cache; Gradio cleans its upload/download copies on
its own interval. Reports embed inspection images intentionally but omit cache
capabilities and raw model tensors. A database is not required for this demo.

## Deployment checks still required

- Build the Linux image and check the three environments and complete source tree.
- Run new 2/3/5-view images through real LocateAnything, SAM and DINO.
- Check portrait/landscape mask, box and heatmap coordinates.
- Confirm IDs without box edits and verify zero additional neural calls.
- Edit one box and verify only its affected candidates are recomputed.
- Check ties, insufficient peers, empty masks, missing views, expired caches,
  cancel, timeout, export and isolation between two browser sessions.
- Record cold/warm latency, host RAM and peak GPU memory.

Development benchmark accuracy is not an accuracy guarantee for arbitrary
uploads. Supplied dataset correspondence and human-confirmed correspondence
must be labelled separately in evaluation.

References: [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker),
[PyTorch CPU installation](https://pytorch.org/get-started/locally/).
