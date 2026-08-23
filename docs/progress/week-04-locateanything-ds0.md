# Week 4 LocateAnything DS0 proposals

Notebook 04 tested whether the frozen LocateAnything-3B model can generate
object boxes reliably before SAM 2.1 is added to the pipeline. It used only the
opaque colour images prepared by Notebook 03. Reference boxes, masks, anomaly
labels, source identifiers, and private crosswalks did not enter the model path.

The work was run in stages. A one-image check confirmed that the model could
load and return parseable boxes. A 20-image pilot then covered one image from
each selected scene. The pilot completed successfully, and an exact repeat of
the same 20 images produced identical boxes.

Kaggle Version 13 then processed all 400 images on GPU T4 x2. Every image
completed, all 20 views were covered, and the notebook reported no failures,
truncated responses, zero-proposal images, invalid boxes, or duplicate boxes.
The full run produced 1,616 boxes. Mean inference time was 0.436 seconds per
image, with a p95 time of 0.601 seconds.

ClearML recorded the safe aggregate configuration, metrics, tables, charts,
hashes, and task artifacts under `Visual Peer Inspector/02 Detection and
Segmentation`. Raw images, per-image responses, proposal coordinates, private
overlays, and credentials were not uploaded.

The visual samples are useful but are not a final quality score. Most normal
examples show separate boxes around the toys. Some high-count outliers include
extra background regions. The next evaluator-only notebook must compare these
frozen proposals with the supplied reference boxes before the model is accepted,
adjusted, or replaced.

Version 13 is kept as candidate evidence because its run record reports
`git_dirty: true`. A later clean-snapshot rerun can replace the candidate folder
without changing the recorded result.
