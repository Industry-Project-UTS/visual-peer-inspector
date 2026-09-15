# Runtime and recovery record

This stage currently runs on the local CPU. Kaggle execution is a fallback and has not been claimed as verified.

| Observed condition | Resolution | Scientific effect |
|---|---|---|
| Jupyter cannot bind localhost ports inside the sandbox | Execute the local kernel with authorised loopback access | None; no model or data setting changes |
| Re-entering preparation rejects an existing output directory | The notebook launcher verifies a complete prepared package and its configured source hashes before reuse | No input rewrite or feature extraction repeated |
| Local float32 DINO differs slightly from earlier CUDA output | Keep the previous cached score and a regenerated baseline as separate methods | Avoid attributing device/precision differences to the patch-bank change |

A failed notebook attempt and its local log remain under `.local/notebooks/failed-attempts/`. Completed feature checkpoints remain independently sealed. Corruption and changed scientific bindings fail closed; they are not treated as reasons to silently recompute.
