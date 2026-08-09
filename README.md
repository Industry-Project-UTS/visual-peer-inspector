# Visual Peer Inspector

Visual Peer Inspector compares similar objects in one scene and ranks the
objects that look different from their peers. The first experiment uses
ToysAD-8K, supplied object masks and frozen DINOv2 features.

## Current scope

- One fixed camera view
- Supplied object segmentations
- Frozen DINOv2 ViT-S/14 features
- B0 peer-mean anomaly score
- A fixed 40-scene training cohort
- No use of object order or filenames as predictive inputs

The current ToysAD-8K release contains one-anomaly and two-anomaly scenes. It
does not provide all-normal scenes in the inspected scene catalogue.

## Repository layout

```text
data/manifests/       Dataset and cohort records
docs/                 Problem, method and experiment notes
scripts/              Verification, extraction and B0 commands
src/                  Reusable project code
tests/                Small unit and integration tests
```

## Run the checks

```bash
python3 -m unittest discover -s tests -v
```

Verify the downloaded chunks:

```bash
python3 scripts/verify_chunks.py /path/to/ToysAD-8K/chunks
```

Extract only the fixed development cohort:

```bash
python3 scripts/extract_cohort.py \
  /path/to/ToysAD-8K/chunks \
  data/local/development-40
```

Run B0 after installing the optional baseline dependencies:

```bash
python3 -m pip install -e '.[baseline]'

python3 scripts/run_b0.py \
  data/local/development-40 \
  outputs/b0-development.json
```

Add `--local-files-only` when the pinned DINOv2 model is already cached.

## Documents

- [Problem and proposed method](docs/problem-and-method.md)
- [Dataset checks](docs/dataset.md)
- [B0 development result](docs/experiments/b0-development.md)

## Data boundary

Raw datasets, extracted scenes, model files, credentials and private university
documents are not committed. The tracked manifests record the exact dataset
revision and development cohort.

## Team

- Sreekar Reddy
- Vaibhav Bairathi

UTS 32040 Industry Project, Spring 2026.
