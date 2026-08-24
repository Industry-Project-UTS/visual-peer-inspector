# ToysAD-8K dataset and split protocol

## Source

- Hugging Face repository: `ankankbhunia/odd-one-out`
- Pinned revision: `ad98caffa9d136bfaec49d847e5c9eec44ffe30c`
- Archive chunks: `toysAD8K_part_aa` to `toysAD8K_part_ak`
- Total compressed size: 52,993,751,003 bytes

The 11 files are consecutive parts of one ZIP archive. They are not separate
dataset subsets. The extraction tools read them as one logical archive and do
not create a second full-size ZIP.

## Dataset audit

The full-dataset EDA and integrity checks found:

| Check | Result |
| --- | ---: |
| Catalogue scene metadata records | 7,396 |
| Availability-ready scenes | 7,373 |
| Eligible scenes after the selection rules | 4,107 |
| Zero-byte files | 11 |
| Scenes with one anomaly | 6,310 |
| Scenes with two anomalies | 1,086 |
| All-normal scenes | 0 |
| Label and asset-path disagreements | 0 |

`scene2d.metadata.json` uses `0` for anomaly and `1` for normal. The loader also
checks the asset path in `scene3d.metadata.json`. Every audited scene places
anomalous objects first in metadata order. Object order and source filenames
must therefore stay outside the predictive path.

## Primary 2,000-scene protocol

The project uses a deterministic, leakage-safe selection from the 4,107
eligible scenes. Selection is performed at scene level so that all views of one
scene remain in one role.

| Role | Scenes | Use |
| --- | ---: | --- |
| Development | 1,000 | Method development, ablations, and error analysis |
| Calibration | 600 | Thresholds and operating-point choices |
| Internal lock | 400 | Final internal confirmation after method freeze |

Private role labels and the opaque-to-source crosswalk are not committed. The
released model-facing package contains opaque scene identifiers and selected
RGB views without protected anomaly labels.

The project starts with nested gates before expanding to the full role:

- DS0: 20 scenes with 20 views each for operational and end-to-end checks
- DS1: 100 scenes for the first method and encoder screen
- DS2: a separate 300-scene confirmation after the DS1 choices are frozen

The full eligible pool remains useful for later robustness work, but it should
not replace the controlled 2,000-scene experiment or expose the locked scenes
during development.

## Multi-view unit

The scene is the unit of assignment and evaluation. Each selected scene can
provide up to 20 RGB views. Per-view predictions are kept separate until the
identity and fusion stage, which prevents a strong or weak camera view from
being hidden by early averaging.

## Historical 40-scene baseline

The repository still contains a deterministic 40-scene, one-view development
cohort used for the first B0 and B1 baseline. It is retained for reproducibility
and should not be described as the current primary experiment.

To reproduce that historical extraction:

```bash
python3 scripts/verify_chunks.py /path/to/ToysAD-8K/chunks
python3 scripts/extract_cohort.py \
  /path/to/ToysAD-8K/chunks \
  data/local/development-40
```

The extraction reads view `0000`, the two metadata files, and the required
object masks. It validates ZIP CRC values and records size and SHA-256 evidence.

## Data zones and leakage controls

1. Trusted preparation may read source identifiers and protected labels.
2. Model-facing jobs receive only the inputs required by the active route.
3. Evaluator-only jobs join predictions to protected boxes, masks, and labels.
4. Public or tracked evidence contains reviewed aggregate results and safe
   diagnostic figures only.

The inspected release contains no all-normal scenes. Results from this dataset
cannot establish false-positive behaviour for an all-normal user upload.

## Verify the source archive

Full verification checks each SHA-256 value:

```bash
python3 scripts/verify_chunks.py /path/to/ToysAD-8K/chunks
```

When hashes were already checked, a quick size-only check is available:

```bash
python3 scripts/verify_chunks.py \
  /path/to/ToysAD-8K/chunks \
  --sizes-only
```

Expected names, sizes, and hashes are stored in
`data/manifests/toysad_chunks.json`.

## Licence and sources

The official repository provides the dataset under Creative Commons
Attribution-NonCommercial-ShareAlike 4.0. Derived figures must retain
attribution and the applicable licence notice. Source asset licences must also
be checked before any public dataset upload.

- [Official dataset instructions](https://github.com/VICO-UoE/OddOneOutAD/blob/main/DATASET.md)
- [ToysAD-8K files on Hugging Face](https://huggingface.co/datasets/ankankbhunia/odd-one-out)
- [Notebook sequence](../notebooks/README.md)
