# ToysAD-8K dataset checks

## Source

- Hugging Face repository: `ankankbhunia/odd-one-out`
- Pinned revision: `ad98caffa9d136bfaec49d847e5c9eec44ffe30c`
- Archive chunks: `toysAD8K_part_aa` to `toysAD8K_part_ak`
- Total compressed size: 52,993,751,003 bytes

The 11 files are consecutive parts of one ZIP archive. They are not separate
dataset subsets. The extraction code reads them as one logical file and does
not create a second 53 GB ZIP.

## Verify the download

Full verification checks each SHA-256 value:

```bash
python3 scripts/verify_chunks.py /path/to/ToysAD-8K/chunks
```

A quick size check is available when the files were already hashed:

```bash
python3 scripts/verify_chunks.py \
  /path/to/ToysAD-8K/chunks \
  --sizes-only
```

The expected names, sizes and hashes are stored in
`data/manifests/toysad_chunks.json`.

## Extract the development cohort

```bash
python3 scripts/extract_cohort.py \
  /path/to/ToysAD-8K/chunks \
  data/local/development-40
```

Only view `0000`, the two metadata files and the required object segmentation
masks are extracted for each selected scene. Every file is read through the ZIP
CRC check. The extraction report also records its size and SHA-256 value.

## Label checks

`scene2d.metadata.json` uses `0` for anomaly and `1` for normal. The loader also
checks the object asset path in `scene3d.metadata.json`. It stops if the label,
path or object order disagrees.

The completed catalogue audit found:

- 7,396 usable scene metadata records
- 6,310 scenes with one anomaly
- 1,086 scenes with two anomalies
- no all-normal scenes
- no label and asset-path disagreements

Every inspected scene places anomalous objects first in metadata order. The
experiments therefore test score invariance after permuting the feature order.

## Licence

The official repository provides the dataset under Creative Commons
Attribution-NonCommercial-ShareAlike 4.0. Derived figures must retain
attribution and the applicable licence notice. The original source asset
licences should also be checked before a public dataset upload.

## Sources

- [Official dataset instructions](https://github.com/VICO-UoE/OddOneOutAD/blob/main/DATASET.md)
- [ToysAD-8K files on Hugging Face](https://huggingface.co/datasets/ankankbhunia/odd-one-out)
