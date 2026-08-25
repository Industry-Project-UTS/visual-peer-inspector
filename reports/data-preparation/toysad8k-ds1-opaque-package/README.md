# ToysAD-8K DS1 opaque package

The `kaggle-v3` folder contains the safe evidence from Notebook 03.1. The
notebook verified the published, label-free input package for the 100-scene
DS1 development screen.

The package contains 2,000 images, with 20 views for each scene. Every image
decoded at 256 by 256 pixels. Scene membership, file paths and content hashes
matched the frozen cohort contract. A second pass reused all 2,000 files and
produced the same hashes.

ClearML task `2339ff3dc5e6484292d71cd180c45110` stores the aggregate tables,
figures and run identity.

Raw images, protected identifiers, labels, masks and the private contact sheet
are not stored in this repository.
