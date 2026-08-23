# Selection Audit Results

These files were produced by the completed Kaggle Version 4 run of the ToysAD-8K 2,000-scene selection audit.

The audit passed all gates. It confirmed:

- 4,107 eligible scenes
- 2,000 selected scenes
- 545 selected base assets
- 238,000 indexed source paths
- no official test scenes in the selected corpus
- no base asset shared across experiment roles
- a byte-identical repeat build

The CSV files contain the reviewed tables. The PNG files contain the report figures. `selection_aggregate.json` contains the main result, `run_record.json` contains the execution record, and `artifact_manifest.csv` records the output hashes.

The corresponding ClearML task is `selection-audit-kaggle-v4`. It is completed and tagged `final`, `notebook-02`, and `kaggle-v4`.
