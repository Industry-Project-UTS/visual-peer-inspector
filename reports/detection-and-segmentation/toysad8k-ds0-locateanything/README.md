# ToysAD-8K LocateAnything DS0 evidence

This report records the Notebook 04 operational checks for LocateAnything-3B
on the fixed 20-scene DS0 cohort. The cohort contains 20 views per scene, so
the full run covers 400 images.

`kaggle-v11-pilot/` contains the successful 20-image pilot and repeat check.
`kaggle-v13-candidate/` contains the completed 400-image run. Both folders keep
only aggregate tables, charts, hashes, configuration details, and run records.

The Version 13 run completed all 400 images with no failures, truncated
responses, invalid boxes, duplicate boxes, or zero-proposal images. The exact
20-image repeat check matched in every case.

This evidence confirms that the frozen model can run reliably on the selected
cohort. It does not prove that every proposal is correct. Reference boxes remain
outside the model-facing notebook and will be used later in the evaluator-only
proposal quality notebook.

The Version 13 folder is called `candidate` because the run record reports a
verified pre-commit source snapshot with `git_dirty: true`. That record must not
be changed. A clean-snapshot rerun can replace the candidate evidence after the
workflow is committed.
