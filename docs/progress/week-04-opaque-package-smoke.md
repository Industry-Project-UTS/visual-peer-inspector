# Week 4 opaque package smoke

Notebook 03 prepared and checked the model-facing package for the first pipeline
gate. The package contains 20 selected scenes with 20 views per scene, for 400
images in total.

The notebook checks the package structure, image count, dimensions, decoding,
duplicate paths, checksums, scene completeness, view coverage, and restart
behaviour. It does not use anomaly labels, source identifiers, reference boxes,
or masks in the model-facing path.

Kaggle Version 5 completed successfully and ClearML recorded the run. All eight
validation gates passed. The second check reused all 400 files and produced the
same hashes.

The saved report contains safe aggregate tables and charts only. Raw images and
the private contact sheet remain outside GitHub.

The run is kept as candidate evidence because its source snapshot was marked
dirty. The next clean-snapshot run should use the committed Notebook 03 source
and replace the candidate evidence without changing the package membership.
