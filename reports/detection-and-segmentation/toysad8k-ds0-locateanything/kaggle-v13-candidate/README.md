# Kaggle Version 13 full candidate run

The full run used all 400 images from the fixed 20-scene DS0 cohort.

- 400 of 400 jobs completed successfully
- no failed or truncated jobs
- no zero-proposal images
- no invalid or duplicate boxes
- 20 scenes and all 20 stored views were covered
- 1,616 boxes were produced, with a mean of 4.04 proposals per image
- mean inference time was 0.436 seconds per image
- the 20-image repeat agreement was 1.0
- all 26 recorded artifacts passed the independent checksum verifier
- ClearML task `f08bcfe9027847d29067c41f71002203` completed

The normal diagnostic examples show object-level boxes around the visible toys.
Some high-count outliers also include background regions. Proposal accuracy is
therefore still pending the evaluator-only reference-box comparison.

This is candidate evidence because the recorded source snapshot was dirty. The
record is kept unchanged for traceability.
