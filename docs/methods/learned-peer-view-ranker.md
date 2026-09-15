# Learned peer and view ranker

This development stage tests whether a small, bounded scoring head can improve odd-one-out ranking without changing the established vision pipeline.

The upstream pipeline remains:

1. LocateAnything proposes object boxes.
2. SAM 2.1 produces object masks.
3. DINOv2-small produces object features.
4. The current Hungarian method links candidates across views.
5. A scorer ranks the linked objects.

Notebook 08L reads the sealed Notebook 08G predictions and prepares label-free relative features for fixed one, two, three and five-view settings. It writes one verified checkpoint per scene and seals the complete feature package before labels are available.

Notebook 08M verifies the 08L lock, opens the development evaluator package, and compares fixed baselines, robust score fusion, bounded winner clustering and three small linear learning-to-rank objectives. All reported learned results are grouped out-of-fold results by source asset. The notebook also produces risk-coverage diagnostics and qualitative prediction evidence. It does not use the calibration or internal-lock roles.

The primary decision is based on the automatic route with the fixed three-view set `V00, V07, V14`. Other view sets are sensitivity checks. Scientific metrics average over equal-score orderings so candidate IDs cannot increase accuracy. The selection uses a one-standard-error development heuristic: keep the simplest method close to the best grouped-fold tie-expected top-1 result, while also requiring tie-expected average precision and recall at two to remain within one baseline standard error. Asset-group bootstrap intervals and the complete metric table remain the main evidence; this is not a universal accuracy threshold or a significance test.

## Local validation

Use the existing scoring-comparison environment:

```bash
SCORING_ROOT="../scoring-comparison"
"$SCORING_ROOT/.venv/bin/python" -m pytest -q tests
```

Run the two stages in order:

```bash
PYTHONPATH="src:../identity-association/src" \
  ../scoring-comparison/.venv/bin/python scripts/run_metric_feature_preparation.py \
  --scoring-root ../scoring-comparison/.local/kaggle-runs/08g-screen100-v2/scoring-comparison-08g \
  --output .local/08l-features

PYTHONPATH="src:../identity-association/src" \
  ../scoring-comparison/.venv/bin/python scripts/run_metric_evaluation.py \
  --feature-root .local/08l-features \
  --scoring-root ../scoring-comparison/.local/kaggle-runs/08g-screen100-v2/scoring-comparison-08g \
  --evaluator-root ../scoring-comparison/.local/inputs-v1/evaluator-only \
  --model-input-root ../scoring-comparison/.local/inputs-v1/model-input \
  --output .local/08m-evaluation
```

No file in this directory is committed automatically. Kaggle source packages, executed notebooks, private galleries and transfer archives remain under `.local/` until they are reviewed for a later pull request.
