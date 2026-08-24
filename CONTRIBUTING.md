# Contribution guide

## Before starting

- Start from the latest clean `main` branch.
- Use a short task-based branch name such as `representation-screen`.
- Keep unrelated local work, private data, and transfer packages outside the
  branch.

## Implementation rules

- Put reusable code in `src/` and verification in `tests/`.
- Keep notebooks focused on configuration, execution, evidence, and conclusions.
- Use clear comments only where they help explain a decision or a non-obvious
  operation.
- Keep route names, scene roles, metrics, and output schemas stable across
  notebooks.
- Record configuration, dependency versions, hashes, runtime, failures, and
  acceptance gates for each experiment.

## Data and credentials

Do not commit raw datasets, private labels, crosswalks, credentials, model
weights, feature caches, bulk predictions, Kaggle logs, ClearML configuration,
or transfer ZIP files. Commit only reviewed aggregate evidence and approved
diagnostic figures.

ClearML tracking must not block model execution. Local tests and offline checks
must remain usable without ClearML credentials.

## Validation

Before opening a pull request:

1. run the relevant unit and integration tests
2. run `git diff --check`
3. confirm that Markdown links resolve
4. inspect `git status --short` for unintended files
5. verify that no credential or protected label entered the diff

Use a short human commit message and a pull request description that states the
purpose, evidence, checks, and known limits. Do not add automated co-author
lines.
