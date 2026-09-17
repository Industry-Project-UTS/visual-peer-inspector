# Development 1,000-scene multi-view evaluation

The fixed pipeline completed on all 1,000 development scenes grouped into 273 asset groups. The result is exposure-qualified because all asset groups are potentially exposed; it is not an independent confirmation set.

## Decision

- Keep `winner_cluster_consensus` as the automatic application route. Its five-view tie-aware top-1 result is 22.24%, so the report and app must present its limitations clearly.
- Keep `object_bank_top2` as the assisted route. With dataset-supplied correspondence it reaches 62.42% tie-aware top-1 for five views, with a 95% asset-group interval of 59.57% to 65.16%.
- Do not reopen backbone, scorer, or association-model search. Freeze these functions and proceed to calibration, confidence policy, internal lock, application integration, and official evaluation.
- Support both modes in the app: automatic association and optional user confirmation/correction of object correspondence.

## Evidence integrity

The downloaded evaluation was checked against a 2,009-file evaluation lock, a 55-file report lock, 1,000 scene locks, seven gallery locks, and the Kaggle completion marker. All hashes matched. The final executed notebook has no error output.

ClearML publication is pending because the Kaggle tracking runtime raised `MissingConfigError`. Publication must be retried from the sealed aggregate allowlist without rerunning the evaluation.

Repository-safe tables, figures, bindings, receipts, and the report summary are under `reports/remaining-cohort-workbench/development1000/`. Private scene-level evidence remains outside GitHub.
