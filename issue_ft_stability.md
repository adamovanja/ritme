# `explain_stability`: reference trial can differ from the deployed model

## Problem

`explain_stability()` selects its reference trial by re-applying the deployment
rule to a candidate band that is **truncated to the `max_trials` (default 15)
best trials by mean** (`select_band_trials` → `band.head(max_trials)` after
sorting by mean, then `select_reference_run_id`). `find_best_model_config`
applies the same 1-SE rule to the **untruncated** pool and deploys the simplest
in-band trial, which is often not among the top 15 by mean. The two can
therefore pick different trials, so the stability figure is anchored on a trial
that is not the deployed model. The `select_reference_run_id` docstring already
acknowledges this.

## Observed (manuscript use-case runs)

| Run | stability reference | deployed 1-SE winner (full `mlflow_logs.csv`) | same? |
|---|---|---|---|
| u1 (`xgb`) | `bbc521c6`, 2'438 features, CV 2.260 | `65ff37a1`, 215 features, CV 2.375 | no |
| u2 (`linreg`) | `86761fd6`, 1'981 features | `86761fd6` | yes (coincidence: deployed trial is within the top 15 by mean) |
| u3 (`logreg`) | `16f9a4f2`, 1'134 features, `shannon_and_metadata` | `ec282d09`, 1'132 features, `metadata_only` | no |

Reproduce the deployed winner with
`evaluate_models._select_best_with_one_se` over the experiment's full
`mlflow_logs.csv` (restricted to `filter_reproducible_trials`) and compare its
`run_id` to `is_reference` in `stability_trials.csv`.

## Fix

1. In `explain_stability()`, select the reference from the **untruncated**
   candidate pool (do not apply `max_trials` before `select_reference_run_id`).
   The reference then equals the deployed winner by construction.
2. When truncating the symmetric plotted band to `max_trials`, **always retain
   the anchor/reference row**. Currently `band.head(max_trials)` keeps the best
   performers and can drop the anchor (the deployed winner sits at the low end
   of the band, e.g. u1: anchor mean 2.375 vs top-15 range 2.255–2.275).

## Tests

- Reference `run_id` equals the `_select_best_with_one_se` winner over the full
  trial records, also when more than `max_trials` trials lie in the 1-SE band.
- Truncated symmetric band always contains the reference row
  (`is_reference` present in the manifest), with at most `max_trials` rows.
