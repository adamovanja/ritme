# Validation leakage: feature engineering runs on the full train_val before any internal split

## Problem

ritme's feature-engineering pipeline runs on the full `train_val` before any
internal train/validation split. Cross-sample statistics computed in this
step (feature selection thresholds, ALR denominator, rank transform) include
the rows the trainable later holds out as validation. The validation set in
every trial is therefore not truly held out from the engineering pipeline.

The held-out **test set** is unaffected: `split_train_test` separates `test`
from `train_val` before `find_best_model_config` is called, and engineering
only sees `train_val`. The issue is strictly about the internal
train/validation boundary inside each trial.

## Where it happens

Both code paths in `ritme/feature_space/_process_train.py` call
`_engineer_features` on the full `train_val` before any split:

- `process_train` -- single 80/20 split via `_split_data_grouped`.
- `process_train_kfold` -- K fold index-pairs via `_make_kfold_splitter`.

Per-step audit of `_engineer_features`:

| Engineering step | Leaks? | Cross-sample stat |
|---|---|---|
| `aggregate_microbial_features` (taxonomy-driven) | no | none |
| `select_microbial_features` (`*_threshold`, `*_quantile`, `*_topi`, `*_ith`) | **yes** | thresholds / quantiles / per-feature ranks |
| `transform_microbial_features` -> CLR, ILR, pa | no | per-sample only |
| `transform_microbial_features` -> ALR (`_find_most_nonzero_feature_idx`) | **yes** | denominator feature index |
| `transform_microbial_features` -> rank | **yes** | per-feature sort order |
| `enrich_features` | no | metadata-driven |

## Impact

1. **Validation scores are optimistic.** Models are scored on features chosen
   and transformed using information about the validation rows.
2. **Inter-fold correlation undermines the one-standard-error rule** in the
   K-fold path. Folds share engineering parameters, so `std / sqrt(K)`
   underestimates the true standard error; the 1-SE tolerance band defined in
   `evaluate_models._select_best_with_one_se` is narrower than the noise
   warrants.

## Required change

Split `_engineer_features` into explicit fit and transform phases:

```text
fit_engineering(train_slice, tax, config) -> learned_params
apply_engineering(slice, tax, learned_params) -> engineered_matrix
```

Per code path:

- `process_train`: split raw `train_val` first, fit engineering on the train
  slice, apply to both train and val slices.
- `process_train_kfold`: per fold, fit on `train_val.iloc[tr_idx]`, apply to
  both the train and val slices of that fold.
- Deployable refit: fit on full `train_val`, apply to full `train_val`
  (matches the `train_val -> test` contract `TunedModel` honors at
  prediction time).

Engineering steps that need fit/transform separation:

- `select_microbial_features` -- capture the chosen feature ids; reapply
  (filter, no recompute) on the val slice.
- `_find_most_nonzero_feature_idx` -- capture the denominator index; reuse
  on the val slice.
- `transform_microbial_features` -> rank -- capture per-feature
  sorted-value arrays; rank the val slice against those.

`TunedModel.select` / `.transform` / `.enrich` already implement the
apply-frozen-parameters pattern at the `train_val -> test` boundary; lift
that logic into a shared helper.

## Tests required

Per engineering step with a fit phase:

- "fit then apply on the same slice" matches the current `_engineer_features`
  output on that slice.
- "fit on train slice then apply to val slice" matches a hand-constructed
  reference where the val slice is transformed with the train-fitted
  parameters.

Integration test: `process_train_kfold` per-fold val output equals what an
external fit-on-train + apply-on-val would produce.

## Cost

- Engineering runs more than once per trial: K times for the K-fold path,
  twice for the single-split path (fit on train slice + fit on full
  `train_val` for the deployable refit). Sub-2x trial wall time for RF / XGB
  / NN; 2-4x for linreg / trac where engineering is a larger fraction of
  trial wall time.
- ~80-120 lines in `ritme/feature_space/_process_train.py`, plus a shared
  frozen-params container.

## Effort

Medium. Risk is in subtle differences between cross-sample statistics
computed on full data vs on a train slice -- especially the rank transform,
whose outputs depend on the sorted-value arrays.
