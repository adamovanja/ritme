# Extend K-fold CV + one-standard-error rule to iterative trainables

## Problem

The K-fold path covers only the sklearn-style single-shot trainables. The
iterative trainables (`train_xgb`, `train_xgb_class`, `train_nn_reg`,
`train_nn_class`, `train_nn_corn`) accept `k_folds` for signature parity but
ignore it: each trial uses a single 80/20 split, reports a single
`<metric>_val` per trial, and `_select_best_with_one_se` falls back to
`ResultGrid.get_best_result` because no `<metric>_se` field is present.

## Required behavior

For `k_folds > 1`, each iterative trainable must:

1. Run K folds via `process_train_kfold`.
2. Aggregate per-fold final validation scores via `_aggregate_fold_metrics`
   so the trial reports `<metric>_mean`, `<metric>_std`, `<metric>_se`, and
   `n_folds` for every metric the single-split path emits.
3. Refit one model on the full design matrix and save that as the deployable
   checkpoint.
4. Single-split path (`k_folds <= 1`) stays unchanged.

`_select_best_with_one_se` already covers iterative trainables --
`_MODEL_SIMPLICITY_KNOBS` in `evaluate_models.py:452-489` already has `xgb`,
`xgb_class`, `nn_reg`, `nn_class`, `nn_corn` entries.

## Approach: sequential per-fold training, one aggregated `tune.report` at end

Run K iterative-training calls back to back inside the trainable. After fold
K, aggregate via `_aggregate_fold_metrics` and emit one
`tune.report(metrics=aggregated)`. After the K folds, run one more training
call on the full design matrix to produce the deployable checkpoint.

Alternatives considered and rejected:

- **Parallel folds via joblib.** `tune.report` from a worker subprocess does
  not flow back to the Ray Tune actor; K models live in memory simultaneously;
  K Lightning Trainers serialise on a single GPU.
- **Aggregated per-iteration callback (lockstep folds).** Intrusive in
  `xgb.train`'s internal loop and Lightning's `Trainer`; fragile when folds
  diverge in early-stopping behaviour.

## Code-level changes

### `train_xgb` and `train_xgb_class` (`ritme/model_space/static_trainables.py:1498`, `:1596`)

For `k_folds > 1`:

- Call `process_train_kfold` to get fold splits and the full design matrix.
- Loop sequentially: build `xgb.DMatrix` for train and val, call
  `xgb.train(..., early_stopping_rounds=max(10, int(0.1 * n_estimators)))`
  *without* a checkpoint callback, read each fold's `booster.best_score` and
  `booster.best_iteration`, recompute the standard metric set on
  `(X_tr, y_tr)` and `(X_va, y_va)` at the best iteration.
- Refit on `X_full` with `num_boost_round = median(per_fold_best_iteration)`
  (fallback `config["n_estimators"]` when any fold's `best_iteration` is
  `None`). Save the refit booster.
- `_save_taxonomy` and the label-encoder write (`train_xgb_class`) happen
  once at trial end.

`_RitmeXGBCheckpointCallback` stays scoped to the single-split path.

### `train_nn` (`:1126`) and the three wrappers

For `k_folds > 1`:

- Call `process_train_kfold`.
- Loop sequentially: build per-fold `train_loader` / `val_loader` via
  `load_data`, build a fresh `NeuralNet`, `Trainer.fit(...)` with
  `EarlyStopping(monitor="val_loss", ...)` and no
  `NNTuneReportCheckpointCallback`. After `fit`, call
  `trainer.validate(model, val_loader)` to read the standard metric set
  emitted by `NeuralNet.on_validation_epoch_end`.
- Refit on `X_full` for the deployable checkpoint. Save the final-epoch state
  (no held-out set under K-fold to rank epochs against).
- `_save_taxonomy` once at trial end. `_save_label_encoder` once at trial end
  for classification and ordinal regression.

`NNTuneReportCheckpointCallback` stays scoped to the single-split path.

## Resource allocation

Sequential folds; all `cpus_per_trial` go to each fold's training call.

- **xgb / xgb_class**: `config["nthread"] = cpus_per_trial`,
  `config["device"] = "cuda"` when `gpus_per_trial > 0`. No call to
  `_allocate_fold_resources` in the K-fold branch.
- **nn**: `torch.set_num_threads(cpus_per_trial)`,
  `num_workers = max(0, cpus_per_trial - 1)` per fold's DataLoader. GPU is
  implicit via Ray Tune's `CUDA_VISIBLE_DEVICES`.

## Effect on ASHA pruning

ASHA receives one `tune.report` per trial (the aggregate) instead of one per
iteration. Trial-level pruning at fold boundaries is the only remaining
mechanism. Internal early-stopping inside each fold still terminates hopeless
folds (`early_stopping_rounds` for xgb, `EarlyStopping` for nn), so the loss
is bounded.

Optional follow-up if profiling shows wall-time regression: emit a mid-trial
`tune.report` after each fold completes (running mean over folds completed so
far) so ASHA can prune before all K folds finish. Not in scope here.

## Tests

For each of `train_xgb`, `train_xgb_class`, `train_nn_reg`, `train_nn_class`,
`train_nn_corn`, add to `ritme/tests/test_model_static_trainables.py`:

1. **Single-split parity**: `k_folds=1` produces the existing metric keys and
   approximately the existing numeric values.
2. **K-fold metric reporting**: `k_folds=3` produces `<metric>_mean`,
   `<metric>_std`, `<metric>_se`, and `n_folds == 3` for every metric the
   single-split path emits.
3. **Deployable checkpoint**: with `k_folds=3` the saved model loads cleanly
   via `TunedModel` and predicts with the correct shape on a small holdout.

Extend `test_evaluate_models.py` so a `ResultGrid` containing
iterative-trainable trials with `<metric>_se` set is selected via the 1-SE
rule.

## Effort

- **xgb / xgb_class**: ~150 lines in `static_trainables.py`, ~80 lines of
  tests. Open question during implementation: refit `num_boost_round`
  (median best-iteration vs `n_estimators`).
- **nn**: ~150 lines in `static_trainables.py`, ~100 lines of tests. Open
  question: full-refit final-epoch vs best-epoch state.

## Out of scope

- Validation leakage in feature engineering (see `issue_val_leakage.md`).
- Re-tuning scheduler defaults (`DEFAULT_SCHEDULER_GRACE_PERIOD`,
  `DEFAULT_SCHEDULER_MAX_T`).
- Parallel folds via joblib for iterative trainables.
