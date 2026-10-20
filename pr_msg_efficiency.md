ENH: More compute-efficient model search

## Summary

This PR makes `fit()` far more compute-efficient: the same allocation runs
many more trials (linreg: ~20 → ~300 per 180 s sweep on the example
dataset), weak trials are pruned after two folds instead of running to
completion, and a broken search aborts within minutes instead of after the
full time budget. It fixes five throughput sinks: serialized trial launches
(~7.5 s each, capping throughput near 400 trials/h regardless of
allocation), CPU reservations that model families cannot occupy, an ASHA
first rung above every reachable `training_iteration`, unbounded per-trial
cost, and fatal search conditions (all trials erroring, metric never
reported) that went undetected until the budget ran out.

- **Launch path.** `reuse_actors=True`; `TUNE_MAX_PENDING_TRIALS_PG` is set
  so actor bootstrap overlaps across trials. New optional config key
  `max_pending_trials` (default: one pending trial per concurrency slot; an
  externally set env var wins). The nn trainables move to
  `model_space/nn_trainables.py`, so workers running non-nn trials do not
  import lightning / torchmetrics / torchvision / coral_pytorch (a subprocess
  test pins the import footprint; bare `torch` remains via skbio).
- **Reachable scheduler rungs.** In K-fold mode a trial reports once per
  fold, so ASHA's rungs are derived from the fold count: `grace_period=2`,
  `max_t=k_folds`. New optional config keys `scheduler_grace_period` /
  `scheduler_max_t` override them (set `scheduler_grace_period >= k_folds`
  to disable K-fold pruning). The parallel sklearn / trac K-fold paths emit
  running fold aggregates so every family is prunable; a warning prints when
  a configuration cannot prune. `fully_reproducible: true` keeps the
  previous rungs.
- **Per-trial wall-clock cap.** New optional config key
  `max_trial_duration_s` (default: off). K-fold trials stop at the next fold
  boundary and report a `time_capped` partial aggregate that best-model
  selection excludes; single-split xgb / nn trials stop at the next reported
  iteration / epoch and keep their best checkpoint.
- **Right-sized reservations.** Fold-parallel families (linreg, logreg,
  trac) reserve one CPU per fold; trac's reservation is sized from the
  configured concurrency rather than its memory-reduced slot count. ritme
  prints right-sizing guidance when a reservation cannot fill the
  allocation.
- **Parallel feature engineering.** `process_train_kfold` runs its `2K + 1`
  engineering passes as node-pinned Ray tasks throttled to the trial's
  reserved CPUs (bit-identical to the serial path, pinned by a real-Ray
  equality test).
- **Early failure detection.** A search health guard (stopper + callback)
  aborts when ≥ 10 trials have errored at ≥ 50 % error rate (the
  `max_trial_failure_rate` policy still applies after `fit()`), or when
  ≥ 10 trials completed without a finite tuning metric, raising the same
  "No best trial found" error as the retrieve step. Degenerate
  `max_trial_duration_s` / `max_pending_trials` values are rejected at
  config time, and no-deployable-trial errors distinguish "stopped early by
  cap / scheduler / budget" from crashes.

Bug fixes, each with a regression test:

- Under `reuse_actors=True`, Ray 2.55 returns an unsynced scratch directory
  from `get_trial_dir()` on reused actors, so `model.pkl` / `taxonomy.pkl` /
  `label_encoder.pkl` never reached `Result.path`. Trainables now write
  artifacts to the storage `trial_fs_path` (`_trial_artifact_dir`), the path
  retrieval reads.
- `retrieve_n_init_best_models` selected dual-task `nn_class` by the
  classification metric even in regression sweeps. Selection now follows the
  run's `task_type`.
- `evaluate_mlflow.create_color_map` used `matplotlib.cm.get_cmap`, removed
  in matplotlib 3.9.

## Comparison (example dataset, identical config, 180 s budget per family)

| | before | this PR |
|---|---|---|
| linreg trials per 180 s sweep | ~20 | **~300** |
| scheduler pruning in K-fold mode | never | at the fold-2 running mean |
| broken search (all-NaN metric) detected after | full `time_budget_s` × remaining families | ~10 completed trials |
| artifacts at `Result.path` under actor reuse | first trial only | every trial |

## Verification

Full unit suite (615 tests); MLflow regression + classification, snapshot
(multi-CPU, exercising the engineering fan-out), WandB, and both
example-notebook smokes end-to-end.

Config schema: four new optional keys, documented in the README. No changes
to metric definitions, checkpoint retrieval, or the MLflow/WandB wiring.
