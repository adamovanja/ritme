# Efficiency fixes implementation plan (branch `improve_efficiency`)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline
> execution in this session). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve every issue in `issue_inefficient_large_trials.md` and
`issue_underutilization.md`: make ASHA pruning real, add a safe per-trial
wall-clock cap, unblock the serial trial-launch path, right-size per-family CPU
reservations, parallelise per-fold feature engineering, and abort broken
searches within minutes instead of after the full budget.

**Architecture:** All changes live in `ritme/tune_models.py`,
`ritme/model_space/static_trainables.py` (plus a new
`ritme/model_space/nn_trainables.py`), `ritme/feature_space/_process_train.py`,
`ritme/find_best_model_config.py`, and `ritme/evaluate_models.py`. New behavior
is config-key driven with backward-compatible defaults except where the issue
explicitly calls for a new default (scheduler rungs, actor reuse, pending
trials, per-family CPU caps).

**Tech stack:** Ray Tune 2.55.1 (verified in the `ritme` conda env: function
trainables implement `reset_config` so `reuse_actors=True` is safe;
`TUNE_MAX_PENDING_TRIALS_PG` env override exists in
`tune_controller._get_max_pending_trials`; `Stopper.stop_all()` is polled every
controller step; ASHA `reduction_factor=4`).

**Spec:** `issue_inefficient_large_trials.md`, `issue_underutilization.md`.

## Global constraints

- No git commits/pushes — file edits only (user rule).
- Test in the activated `ritme` conda env only.
- No imports inside function bodies; no try/except unless necessary; no nested
  closures as factory pattern (partial application is fine).
- Every new behavior gets unit tests in `ritme/tests/`; `ruff check` + `py.test`
  must pass; smoke tests per AGENTS.md afterwards (tune_models + feature_space +
  trainables are all touched → MLflow regression, MLflow classification, WandB,
  snapshot, example notebook).
- README config-table additions stay concise, no rationale transplants.

## Key design decisions (from the issue caveats)

1. **Time cap stops at fold boundaries, never via injected exceptions**
   (TPOT's `stopit` corrupted the heap inside XGBoost C code). Capped trials
   report their running fold aggregate with suffixed keys only (`*_mean/_std/_se`)
   plus `time_capped: True`, no bare metric keys, no checkpoint → they can never
   be selected as best (same contract as ASHA-pruned K-fold trials).
2. **`max_trial_duration_s` defaults to None (off)** — the issue shows the right
   value is dataset-dependent and the expensive branch can hold the winners; any
   finite default would bias some search. Config key documented in README.
3. **Early failure-rate abort is deliberately more conservative than the
   post-hoc policy**: it fires only on *systemic* failure
   (≥ `EARLY_ABORT_MIN_ERRORS = 10` errored trials AND error rate ≥
   `max(0.5, max_trial_failure_rate)` over finished trials). The exact
   user-configured `max_trial_failure_rate` continues to be enforced post-hoc
   by `_check_for_errors_in_trials` — with the guard, that raise now happens
   minutes in, with the identical message.
4. **No-metric abort**: if ≥ `EARLY_ABORT_MIN_COMPLETED = 10` trials completed
   and no report ever carried a finite runtime metric, stop the experiment and
   raise the same "No best trial found for the given metric: ..." message.
5. **trac keeps its /3 concurrency reduction** (the memory constraint is real
   and unimplemented) but CPU sizing now divides by the *original* concurrency
   and is capped by the family appetite, so the memory workaround no longer
   triples the reservation (issue 2b). ritme prints right-sizing guidance when
   the reservation cannot fill the allocation.
6. **Concurrency is never silently raised** above the user's
   `max_cuncurrent_trials` — guidance is printed instead (memory safety).
7. **torch import split via module split**, not in-function imports (user
   rule): all nn code moves to `ritme/model_space/nn_trainables.py`;
   `static_trainables.py` becomes torch-free. No compat re-exports in
   `static_trainables` (would defeat the purpose).

---

### Task 1: nn module split (underutil issue 1, import cost / worker RSS)

**Files:**
- Create: `ritme/model_space/nn_trainables.py`
- Modify: `ritme/model_space/static_trainables.py` (remove nn code + torch/
  lightning/torchmetrics/coral imports), `ritme/tune_models.py` (MODEL_TRAINABLES),
  `ritme/evaluate_models.py:21`, `ritme/explain_features.py:17` (NeuralNet import)
- Test: `ritme/tests/test_model_static_trainables.py` (patch-target renames for
  nn tests), `ritme/tests/test_evaluate_models.py:90` (patch target), new
  subprocess test asserting torch-free import.

**Moves to `nn_trainables.py`** (unchanged bodies): `NeuralNet`, `seed_worker`,
`load_data`, `NNTuneReportCheckpointCallback`, `_nn_build_n_units`,
`_nn_classes_from_y`, `_build_neural_net_for_kfold`, `_NN_REG_METRIC_MAP`,
`_NN_CLASS_METRIC_MAP`, `_nn_extract_fold_metrics`, `_extract_best_epoch`,
`_nn_refit_epochs`, `_save_nn_checkpoint`, `_run_kfold_nn`, `train_nn`,
`train_nn_reg`, `train_nn_class`, `train_nn_corn`.

**Stays in `static_trainables.py`** (torch-free, still needed by tune_models):
`_check_nn_corn_levels`, `DEFAULT_NN_CORN_MAX_LEVELS`; nn_trainables imports
them plus `_aggregate_fold_metrics`, `_emit_running_fold_aggregate`,
`_save_label_encoder`, `_save_taxonomy`, `_classification_metrics_dict` from
`static_trainables` (import direction: nn → static, never reverse).

- [ ] Write failing subprocess test in `test_model_static_trainables.py`:

```python
def test_static_trainables_imports_without_torch(self):
    # Non-nn Ray workers must not pay the torch/lightning import cost
    # (issue: serial launch path); guard the module split.
    code = (
        "import sys; import ritme.model_space.static_trainables; "
        "mods = [m for m in ('torch', 'lightning', 'torchmetrics', "
        "'coral_pytorch') if m in sys.modules]; "
        "sys.exit(1 if mods else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True)
    self.assertEqual(proc.returncode, 0, proc.stderr.decode())
```

- [ ] Create `nn_trainables.py`; strip moved code + now-unused imports from
      `static_trainables.py`.
- [ ] Update `MODEL_TRAINABLES` in tune_models (`nn_*` from `nn_trainables`),
      `evaluate_models` / `explain_features` NeuralNet imports.
- [ ] Rename test patch targets `ritme.model_space.static_trainables.<X>` →
      `...nn_trainables.<X>` for X in {seed_everything, process_train, load_data,
      NeuralNet, Trainer, tune.report, ray.tune.get_context, process_train_kfold,
      _save_taxonomy, ray.train.Checkpoint.from_directory} *only inside nn test
      classes*; same in `test_evaluate_models.py:90`.
- [ ] Run: `conda run -n ritme python -m pytest ritme/tests/test_model_static_trainables.py -x -q`

### Task 2: reachable ASHA rungs + config exposure + guard print (ineff. issue 1)

**Files:** `ritme/tune_models.py`, `ritme/find_best_model_config.py`,
`ritme/tests/test_tune_models.py`, `ritme/tests/test_find_best_model_config.py`.

- [ ] `_define_scheduler`: accept `Optional[int]` grace/max_t; derive when None:

```python
KFOLD_SCHEDULER_GRACE_PERIOD = 1  # prune after fold 1 (rungs 1 and 4 at eta=4)

if scheduler_grace_period is None:
    scheduler_grace_period = (
        KFOLD_SCHEDULER_GRACE_PERIOD if n_folds > 1 else DEFAULT_SCHEDULER_GRACE_PERIOD
    )
if scheduler_max_t is None:
    scheduler_max_t = n_folds if n_folds > 1 else DEFAULT_SCHEDULER_MAX_T
```

- [ ] `run_trials` defaults become None; `run_all_trials` gains + forwards both;
      `find_best_model_config` reads `config.get("scheduler_grace_period")` /
      `config.get("scheduler_max_t")`.
- [ ] Guard print in `run_trials`: models emitting a bounded report count
      (K-fold: k_folds; single-split sklearn/trac: 1) where
      `grace_period > bound` → print "ASHA cannot prune ..." guidance.
- [ ] Tests: derived defaults (k=5 → grace 1 / max_t 5; k=1 → 10/100), explicit
      override wins, forwarding from run_all_trials + config keys, guard print.

### Task 3: per-fold running reports for the parallel sklearn/trac K-fold paths

**Files:** `ritme/model_space/static_trainables.py`, `ritme/evaluate_models.py`,
tests for both.

- [ ] `_dispatch_folds_then_refit(..., report_running=None)`: after each
      completed fold result is collected, call `report_running(fold_results)`.
      Module-level reporter (no closures):

```python
def _emit_running_aggregate_for_completed(fold_results, nb_features, n_splits):
    completed = [r for r in fold_results if r is not None]
    _emit_running_fold_aggregate(completed, nb_features, len(completed) - 1, n_splits)
```

      passed via `functools.partial(_emit_running_aggregate_for_completed,
      nb_features=..., n_splits=...)` from `_dispatch_kfold_and_refit_sklearn`
      / `_dispatch_kfold_and_refit_trac` (new args threaded from
      `_run_kfold_sklearn` / `train_trac`).
- [ ] `evaluate_models._select_best_with_one_se`: non-checkpoint families now
      require `"model_path" in result.metrics` to be selection candidates
      (pruned/capped trials end on a running aggregate without it).
- [ ] `evaluate_models._drop_checkpointless_trials`: warn only when the bare
      `metric` key is present (final-report shape ⇒ genuine mid-training crash);
      skip silently when only `<metric>_mean` exists (pruned/capped row).
- [ ] Tests: running reports emitted as folds complete (mocked ray/tune.report);
      1-SE selection skips model_path-less results; checkpointless warning gating.

### Task 4: `_SearchHealthGuard` — early aborts (underutil issue 3)

**Files:** `ritme/tune_models.py`, `ritme/tests/test_tune_models.py`.

- [ ] Implement (Stopper + Callback in one object, passed as both
      `RunConfig(stop=guard)` and appended to `callbacks`):

```python
EARLY_ABORT_MIN_ERRORS = 10
EARLY_ABORT_MIN_COMPLETED = 10
EARLY_ABORT_FAILURE_RATE = 0.5

class _SearchHealthGuard(Stopper, tune.Callback):
    def __init__(self, runtime_metric, max_trial_failure_rate,
                 max_trial_duration_s=None):
        ...counters: _num_errors, _num_completed, _seen_finite_metric...

    def __call__(self, trial_id, result):          # per-trial time cap
        if self._max_trial_duration_s is None: ... track metric; return False
        return result.get("time_total_s", 0.0) >= self._max_trial_duration_s

    def on_trial_error(self, iteration, trials, trial, **info): _num_errors += 1
    def on_trial_complete(self, iteration, trials, trial, **info): _num_completed += 1

    def stop_all(self):
        # systemic-failure abort + no-finite-metric abort; print once on trigger
```

- [ ] `run_trials`: build guard with `runtime_metric`; after `fit()` +
      `_check_for_errors_in_trials`, raise the no-best-trial RuntimeError when
      `guard.stopped_for_missing_metric` (same first sentence as Ray's message,
      naming the bare metric).
- [ ] Tests: guard unit tests (healthy search never stops; all-error search
      stops at thresholds; NaN-metric search stops after N completions; time cap
      returns True per trial; rate uses max(user_rate, 0.5)); run_trials raises
      early no-metric error.

### Task 5: per-trial wall-clock cap (ineff. issue 2)

**Files:** `ritme/tune_models.py`, `ritme/find_best_model_config.py`,
`ritme/model_space/static_trainables.py`, `ritme/model_space/nn_trainables.py`,
tests.

- [ ] Config key `max_trial_duration_s` (default None) → `find_best_model_config`
      → `run_all_trials` → `run_trials` → guard (report-boundary stop) and
      trainables via `tune.with_parameters` (fold-boundary stop).
- [ ] `static_trainables`: `import time`; helpers

```python
def _strip_bare_metric_keys(metrics, keep=("n_folds", "nb_features", "time_capped")):
    suffixes = ("_mean", "_std", "_se")
    return {k: v for k, v in metrics.items() if k in keep or k.endswith(suffixes)}

def _time_cap_reached(trial_start, max_trial_duration_s):
    return (max_trial_duration_s is not None
            and time.monotonic() - trial_start >= max_trial_duration_s)

def _report_time_capped_aggregate(per_fold_metrics, nb_features):
    metrics = _aggregate_fold_metrics(per_fold_metrics)
    metrics["nb_features"] = nb_features
    metrics["time_capped"] = True
    tune.report(metrics=_strip_bare_metric_keys(metrics))
```

      (`_emit_running_fold_aggregate` reuses `_strip_bare_metric_keys`.)
- [ ] `_run_kfold_xgb` / `_run_kfold_nn`: `trial_start` at entry; check before
      each fold (only when ≥1 fold done) and before the refit → truncated report
      + return. All `train_*` signatures gain `max_trial_duration_s=None`.
- [ ] `_dispatch_folds_then_refit(..., deadline=None)`: stop submitting new
      folds past deadline, drain in-flight (no ray.cancel — no injected
      exceptions), skip refit → `(fold_results, None)`. `_finalize_and_report_sklearn`
      and trac's K-fold branch treat `full_model is None` as truncation →
      `_report_time_capped_aggregate` (+ completed folds only) and return.
- [ ] Single-split: cap enforced only via guard at report boundaries (xgb per
      iteration, nn per epoch); single-split sklearn documented as un-cappable.
- [ ] Tests: per-path truncation with patched `time.monotonic`; report shape
      (suffixed keys only, `time_capped`, no checkpoint/model_path); cap=None
      unchanged behavior; dispatch deadline stops submissions but drains
      in-flight.

### Task 6: launch path — `reuse_actors` + pending trials (underutil issue 1)

**Files:** `ritme/tune_models.py`, `ritme/find_best_model_config.py`, tests.

- [ ] `tune.TuneConfig(reuse_actors=True, ...)` (function trainables re-seed at
      entry, mutate only their per-trial config copy, and get a fresh trial dir
      via `FunctionTrainable.reset_config` — verified in ray 2.55.1).
- [ ] Pending-trials override, honoring an externally set env var:

```python
_EXTERNAL_MAX_PENDING_TRIALS = os.environ.get("TUNE_MAX_PENDING_TRIALS_PG")

def _resolve_max_pending_trials(max_concurrent_trials, requested=None):
    if requested is not None:
        return int(requested)
    if _EXTERNAL_MAX_PENDING_TRIALS is not None:
        return int(_EXTERNAL_MAX_PENDING_TRIALS)
    return max(2, int(max_concurrent_trials))
```

      set `os.environ["TUNE_MAX_PENDING_TRIALS_PG"]` in `run_trials` before the
      Tuner is built; new config key `max_pending_trials` (default None) threaded
      like the others. (TPE already runs `constant_liar=True`, which is designed
      for parallel suggests.)
- [ ] Tests: Tuner kwargs carry `reuse_actors=True`; resolution precedence
      (explicit > external env > derived); env var actually set by run_trials.

### Task 7: per-family CPU appetite + trac decoupling (underutil issue 2a/2b)

**Files:** `ritme/tune_models.py`, tests.

- [ ] Appetite:

```python
# Families whose per-fold fit is single-threaded (no n_jobs/nthread wiring):
# they can occupy at most one CPU per parallel fold.
FOLD_PARALLEL_MODELS = frozenset({"linreg", "logreg", "trac"})

def _max_usable_cpus_per_trial(model_type, k_folds):
    if model_type in FOLD_PARALLEL_MODELS:
        return max(1, int(k_folds or 1))
    return None
```

- [ ] `_get_resources(max_concurrent_trials, model_type=None, k_folds=1)`:
      cap `cpus` by appetite; print right-sizing guidance when
      `cpus * max_concurrent_trials < all_cpus_avail`.
- [ ] `run_trials` gains `resource_sizing_concurrency=None`; resources derived
      from it (fallback `max_concurrent_trials`) + `exp_name` + `k_folds`.
      `run_all_trials` passes the *original* `max_concurrent_trials` as
      `resource_sizing_concurrency` while trac still launches at the /3-reduced
      concurrency → trac reserves `min(all_cpus // original, k_folds)` per trial.
- [ ] Tests: linreg/logreg/trac capped at k_folds; xgb/rf/nn uncapped; trac
      resources sized from original concurrency; guidance print; existing
      `_get_resources(2)` call shape still valid.

### Task 8: parallel per-fold feature engineering (underutil issue 2d)

**Files:** `ritme/feature_space/_process_train.py`,
`ritme/model_space/static_trainables.py`, `ritme/model_space/nn_trainables.py`,
`ritme/tests/test_feature_space.py`.

- [ ] Extract module-level per-fold/refit helpers used by both serial and
      parallel paths (identical math, so parallel == serial by construction):

```python
def _engineer_fold_arrays(config, train_val, tax, tr_idx, va_idx):
    train_raw, val_raw = train_val.iloc[tr_idx], train_val.iloc[va_idx]
    train_eng, fold_ft_ls, frozen = _engineer_features(config, train_raw, tax)
    val_eng, _, _ = _engineer_features(config, val_raw, tax, frozen_params=frozen)
    X_tr = train_eng[fold_ft_ls].fillna(np.nan).astype(float).values
    X_va = val_eng[fold_ft_ls].fillna(np.nan).astype(float).values
    return X_tr, X_va

def _engineer_refit_arrays(config, train_val, tax):
    refit_eng, refit_ft_ls, _ = _engineer_features(config, train_val, tax)
    X_refit = refit_eng[refit_ft_ls].fillna(np.nan).astype(float).values
    return X_refit, refit_ft_ls, config.get("data_alr_denom_idx_map")
```

- [ ] `process_train_kfold(..., n_workers: int = 1)`: when `n_workers > 1 and
      ray.is_initialized()`, `ray.put(train_val)` / `ray.put(tax)` once, run the
      K fold tasks + 1 refit task as node-pinned `num_cpus=0` remotes throttled
      to `n_workers` (same NodeAffinity pattern as the fold fits); parent
      applies the refit task's returned ALR denom map to `config` and encodes
      all `y` arrays in-process (unchanged `_encode_target` semantics, including
      the `_label_encoder` stash — `_stash_enrich_categories` runs before the
      fan-out so tasks see the stash). Serial path (default) routes through the
      same helpers.
- [ ] Trainable K-fold paths pass
      `n_workers=max(1, min(n_splits + 1, int(cpus_per_trial)))`.
- [ ] Tests: helper equality vs. legacy serial outputs; real-ray equality test
      (small fixture incl. `data_transform="alr"` + data-dependent selection):
      `n_workers=3` folds/X_refit/denom-map identical to `n_workers=1`.

### Task 9: docs

- [ ] README config table: add `scheduler_grace_period`, `scheduler_max_t`,
      `max_trial_duration_s`, `max_pending_trials` rows (concise, no rationale
      dumps).

### Task 10: verification

- [ ] `conda run -n ritme ruff check ritme/`
- [ ] `conda run -n ritme python -m pytest ritme/tests -q`
- [ ] Smokes (AGENTS.md, forced clean re-run, lowered budgets): MLflow
      regression sweep, MLflow classification sweep, WandB sweep (skip + report
      if no WANDB_API_KEY), snapshot mode, `ritme_example_usage.ipynb`.
- [ ] Verify fresh log dirs + non-empty `mlflow_logs.csv` per AGENTS.md.

---

## Execution status (2026-08-27)

- Tasks 1-9: implemented; 595 unit tests pass; ruff + pinned pre-commit hooks pass.
- Extra fix found by the smoke run (not in the plan): with `reuse_actors=True`,
  Ray 2.55 returns an *unsynced* scratch working dir from `get_trial_dir()` on
  reused actors, so `model.pkl` / `taxonomy.pkl` / `label_encoder.pkl` never
  reached `Result.path` and best-model retrieval crashed with
  FileNotFoundError. Fixed by writing artifacts to the storage
  `trial_fs_path` (== `Result.path`) via `_trial_artifact_dir()`; regression
  test `test_artifacts_land_at_result_path_with_actor_reuse` fails on the old
  behavior and passes on the fix.
- Task 10 (smokes): PASSED on the final code -- snapshot (with
  `SLURM_CPUS_PER_TASK=8`, exercising multi-CPU reservations + parallel
  engineering), MLflow regression (7 families, retrieval + best-model save),
  MLflow classification (4 families), `ritme_example_usage.ipynb`, and
  `evaluate_trials_mlflow.ipynb` (eval + SHAP). WandB: sink plumbing
  validated (full linreg sweep with synced per-trial runs), but two later
  attempts stalled inside wandb.ai uploads (rate limiting / flaky network
  after machine sleep) -- re-run `./run_experiment_wandb.sh` once on a
  stable connection.
- Post-review fixes (from the 4-agent PR review): HyperBand
  (`fully_reproducible: true`) keeps inert scheduler rungs in K-fold mode
  (reachable milestones made it pause + restart checkpoint-less trials from
  fold 0); systemic-failure early stop now raises unconditionally after
  fit(); early-stopped-vs-crashed trials are distinguished via
  `result.error` with actionable no-deployable-trial errors for both
  checkpoint and sklearn/trac families; trac right-sizing guidance counts
  launched slots; `TUNE_MAX_PENDING_TRIALS_PG` user overrides are detected
  dynamically; degenerate `max_trial_duration_s` / `max_pending_trials`
  rejected at config time; plus the review's test-coverage gaps (wiring
  assertions, pre-refit cap branch, task_type forwarding).
- Separate pre-existing fixes bundled (flag for possible split into own
  commits): `retrieve_n_init_best_models` selecting dual-task nn_class by
  the wrong metric in regression runs, and `evaluate_mlflow.create_color_map`
  using `matplotlib.cm.get_cmap` (removed in matplotlib >= 3.9).
