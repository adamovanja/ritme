# ritme: about half of the manuscript campaign's CPU time did no work

Three issues to file against [adamovanja/ritme](https://github.com/adamovanja/ritme),
derived from **every ritme search behind the manuscript submission** (22
experiments across use cases 1-3, launched 13-18 May 2026, all before the
August 2026 submission). Each section is self-contained: copy one into a new
issue, including its figure.

Shared method, run set and reproduction steps are in
[Background](#background) at the bottom.

## Summary

The 20 searches that produced a result were charged **24,434 CPU core-hours**
and consumed **12,820 (52%)**; the rest sat idle inside the allocation. A
further **6,014 core-h** went to searches and attempts that returned nothing
at all.

Those two figures are what the three issues below account for, and they sum
exactly: 4,319 + 7,295 + 6,014 = 17,628. SLURM charged 34,076 core-h in
total; the remaining 3,629 were earlier runs that completed and were later
re-run deliberately — a choice rather than a defect, excluded throughout.

"Consumed" is total job CPU, which also covers the driver, raylet and MLflow,
so 52% is an upper bound on time spent inside a trial.

![What the campaign was charged for](figures/issue_underutilization/fig4-campaign-outcome.png)

| # | Title | Cost | Labels |
|---|---|---|---|
| [1](#issue-1--the-trial-launch-path-is-serial-so-short-trial-searches-cannot-fill-their-allocation) | The trial launch path is serial, so short-trial searches cannot fill their allocation | 4,319 core-h idle (17.7%) | `performance` `scheduler` |
| [2](#issue-2--a-trial-reserves-cores-its-model-cannot-use) | A trial reserves cores its model cannot use | 7,295 core-h idle (29.9%) | `performance` `resources` |
| [3](#issue-3--fatal-conditions-are-detected-only-after-the-budget-is-spent) | Fatal conditions are detected only after the budget is spent | 6,014 core-h wasted (24.6%) | `bug` `usability` |

All three percentages share the 24,434 core-h base, so they add.

The two idle-core issues are independent and multiply, so fixing one alone
leaves the other in place:

```
utilisation = (trials in flight / concurrency)   x  (cores a trial uses / cores reserved)
              slot fill        -> Issue 1           core fill         -> Issue 2
```

![Where each final manuscript search's allocation went](figures/issue_underutilization/fig1-allocation-breakdown.png)

Blue is work; grey is Issue 2; orange is Issue 1. Only the four `xgb`
searches clear 85%. Utilisation by model family:

| family | n | allocated | utilisation | slot fill | core fill | median trial |
|---|---|---|---|---|---|---|
| `logreg`    | 1 | 1152 core-h | **14.3%** | 21.8% | 65.4% | 16 s |
| `linreg`    | 3 | 3461 core-h | **21.4%** | 63.4% | 33.7% | 33 s |
| `trac`      | 2 | 3225 core-h | **29.3%** | 96.7% | 30.3% | 751 s |
| `rf_class`  | 1 | 1152 core-h | **33.7%** | 44.0% | 76.6% | 26 s |
| `rf`        | 2 | 2535 core-h | **49.5%** | 82.1% | 60.4% | 49 s |
| `nn_class`  | 1 | 1153 core-h | **54.2%** | 97.6% | 55.6% | 2604 s |
| `nn_corn`   | 2 | 2307 core-h | **58.5%** | 98.5% | 59.4% | 4703 s |
| `nn_reg`    | 2 | 2306 core-h | **60.1%** | 98.3% | 61.1% | 3958 s |
| `xgb_class` | 2 | 2304 core-h | **65.7%** | 67.3% | 97.7% | 28 s |
| `xgb`       | 4 | 4837 core-h | **92.0%** | 98.2% | 93.7% | 226 s |

Rows are core-hour-weighted, so `slot fill x core fill = utilisation` holds
within rounding.

Read down the two fill columns and the split is obvious: families with short
trials lose to Issue 1, families with unusable reservations lose to Issue 2.
`xgb` is the only family that avoids both, because its trials are long enough
to hide the launch cost *and* `nthread` genuinely uses the reservation.

---

## Issue 1 — The trial launch path is serial, so short-trial searches cannot fill their allocation

**Labels:** `performance` `scheduler` · **Cost:** 4,319 core-h never reserved
(17.7% of the 24,434 charged to successful searches)

### Summary

ritme's Ray Tune driver starts one trial at a time, and each start costs a
fixed ~7.5 s. Throughput is capped near 400 configurations/hour regardless of
allocation, so any search whose trials are shorter than
`7.5 s x concurrency` leaves slots permanently empty.

### Actual behaviour

![Utilisation collapses as cores are added, because the launch path cannot keep up](figures/issue_underutilization/fig2-launch-ceiling.png)

Left: on a controlled 4→128 core sweep, utilisation *peaks at 32 cores* and
falls from 89.4% to 68.7% by 128. Right: slot fill across both the sweep and
the manuscript searches is set entirely by whether the launch interval a
configuration needs (`mean_trial_duration / concurrency`) sits above or below
the driver's floor. No point crosses the ceiling.

The floor is invariant to allocation size:

- Median end-of-trial to start-of-next at **one** slot, where nothing hides
  the cost: **7.52 / 7.59 / 7.54 s** across three seeds.
- 1st percentile of the inter-start gap: **7.32 / 7.29 / 7.12 / 7.65 / 7.20 s**
  at 8 / 16 / 32 / 64 / 128 cores — a 16x range, same floor.
- Across the 72,877 trials inside the 20 producing runs, **no two started
  within 4 s** of each other; the global minimum is 5.24 s and only 2,864
  gaps fall below 6 s. This holds in runs where half the slots sat empty.

Every launch-saturated search lands on the same ceiling:

| search | trials | median launch gap | trials/h achieved | slot fill |
|---|---|---|---|---|
| `u3_logreg_tpe`    | 8612 |  9.61 s | 374 | **21.8%** |
| `u1_linreg_tpe`    | 9023 |  8.74 s | 391 | **27.5%** |
| `u3_rf_class_tpe`  | 9030 |  8.73 s | 392 | **44.0%** |
| `u3_xgb_class_tpe` | 8349 |  9.25 s | 362 | **44.3%** |
| `u1_rf_tpe`        | 5244 | 11.10 s | 228 | 78.0% |
| `u2_linreg_tpe`    | 7469 |  8.68 s | 323 | 81.2% |

None exceeds ~400/h, the rate a ~9 s launch interval allows.

### Root cause

Two Ray Tune settings, both left at their defaults in `tune_models.py`:

1. **`max_pending_trials = 1`.** Ray hard-codes this for any searcher that is
   not a `BasicVariantGenerator`, and ritme passes `OptunaSearch`
   (`tune_models.py:459`, used at `:701`):

   ```python
   # ray/tune/execution/tune_controller.py, _get_max_pending_trials
   # Only BasicVariantGenerator supports > 1 pending trials.
   if not isinstance(search_alg, BasicVariantGenerator):
       return 1
   ```

   Enforced twice — `while len(self._pending_trials) < self._max_pending_trials`
   and `if self._actor_manager.num_pending_actors >= self._max_pending_trials:
   break` — so one trial may be created and one actor bootstrapping at any
   instant, however many slots are free. All 18 sweep logs print only ever
   `| 1 PENDING` while RUNNING reaches the configured maximum.

2. **`reuse_actors = False`** (Ray's default, `ray/tune/tune_config.py:73`;
   ritme never sets it). Each trial gets a fresh worker process that must
   re-import `ritme.model_space.static_trainables`, which is expensive and
   partly avoidable: `torchmetrics` pulls in `torchvision`'s computer-vision
   model zoo (`static_trainables.py:34-62`), paid even by sklearn-only
   searches.

Neither `reuse_actors` nor the `TUNE_MAX_PENDING_TRIALS_PG` override appears
anywhere in ritme.

**How much of the floor the import explains is not established, and we are
not claiming it.** Measured cold-import cost varies over 15-55 s across
consecutive runs on this filesystem, and the non-torch subset alone (numpy,
pandas, sklearn, xgboost, skbio, ray, classo) takes 9-13 s — longer than the
7.5 s floor it would be invoked to explain. The serialisation *mechanism*
above is what the evidence supports; the per-launch cost model is not, and
the A/B in the acceptance criteria is the way to settle it.

### Proposed fix

- Pass `reuse_actors=True` to `tune.TuneConfig`, which takes process spawn
  and module import off the launch path entirely. The trainables are not
  quite pure functions of `config` — `_save_label_encoder` pops from it
  (`static_trainables.py:315`) and the xgb path assigns `config["nthread"]`
  in place (`:2699`) — so the mutations and the per-trial directory obtained
  via `get_trial_dir()` need checking under reuse. Two things that make it
  look feasible: there is no module-level mutable state, and every path
  re-seeds numpy/random/torch on entry, so RNG carry-over is not a risk.
  Memory growth across thousands of trials in one long-lived actor (u3_logreg
  ran 8,612) is the other thing to watch.
- Set `TUNE_MAX_PENDING_TRIALS_PG` (or use a searcher Ray accepts as
  supporting >1 pending trial) so actor bootstrap overlaps across trials.
- Move the torch / lightning / torchmetrics / coral_pytorch imports into the
  `nn_*` trainables so `static_trainables` imports with sklearn, xgboost and
  numpy alone.
- Consider batched Optuna asks so one driver step yields several suggestions.

Ray's default exists for a reason — its comment says it does not want to
generate configurations before the searcher has learned from the last results
— so raising it trades a little sampler quality for throughput and should be
measured, not assumed.

### Acceptance criteria

- [ ] rerunning `u3_xgb_class_tpe` at the same allocation: median inter-start
      gap below ~4 s, slot fill above 80%, >400 configurations/hour
- [ ] the same run with only the import change, to attribute the gain — note
      the sklearn/xgboost/ray floor is 9-13 s, so import time cannot drop to
      1-2 s and should not be the target; the launch gap is
- [ ] `nn_*` searches still run unchanged

---

## Issue 2 — A trial reserves cores its model cannot use

**Labels:** `performance` `resources` · **Cost:** 7,295 core-h reserved and
idle (29.9% of the same base) — the larger of the two idle-core losses

### Summary

`cpus_per_trial` is one number derived from the allocation and the
concurrency setting, with no reference to the model. Eight of the ten model
families cannot occupy what they are given, and for `trac` a memory
workaround makes the mismatch three times worse.

### Actual behaviour

![Every family but xgb reserves more cores than it can occupy](figures/issue_underutilization/fig3-core-fill-by-model.png)

### Root cause

`tune_models.py:357-365` produces one number for every model family:

```python
cpus = max(1, all_cpus_avail // max_concurrent_trials)
```

which becomes a `PlacementGroupFactory([{"CPU": n, ...}])` reservation held
for the trial's whole lifetime. Four distinct mismatches follow, all from
that single decision:

**2a. Models that cannot thread at all.** ritme's own comment at
`tune_models.py:570-576` already records it — *"linreg: not parallelizable,
CPU-only / trac: solver Path-Alg not parallelized"*. `_build_linreg` accepts
`n_jobs` and its docstring says it is *"accepted for factory-signature parity
… but unused"*; `_build_logreg` does not wire `n_jobs` to the saga solver;
`trac` has no thread parameter anywhere. linreg and logreg reserve 5 CPUs
each; trac reserves 20 or 26, see 2b. ritme does fan the K folds out across a
trial's CPUs (`_allocate_fold_resources`, `static_trainables.py:163-174`),
which is why these families still register above one core per trial; what is
missing is any use of the reservation *within* a fold.

**2b. The trac memory workaround inflates the reservation ~3.3x.** At
`tune_models.py:800-805`:

```python
# reduce number of concurrent trials in case of trac - requires too much memory
if model == "trac":
    # todo: implement trac to reduce memory usage
    max_concurrent_trials_launched = max(1, round(max_concurrent_trials / 3))
```

Because `_get_resources` divides the allocation by the *reduced* number, the
memory fix also triples the CPU reservation of the one model ritme documents
as unparallelised. Confirmed in the logs — ritme printed `Using these
resources: CPU 20` for `u1_trac_tpe` (60 CPUs, config concurrency 10) and
`CPU 26` for `u2_trac_tpe` (80 CPUs) instead of 6 and 8. `u2_trac_tpe` then
held 78 of its 80 cores with three running trials using ~7 each: **2,173
core-h idle across the two trac runs**, the single largest per-family loss.

**2c. `nn_*` trials do not occupy their reservation.**
`torch.set_num_threads(cpus_per_trial)` sets 10 (`static_trainables.py:2088`,
`:2253`), and the five `nn_*` searches sustain roughly 6 of those 10 cores
(core fill 49.5-69.5% per run) — **2,000-2,300 core-h idle**, the range
spanning whether truncated trials are imputed. Attribution caveat: `core
fill` is a residual, so nothing here measures thread occupancy directly, and
2d below is a competing explanation for the same core-hours. Read it as "the
trial does not use its cores", not specifically as "torch does not scale".

**2d. Serial feature engineering holds the trial while its cores idle.**
`process_train_kfold` (`feature_space/_process_train.py:316-403`) engineers
`2K + 1 = 11` design matrices in a plain loop before any fan-out, and a grep
for `n_jobs|Parallel|multiprocess|ray.remote|ThreadPool|concurrent.futures`
across all of `ritme/feature_space/` returns **zero** hits. For `linreg` this
prologue is the whole trial: duration *anti*-correlates with the number of
features kept (Spearman −0.34 in u2, −0.50 in u1), because the cost is
computing the selection over ~35k input features, not the sub-second fit.

### Proposed fix

- Let each model family declare its CPU appetite alongside its search space,
  and size the reservation from that rather than from `max_cuncurrent_trials`
  alone. Single-threaded families should reserve 1 CPU plus whatever the fold
  fan-out uses.
- Decouple trac's memory cap from the CPU divisor. Note the obvious version
  of this makes things *worse*: capping trac at 6 CPUs per trial while still
  running 3 slots reserves 18 of 60 cores, below the 21.7 u1_trac uses today,
  and `core fill` would read ~100% while utilisation *falls*. The fix has to
  restore concurrency too — either implement trac's memory reduction so more
  slots fit, or right-size the allocation to what 3 slots can use (see
  "right-size the allocations" below).
- Parallelise the per-fold engineering across the CPUs the trial has already
  reserved — they are idle during exactly that phase — and/or memoise
  engineering across trials sharing the feature-engineering half of the
  config, which the search changes far less often than model hyperparameters.

### Blocked by Issue 1

Shrinking `cpus_per_trial` and raising concurrency to match will **not** help
on its own. Trials in flight is `mean_trial_duration / launch_interval`
(Little's law), not the slot count. With a 7.5-9 s launch floor,
`u2_linreg_tpe` at 50 slots x 1 CPU would still run ~10 trials in flight and
end at ~10 busy cores — no better than the 12 it gets today. Fix Issue 1
first, or this change will show no improvement and read like a failed fix.

`trac` is a partial exception: its trials are long (751 s median) and its
slot fill is already 97%, so Issue 1 is not what constrains it. But as noted
above, 2b cannot be fixed by shrinking the reservation alone — trac needs
either a real memory fix or a smaller allocation, and only the first recovers
search throughput.

### Acceptance criteria

- [ ] a `trac` search reserves the same cores per trial whether or not the
      memory reduction applies, **and** its overall utilisation rises — core
      fill alone can be satisfied while utilisation falls
- [ ] `core fill` from `analyze_utilization.py` rises above 80% for `trac`
      and `linreg`
- [ ] `xgb` is unaffected (already 94-98%)

---

## Issue 3 — Fatal conditions are detected only after the budget is spent

**Labels:** `bug` `usability` · **Cost:** 6,014 core-h that returned no result

### Summary

Every health check ritme applies to a search runs *after* `Tuner.fit()`
returns. A search whose trials all error, or that never reports the metric it
optimises, therefore consumes its entire time budget before raising. Six
attempts in the final set failed only after burning a full ~23 h budget, and
two experiments never produced a result at all.

### Actual behaviour

`u1_nn_class_tpe` and `u2_nn_class_tpe` each ran the full 23 h on 50 CPUs and
ended with:

```
RuntimeError: No best trial found for the given metric: roc_auc_macro_ovr_val.
This means that no trial has reported this metric, or all values reported for
this metric are NaN.
```

Trials had been completing all along — 67 and 120 of them — but every
reported value was NaN. Both appear in the manuscript's merged metrics table
as all-NaN rows. Cost: **2,302 core-h** for two empty rows.

A second signature, on `u1_nn_reg_tpe`, `u1_nn_corn_tpe`, `u3_nn_class_tpe`
and `u3_logreg_tpe`, is the failure-rate policy firing post-hoc:

```
RuntimeError: Some trials encountered errors.
  ValueError: Expected more than 1 value per channel when training, got input size ...
  ValueError: Input X contains NaN.
```

Failed attempts cost **3,712 core-h** in total, but only **3,247** of that
belongs to this mechanism — the six attempts that burned a full budget before
failing. Most of the remainder is `u1_rf_tpe`, which failed 12 times quickly
(0 full-budget failures) and is a different problem. Any attempt not ending
`COMPLETED` is counted, including ones cancelled by hand.

ritme *does* validate cheap preconditions up front, and that part works well
— `nn_corn requires a target with few discrete rounded levels`, `Model types
['logreg'] are not compatible with task_type`, and `This experiment tag
already exists` all abort within seconds. The gap is that nothing checks the
search's health *while it runs*.

### Root cause

`tune_models.py:703-707`:

```python
result = analysis.fit()          # consumes the entire time_budget_s

# Check all trials & apply failure-rate policy
_check_for_errors_in_trials(result, max_trial_failure_rate=max_trial_failure_rate)
```

`DEFAULT_MAX_TRIAL_FAILURE_RATE = 0.005` (`tune_models.py:61`) is strict
enough that a systematically broken search is certain to fail the policy —
but it is only ever evaluated once the budget is gone. The same applies to
`retrieve_n_init_best_models` → `_select_best_with_one_se` →
`result_grid.get_best_result(...)` (`evaluate_models.py:602`, `:553`), which
is the call that raises "No best trial found".

### Proposed fix

- Evaluate the failure-rate policy *during* the search — a Ray Tune callback
  on `on_trial_error` that aborts once the rate exceeds
  `max_trial_failure_rate` over a minimum sample of trials.
- Assert after the first N completed trials that the optimisation metric has
  been reported at least once and is finite; abort with the existing "no best
  trial" message immediately rather than at the end.
- Related, already filed as `issue_inefficient_large_trials.md`: ASHA can
  never prune, because `grace_period = 10` exceeds the maximum
  `training_iteration` any trainable emits. Worth noting the ceiling is lower
  than that document states: only the sequential xgb and `nn_*` paths report
  per fold (5), while the parallel sklearn path — linreg, logreg, rf,
  rf_class, trac — reports **once**, so `grace_period = 1, max_t = k_folds`
  would still prune nothing for half the families. A guard warning when
  `grace_period` exceeds what the trainable can emit belongs with the checks
  above.

### Acceptance criteria

- [ ] a search whose trials all raise aborts within minutes, not at the end
      of `time_budget_s`
- [ ] a search that never reports its target metric aborts with the same
      message it raises today, but early
- [ ] a healthy search is unaffected (no change in trial count or accuracy)

### Sequencing note

Issue 3's ASHA half interacts with Issue 1 and should follow it. Pruning
shortens trials, and shorter trials lower slot fill under a fixed launch
floor. `u1_xgb_tpe` is safe (it needs 85.4 s per launch against a ~8 s
floor), but `u3_xgb_class_tpe` is already launch-limited at 44.3%: halving
its trial duration would not raise its ~389 configurations/h, only empty more
cores.

---

## Background

### Run set

The manuscript's final set is the 22 ritme experiments in the newest merged
metrics table, `use_cases/all_experiments_metrics_260522.csv` (the same file
the manuscript's accuracy numbers come from), excluding the `*_original` and
`*_automl` baselines. Launch dates are 13, 15 and 18 May 2026. Twenty have
usable trial logs; the two that do not are the `nn_class` failures in Issue 3.

Each experiment's producing job is the last SLURM job that completed under
that experiment tag. It reproduces exactly the three ritme job ids pinned
independently in `benchmarking/collect_b3.py` — the other three B3 cells are
auto-sklearn jobs the tag rule never sees — so the validation covers 3 of the
20 runs. Earlier attempts
are classified separately: **failed** (never completed, counted as waste) or
**superseded** (completed, then deliberately re-run; 3,629 core-h, not counted
as a defect).

### Method

CPU efficiency (sacct `TotalCPU / (NCPUS × Elapsed)`) factorises exactly into
two independent terms recoverable from `mlflow_logs.csv`: trials in flight is
`sum(trial durations) / wall time`, and cores per running trial is
`sacct TotalCPU / sum(trial durations)`.

The two factors multiply back to `utilisation` by construction, so that is a
check on the arithmetic, not evidence for the decomposition. It is exact
wherever the concurrency divides the allocation evenly, and off by 2.6% for
`u2_trac_tpe`, where 80 // 3 = 26 leaves two cores unassignable. What the
split does rest on is that the two factors come from independent sources —
trial timestamps for one, sacct CPU for the other.

Both factors are measured against ritme's *effective* concurrency, not the
configured value, because of Issue 2b — the analysis replicates the `/3`
reduction and cross-checks every derived `cpus_per_trial` against the
`Using these resources: CPU n` line in the run's own log.

Trials with no `end_time` are closed at the last event in the log rather than
dropped — any such row, not only trials truncated at the budget. That
imputation is 1.3% of trial-time for `u1_xgb_tpe` but reaches **8.6%**
(`u2_nn_reg_tpe`), and 29% at 128 cores in the core sweep, where dropping
them would badly undercount occupancy.

Since the factors multiply to a fixed total, every imputed second moves
core-hours from Issue 2 to Issue 1, so the split is worth bounding: imputing
gives **4,319 / 7,295**, dropping gives **4,736 / 6,877**. That ~420 core-h
swing is roughly 10% of Issue 1 and changes no conclusion.

`core fill` is a residual (`utilisation / slot fill`), so it also absorbs
driver, raylet and MLflow CPU and can slightly exceed 100% when trials are
short. It is an upper bound on true per-trial saturation, which strengthens
rather than weakens the conclusions.

### Ruled out

- **TPE sampler cost growing with history.** The evidence is the
  random-search control: 1st-percentile gaps are 7.30-7.93 s for random
  against 7.26-7.74 s for TPE — the same floor with and without a fitted
  sampler. (Quarter medians hold at 8.2-9.9 s only for the launch-saturated
  runs; across all 20 they span 8 s to 21 min, tracking trial duration. For a
  saturated driver a flat median would look the same either way, so the p1
  control is what carries this.)
- **Contention at high core counts — partly, not wholly.** Seed-averaged
  median trial duration is flat at 39-55 s from 4 to 128 cores, but per seed
  it spans 38.2-70.4 s. Two runs are slow, not one: 64 cores seed 2 (70.4 s)
  is the only job of the 18 that landed on a different node class, which
  explains it; 128 cores seed 0 (68.7 s) is not explained. With 128-core
  utilisation at 81.0 / 64.6 / 60.6% across three seeds, the headline "89.4%
  to 68.7%" rests on a noisy mean. The 4-core anchor is weak too: all three
  seeds shared one node and one ended OUT_OF_MEMORY with 35 configurations
  against 85 and 77.
- **Ray's idle-worker prestart.** The archived runs had it on and the sweep
  had it off; both show the same launch floor.
- **TPE's random warm-up.** 75 startup trials for xgb is 0.9% of a 23 h run,
  and warm-up gaps match post-warm-up gaps.
- **Fewer trials meaning worse utilisation.** TPE reached *higher* CPU
  efficiency than random search (90.8-95.2% vs 86.0-88.3%, no overlap)
  because it converges on feature-richer configurations — median
  `nb_features` climbs 73 → 4,394 between the first and last two hours for
  TPE seed 0 while random stays near 37. Larger trials keep the reserved
  cores busy. (Trial *counts* do not separate the arms — TPE 524/394/1568 vs
  random 1578/1257/1482 overlap — so only the efficiency figure supports
  this.)

### Reproducing

From the [ritme_examples](https://github.com/adamovanja/ritme_examples)
benchmarking suite, against already-completed runs (submits no jobs):

```shell
python -m benchmarking.analyze_utilization   # tables incl. both factors
python -m benchmarking.plot_utilization      # the figures embedded above
```

These regenerate the campaign totals, the per-family table, the launch-ceiling
and right-sizing tables and all four figures. Claims taken directly from the
raw logs — import timings, the trial counts and minimum gaps, the
quarter-by-quarter gaps, the random-search control, the `nb_features` drift
and the node-class attribution — are not covered by these two commands and
were computed ad hoc; `benchmarking/results/b1/b1_trials.csv` in particular
carries no start times or `nb_features`.

Environment: ritme 1.4.2, ray 2.55.1, Python 3.14, SLURM on ETH Euler.

### Suggested order

1. **Issue 1** — prerequisite for most of Issue 2, and the only loss that
   grows as allocations grow.
2. **Issue 3** — independent of the other two, and the cheapest to implement;
   it stops whole budgets being lost.
3. **Issue 2** — largest single loss, but most of it only pays off after 1.
   The `trac` half (2b) is the exception and can be fixed immediately.

### Meanwhile: right-size the allocations

Independent of any ritme change, most of these searches request far more than
they can use. Because they are launch-limited rather than compute-limited,
cutting the request costs almost no search — configurations per hour is set
by the launch path, not the core count — while shortening queueing time:

| search | requested | mean cores busy |
|---|---|---|
| `u1_xgb_tpe`       | 50 | 47.4 |
| `u2_xgb_tpe`       | 60 | 50.8 |
| `u3_xgb_class_tpe` | 50 | 21.5 |
| `u2_trac_tpe`      | 80 | 19.4 |
| `u1_trac_tpe`      | 60 | 21.7 |
| `u2_linreg_tpe`    | 50 | 12.2 |
| `u1_linreg_tpe`    | 50 |  7.7 |
| `u3_logreg_tpe`    | 50 |  7.1 |
