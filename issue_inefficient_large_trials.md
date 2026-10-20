# ritme: expensive trials are never cut short

A small minority of trials consumes most of a run's compute. Two mechanisms
that should bound a single trial's cost are either inert or absent.

Paths are relative to the installed package `.../site-packages/ritme/`;
measurements come from the 23 h TPE runs in `use_cases/ritme_runs/local/`.

## Evidence

Per-trial wall-clock, from `metrics.time_total_s` in `mlflow_logs.csv`:

| Run | trials | median | p90 | max |
|---|---|---|---|---|
| u1_xgb_tpe | 950 | 234 s | 2 324 s | **11 360 s** (3.2 h) |
| u2_xgb_tpe | 1 818 | 224 s | 1 116 s | 8 449 s |
| u3_xgb_class_tpe | 8 349 | 18 s | 27 s | 4 786 s |

For u1_xgb, the 33 % of trials exceeding 10 minutes consumed **88.4 %** of all
core-time; the longest single trial ran 3.2 h, 14 % of one slot's wall clock
(1.4 % of the run's compute).

## What actually drives the cost

**Feature engineering, not model hyperparameters.** Median trial time for
u1_xgb by choice:

| `data_aggregation` | n | median | max |
|---|---|---|---|
| `None` (all 18 150 features) | 683 | **525 s** | 11 360 s |
| `tax_genus` | 74 | 48 s | 82 s |
| `tax_family` | 67 | 48 s | 101 s |
| `tax_order` | 65 | 47 s | 69 s |
| `tax_class` | 62 | 45 s | 58 s |

| `data_transform` | n | median |
|---|---|---|
| `ilr` | 574 | 407 s |
| `rank` | 69 | 346 s |
| `clr` | 78 | 306 s |
| `alr` | 65 | 297 s |
| `None` | 58 | 56 s |
| `pa` | 107 | 47 s |

Skipping taxonomic aggregation is a ~11× median and >100× tail cost multiplier;
`ilr` over `pa` is another ~9×.

Model hyperparameters barely matter. Correlation of `log(time)` with each
numeric parameter over the 950 u1_xgb trials: `n_estimators` **r = +0.12**,
`num_parallel_tree` +0.15, `max_depth` −0.07. The reason is that xgb training
early-stops (`booster.best_iteration`, `static_trainables.py:2740`), so
`n_estimators` is only an upper bound on boosting rounds, not the number
actually trained.

**This was tested directly and the hyperparameter hypothesis failed.** An
ablation (`u1_xgb_tpe_reduced`) pinned `n_estimators` to 100 and
`num_parallel_tree` to 1, TPOT's values. Cost did not fall — it rose slightly:

| u1_xgb | trials | median | max | trial-hours |
|---|---|---|---|---|
| full space | 950 | 234 s | 11 360 s | 225.1 |
| reduced space | 888 | **351 s** | 15 348 s | 226.4 |

(The last column is Σ`time_total_s`, i.e. trial-hours; at 5 CPUs per trial the
core-hours are 1 125 and 1 132. It carries little signal either way — both
runs had the same 23 h × 10 slots, so ~225 of a possible 230 slot-hours just
restates that u1_xgb keeps its slots full. The median is the informative row.)

With the cheap model knobs pinned, the sampler simply spent its budget in the
expensive feature-engineering branch instead. **Bounding model hyperparameters
does not bound ritme's trial cost.**

## Issue 1 — the ASHA scheduler can never prune

`tune_models.py:392-399` builds an `AsyncHyperBandScheduler` with
`grace_period=DEFAULT_SCHEDULER_GRACE_PERIOD = 10` (`tune_models.py:68`).

Only the *sequential* K-fold paths report once per fold — xgb and `nn_*`,
via `_emit_running_fold_aggregate` (`static_trainables.py:135, :152`) — so
those trials emit 5 results. The parallel sklearn path (`_run_kfold_sklearn`,
`:760`) dispatches its folds at once and reports **once**. Measured across
the archived logs: `training_iteration` tops out at 5 for xgb and `nn_*`, and
at **1** for linreg, logreg, rf, rf_class and trac. For u1_xgb specifically,
941 of 950 trials reached 5, none higher.

With Ray's `reduction_factor=4`, rungs sit at `grace_period × 4^k` — iterations
10 and 40. **Neither is reachable.** The scheduler runs and prunes nothing;
every trial is paid for in full however poor it looks after fold 1.

Suggested fix: derive the rungs from the fold count inside `_define_scheduler`
(`tune_models.py:368`, which already receives `k_folds`) — `grace_period = 1`,
`max_t = k_folds` — putting decisions after folds 1 and 4. Note this only
helps the families that report per fold; for linreg, logreg, rf, rf_class and
trac a trial still emits a single result, so ASHA has nothing to prune on and
those paths would need mid-trial reporting first. Both parameters are
also unreachable from any config entrypoint: `run_trials` accepts them
(`tune_models.py:555-556`) but `run_all_trials` (`:712-835`) neither accepts nor
forwards them, and `find_best_model_config.py:372-400` reads no scheduler key.

This is the **safer** of the two fixes: it prunes on the metric, so it removes
trials that are *bad*, not merely slow.

Caveat: pruning on one fold's noisy estimate risks culling eventual winners.
Validate by replaying the logged per-fold metrics before shipping.

## Issue 2 — no per-trial wall-clock cap

`tune.RunConfig(...)` (`tune_models.py:664-682`) passes no `stop=`, and the
package defines no `Stopper`. `time_budget_s` bounds the whole experiment, not
an individual trial.

TPOT bounds every evaluation with `max_eval_time_mins` (`tpot/base.py:1526`)
via a `@threading_timeoutable` decorator (`tpot/gp_deap.py:393`); a timed-out
pipeline scores `-inf` and is abandoned.

**A time cap in ritme is not a free win, and is more dangerous than it looks.**
The expensive branch is where u1's winning representations live — all eight
best u1_xgb trials by validation use `data_aggregation=None` with `ilr`, at
206–2 252 s each, and the single best took 1 153 s. Capping trial time
therefore biases the search away from exactly the rich feature representations
ritme exists to explore. Replaying the logged trials under hypothetical caps:

| u1_xgb cap | share of core-time | best `rmse_val` found |
|---|---|---|
| none | 100 % | 2.2552 |
| 1 200 s | 53.6 % | **2.2552** (unchanged) |
| 600 s | 34.9 % | 2.2568 |
| 300 s | 21.1 % | 2.2599 |

The cost column charges each capped trial the full `cap` before it is killed,
which is what a real cap spends. (Counting only the trials that *finish*
under the cap gives a much rosier 24 / 12 / 4.7 % and understates the cost by
2–4.5×.) So a 20-minute cap still buys back nearly half the compute at
identical accuracy; a 5-minute cap saves ~79 % for 0.005 RMSE. The right value
is dataset-dependent, so it must be a config key with a generous default, not a
hard-coded constant. u3 is unaffected either way — all ten of its best configs
use `tax_order` and run in 16–21 s.

Caveat: Ray evaluates stop conditions only when a trial reports. For xgb and
`nn_*` that means a cap truncates at fold boundaries and cannot pre-empt
inside one long `xgb.train` call; for the single-report families above a Ray
`stop=` would **never fire at all**. A hard guarantee needs an in-trainable
timeout either way.

Note also that TPOT's own implementation of this idea is unsafe with native
code: its `stopit` timeout raises asynchronously, and when it landed inside
XGBoost's C code it corrupted the heap and killed a 10.7 h job with
`double free or corruption`. Any ritme equivalent should stop at a checkpoint,
not by injecting an exception into a running fit.

## Relationship between the two

Complementary, not redundant. ASHA prunes trials that look *bad* on their
metric; a wall-clock cap kills trials that are *slow* regardless of quality.
Given that ritme's slow branch is also its most promising one on u1, the ASHA
fix should be preferred and attempted first.
