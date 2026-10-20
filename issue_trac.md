# trac: memory workaround still caps utilisation — implement the memory reduction so the `/3` concurrency cut can go

**Labels:** `performance` `resources` `memory`

## Summary

trac searches cannot fill their allocation. `run_all_trials` cuts trac's
concurrency to a third of `max_cuncurrent_trials` as a memory workaround
(`tune_models.py`, `# todo: implement trac to reduce memory usage`). Since
the resource right-sizing fix, each trac trial reserves only what it can use
(`min(allocation // max_cuncurrent_trials, k_folds)` CPUs), but the reduced
slot count means most of the allocation is idle by construction: at 60 CPUs
with `max_cuncurrent_trials: 10` and `k_folds: 5`, trac holds 3 slots × 5
CPUs = 15 of 60 cores. Reservation-level fixes cannot raise utilisation
further — only restoring concurrency can.

## Actual behaviour

- Manuscript campaign: `u1_trac_tpe` / `u2_trac_tpe` reached 29.3 %
  utilisation on 3,225 allocated core-hours (2,173 core-h idle), the largest
  single per-family loss in the campaign.
- ritme now prints right-sizing guidance ("reserving N/M allocated CPUs
  across k concurrent trial(s)"), but within a fixed allocation the idle
  cores remain unusable.

## Root cause

Per-trial memory of the trac trainable is assumed too high to run at full
concurrency, but the actual driver has never been profiled. Candidates:

1. The A matrix (`create_matrix_from_tree`) and per-fold log-geom design
   matrices are materialized per trial; the A matrix is identical for every
   trial of a search (it depends only on phylogeny + taxonomy).
2. All K folds' log-geom matrices are built up front before dispatch
   (`train_trac` K-fold branch), so peak memory scales with K.
3. Classo (`Path-Alg`) solver internals.

## Proposed fix (in ritme)

1. Profile a trac trial's peak RSS on a reference dataset to attribute the
   memory between (1)–(3).
2. If (1)/(2): share the A matrix across trials via a single object-store
   put per search instead of one per trial; keep it sparse end-to-end;
   build fold log-geom matrices lazily per dispatched fold instead of all
   at once.
3. Remove (or make configurable) the `/3` reduction in `run_all_trials`
   once measured memory allows the configured concurrency.
4. If (3) dominates, document the bound and recommend right-sized
   allocations instead (guidance print already exists).

## Acceptance criteria

- [ ] A trac search runs at the configured `max_cuncurrent_trials` without
      OOM on the reference dataset.
- [ ] Overall utilisation (sacct `TotalCPU / (NCPUS × Elapsed)`) rises for a
      trac search at an unchanged allocation — core fill alone can be
      satisfied while utilisation falls.
- [ ] Peak per-trial memory is measured and documented so future allocation
      guidance is grounded.
