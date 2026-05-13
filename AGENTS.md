# AGENTS.md

Guidelines for AI coding agents working on this repository.

---

## Project overview

**ritme** is a Python package for finding the best microbial feature representation and model algorithm for a predictive regression task on next-generation sequencing data. Microbial features are engineered accounting for their statistical characteristics (compositionality, high dimensionality, hierarchical relationships, sparsity).

The software (ritme v1.2.8) and its implications are described in this preprint: https://www.biorxiv.org/content/10.64898/2025.12.08.693045v2.abstract

A high-level overview of the package functionality is in [`README.md`](README.md).

## Data flow

```
metadata + feature_table (+ phylogeny & taxonomy)
  → split_train_test()          # merge, optional temporal snapshotting, train/test split
  → find_best_model_config()    # search over feature engineering × model combinations
  → evaluate_tuned_models()     # evaluate best models on train + held-out test
```

The package supports two modes:
- **Static**: single time point per sample (`ft_t, md_t → target_t`)
- **Dynamic**: temporal snapshotting with sliding windows (`ft_t-N, ..., ft_t → target_t`), enabled via `time_col`, `host_col`, `n_prev` in `split_train_test`

## Key modules

| Path | Purpose |
|------|---------|
| `ritme/split_train_test.py` | Data merging, temporal snapshotting, grouped/stratified splitting |
| `ritme/find_best_model_config.py` | Orchestrates the full model + feature search |
| `ritme/tune_models.py` | Runs Ray Tune trials, enforces model type restrictions |
| `ritme/evaluate_models.py` | `TunedModel` class with per-snapshot predict pipeline |
| `ritme/evaluate_tuned_models.py` | Final evaluation on train + test sets |
| `ritme/explain_features.py` | SHAP feature importance computation and plotting |
| `ritme/feature_space/` | Feature engineering: aggregate, select, transform, enrich |
| `ritme/feature_space/_process_train.py` | Per-snapshot feature processing pipeline |
| `ritme/feature_space/utils.py` | Snapshot utilities (`_slice_snapshot`, `_add_suffix`, `_PAST_SUFFIX_RE`) |
| `ritme/model_space/static_trainables.py` | Model trainables: linreg, xgb, rf, trac, nn_reg, nn_class, nn_corn |
| `ritme/model_space/static_searchspace.py` | Hyperparameter search spaces per model type |
| `ritme/cli.py` | Typer CLI entry point (`ritme split-train-test`, `find-best-model-config`, `evaluate-tuned-models`, `explain-features`) |
| `ritme/evaluate_mlflow.py` | MLflow visualization utilities |
| `config/` | Example experiment configuration files |
| `experiments/` | Example usage notebooks |

## CLI and Python API pattern

Each of the four main functions exists in two forms:

- **Python API** (e.g. `split_train_test()`): accepts in-memory objects (DataFrames, dicts, TreeNode) and returns results directly. Decorated with `@main_function`.
- **CLI wrapper** (e.g. `cli_split_train_test()`): accepts file paths as strings, loads data, delegates to the Python API function, and writes outputs to disk. Also decorated with `@main_function`. Registered in `ritme/cli.py` via Typer.

| Python API | CLI wrapper | CLI command |
|------------|-------------|-------------|
| `split_train_test()` | `cli_split_train_test()` | `ritme split-train-test` |
| `find_best_model_config()` | `cli_find_best_model_config()` | `ritme find-best-model-config` |
| `evaluate_tuned_models()` | `cli_evaluate_tuned_models()` | `ritme evaluate-tuned-models` |
| `compute_shap_values()` | `cli_explain_features()` | `ritme explain-features` |

Internal/private functions are decorated with `@helper_function`. Both decorators are defined in `ritme/_decorators.py` and are used purely as flags (no runtime behavior).

When modifying a main function's signature, update both the Python API and the CLI wrapper, including any CLI-specific parsing (e.g. `stratify_by` is a comma-separated string in the CLI but a list in the Python API).

## Important best practices

When making changes or additions to this repo, follow best practices in software development. This rule is **load-bearing**, not aspirational — every change is reviewed against it:

- **Clearly structured code.** Single-purpose functions, sensible module boundaries, explicit data flow. If a function does multiple things, split it.
- **Comments only when really needed.** Document *why*, not *what*; well-named identifiers already say what.
- **Tests are not optional.** Every piece of added functionality (the diff vs `main`) must have unit tests in `ritme/tests/`. Integration-only coverage via a smoke test does not count as tested for CI purposes.
- **No nested closures used as a default factory pattern.** If a helper function captures state from its enclosing scope purely as a convenience, lift it to module scope and pass the state as explicit arguments. Closures used inside Ray / joblib / multiprocessing dispatch are especially dangerous: cloudpickle ships the entire captured scope to workers. Use closures only when their semantics are genuinely load-bearing (decorators, callbacks, partial application, one-shot tightly-scoped helpers).
- **No imports inside function bodies.** All imports go at the top of the module.
- **Module-level functions over closures when crossing IPC boundaries.** Module-level functions pickle by reference (cheap); closures over arbitrary scope require cloudpickle and ship more than intended.
- **When a helper pattern repeats across sibling functions, lift the helper to module scope and parameterise.** Duplication of the same shape is a signal of a missed abstraction.

When testing new functionality always do it in an activated conda environment called `ritme` - if it does not exist yet, make sure to create with with `make create-env`. Never install packages in the base environment!

## Design constraints

- **Compositionality per snapshot**: Microbial features within a single time point are compositional. Transforms (CLR/ILR/ALR) are applied per snapshot, never across time points.
- **No data leakage**: Feature engineering parameters (selection, ALR denominator, enrichment schema) are learned on train and reused as-is on test via `TunedModel` state.
- **TRAC incompatibility**: TRAC requires a single compositional snapshot + phylogenetic tree. It is automatically excluded when dynamic snapshots are detected.
- **NaN handling**: `missing_mode="nan"` requires `ls_model_types` to contain only XGBoost; requesting any other model raises a `ValueError`. NaN rows are separated before compositional transforms and reintroduced afterward.
- **Column naming**: Past snapshots are suffixed (`F0__t-1`, `age__t-2`); current (t0) columns remain unsuffixed. This ensures backward compatibility with the static workflow.
- **K-fold cross-validation**: Trainables default to 5-fold cross-validation (adaptively capped by group count and smallest stratum; `k_folds: 1` opts out) and `find_best_model_config` selects the best configuration using a one-standard-error rule on the per-fold metric.

## Commands

```bash
# Install
make create-env        # create conda environment
make install           # install package
make dev               # install dev dependencies + pre-commit hooks
make test              # run implemented unit tests

# CLI
ritme split-train-test --help
ritme find-best-model-config --help
ritme evaluate-tuned-models --help
```

## Rules

### Before writing code
- Read the relevant source files before proposing changes.
- Check existing tests in `ritme/tests/` to understand expected behavior.
- For feature engineering changes, understand the per-snapshot processing in `_process_train.py` - verifying that the statistical properties of microbial features are considered.

### Pull requests
- Run `ruff check` and `py.test` before opening a PR. Do not open PRs with failing tests.
- One concern per PR. No unrelated changes.
- PR descriptions: what changed and why. Keep it concise.
- PR title: stick to the naming of former PRs with the prefix (FIX, ADD, ENH, MAINT, ...) in this project.

### Testing
- All new functionality must have corresponding tests in `ritme/tests/`.

### Smoke tests
Unit tests live in `ritme/tests/`; the smoke runs below exercise the
full end-to-end pipeline (`split_train_test` -> `find_best_model_config`
-> evaluation / SHAP) under different tracking sinks and data shapes.
They are not part of CI -- run them locally after any change that
touches Ray Tune integration (`tune_models.py`), feature engineering
(`_process_train.py`, `enrich_features.py`), trainables
(`model_space/`), or the CLI / shell scripts.

**Always force a clean re-run** (the shell scripts skip steps when
their outputs already exist):

1. Delete the relevant data splits and log dirs:
   ```bash
   rm -rf experiments/data_splits_{mlflow,mlflow_class,wandb,snapshot}
   rm -rf experiments/ritme_example_logs/{trials_mlflow,trials_mlflow_class,trials_wandb,trials_snapshot}
   rm -rf experiments/ritme_example_logs/{example_linreg,example_linreg_py,example_logreg,example_logreg_py}
   ```
2. Lower the per-model time budget in the relevant config(s) so the
   smoke finishes in minutes rather than hours -- e.g. set
   `time_budget_s` to 60-180 in `config/trials_*.json`. Restore the
   original value when done.
3. Run the smoke (see table below).
4. Verify the run actually executed -- check that the log dir was
   created in this session:
   ```bash
   ls -la experiments/ritme_example_logs/<exp_tag>/   # mtime should be now
   wc -l experiments/ritme_example_logs/<exp_tag>/mlflow_logs.csv
   ```
   A non-empty `mlflow_logs.csv` (or `wandb_logs.csv`) with a fresh
   mtime confirms training actually ran. An empty / missing log file
   means the shell script's "skip if non-empty" guard fired against
   stale state -- repeat step 1 and retry.

| Smoke | Command (from `experiments/`) | Covers | ETA at recommended budget |
|-------|-------------------------------|--------|----------------------------|
| Python API + CLI examples | execute `ritme_example_usage.ipynb` via `jupyter nbconvert --to notebook --execute ... --allow-errors` | `find_best_model_config` Python API + CLI, single-model linreg/logreg | ~9 min |
| MLflow regression sweep | `./run_experiment_mlflow.sh` (default `regression`) followed by executing `evaluate_trials_mlflow.ipynb` | all 7 regression trainables incl. `nn_corn`, MLflow tracking, SHAP | ~20 min (`time_budget_s=180`/model) |
| MLflow classification sweep | edit `task_type = "classification"` in `evaluate_trials_mlflow.ipynb` (or copy to a scratch ipynb) + run | `logreg`, `xgb_class`, `nn_class`, `rf_class`, K-fold `get_best_result` post-processing | ~12 min (`time_budget_s=180`/model) |
| WandB regression sweep | `./run_experiment_wandb.sh` | `WandbLoggerCallback`, same `run_trials` / scheduler / metric plumbing as MLflow but the wandb tracking sink | ~10 min (`time_budget_s=60`/model) |
| Snapshot / temporal mode | `./run_experiment_snapshot.sh` | `split_train_test` with `--time-col` / `--host-col` / `--n-prev`, the dynamic-mode `__t-N` column suffixing, K-fold + categorical-enrich on snapshot data | ~5 min (`time_budget_s=60`/model) |

When a change is scoped to a specific path, the minimum required smoke is:

- Ray Tune scheduler / metric / `get_best_result` plumbing
  (`tune_models.py`, `evaluate_models.py`): MLflow classification +
  WandB (different tracking sinks share the same plumbing).
- Feature engineering (`feature_space/`): MLflow regression (small N,
  `data_enrich_with` exercises the categorical-universe path) +
  snapshot (temporal column suffixing).
- Trainables (`model_space/static_trainables.py`): MLflow regression
  (covers all 7 regression trainables) +
  `ritme_example_usage.ipynb` (single-split path on linreg/logreg).
- CLI wrappers (`split_train_test.py::cli_*`, `find_best_model_config.py::cli_*`,
  etc.): `ritme_example_usage.ipynb` (CLI section runs each command).

### Formatting
- Ensure all code is formatted according to the pre-commit hooks in this repos.

### Commits
- Imperative form, matching existing `git log` style.
- AI attribution is mandatory: include `Co-Authored-By: <tool>` in commit trailers.

### Writing style
- When writing documentation (e.g. in README files) - make sure the text is concise and does not contain unnecessary legacy/context information that is not crucial for the comment being made.
- Do not transplant conversational rationale into the document. If you justified a change in chat (e.g. "this works because tool X reads file Y..."), the file itself should still only state *what* the reader needs to do. If they want the rationale, upstream tool docs are the right place to send them.
