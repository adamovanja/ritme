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
When performing and changes or additions to this repos make sure to follow best practices in software development. making sure all added code is clearly structured, only contains comments when really needed. Any added functionality (= the difference to main branch) should be properly tested with unit tests.
When testing new functionality always do it in an activated conda environment called `ritme` - if it does not exist yet, make sure to create with with `make create-env`. Never install packages in the base environment!

## Design constraints

- **Compositionality per snapshot**: Microbial features within a single time point are compositional. Transforms (CLR/ILR/ALR) are applied per snapshot, never across time points.
- **No data leakage**: Feature engineering parameters (selection, ALR denominator, enrichment schema) are learned on train and reused as-is on test via `TunedModel` state.
- **TRAC incompatibility**: TRAC requires a single compositional snapshot + phylogenetic tree. It is automatically excluded when dynamic snapshots are detected.
- **NaN handling**: `missing_mode="nan"` requires `ls_model_types` to contain only XGBoost; requesting any other model raises a `ValueError`. NaN rows are separated before compositional transforms and reintroduced afterward.
- **Column naming**: Past snapshots are suffixed (`F0__t-1`, `age__t-2`); current (t0) columns remain unsuffixed. This ensures backward compatibility with the static workflow.

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

### Formatting
- Ensure all code is formatted according to the pre-commit hooks in this repos.

### Commits
- Imperative form, matching existing `git log` style.
- AI attribution is mandatory: include `Co-Authored-By: <tool>` in commit trailers.
