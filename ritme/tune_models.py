import os
import random
import warnings
from functools import partial
from typing import Any, Callable, Dict

import dotenv
import numpy as np
import pandas as pd
import ray
import skbio
import torch
from optuna.samplers import (
    CmaEsSampler,
    GPSampler,
    QMCSampler,
    RandomSampler,
    TPESampler,
)
from ray import init, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune import ResultGrid
from ray.tune.schedulers import AsyncHyperBandScheduler, HyperBandScheduler
from ray.tune.search.optuna import OptunaSearch

from ritme.feature_space.utils import _PAST_SUFFIX_RE
from ritme.model_space import static_searchspace as ss
from ritme.model_space import static_trainables as st

# Constants
MODEL_TRAINABLES = {
    # model_type: trainable
    "xgb": st.train_xgb,
    "xgb_class": st.train_xgb_class,
    "nn_reg": st.train_nn_reg,
    "nn_class": st.train_nn_class,
    "nn_corn": st.train_nn_corn,
    "linreg": st.train_linreg,
    "logreg": st.train_logreg,
    "rf": st.train_rf,
    "rf_class": st.train_rf_class,
    "trac": st.train_trac,
}

REGRESSION_MODELS = {"xgb", "nn_reg", "linreg", "rf", "trac", "nn_corn"}
CLASSIFICATION_MODELS = {"xgb_class", "nn_class", "logreg", "rf_class"}

# Re-export so callers can keep importing the cap constant from ``tune_models``
# without learning about its physical home in ``static_trainables``.
DEFAULT_NN_CORN_MAX_LEVELS = st.DEFAULT_NN_CORN_MAX_LEVELS

# Failure-rate threshold for the trial-error policy
# (see issue_m_strict_abort_policy.md). 0.5% surfaces flaky trials as a
# warning while preserving the existing abort for systemic failures.
DEFAULT_MAX_TRIAL_FAILURE_RATE = 0.005

TASK_METRICS = {
    "regression": ("rmse_val", "min"),
    "classification": ("roc_auc_macro_ovr_val", "max"),
}

DEFAULT_SCHEDULER_GRACE_PERIOD = 10
DEFAULT_SCHEDULER_MAX_T = 100

# overview of all optuna samplers is available here:
# https://optuna.readthedocs.io/en/stable/reference/samplers/index.html
OPTUNA_SAMPLER_CLASSES = {
    "RandomSampler": RandomSampler,
    "TPESampler": TPESampler,
    "CmaEsSampler": CmaEsSampler,  # inefficient cat.params + cond. search space
    "GPSampler": GPSampler,  # inefficient cond. search space
    "QMCSampler": QMCSampler,  # inefficient cat.params + cond. search space
}

# Floor and multiplier for n_startup_trials = max(floor, mult * effective_dims).
# 5x is enough to seed multivariate-TPE's grouped Parzen estimator without
# wasting the time budget on pure random search (Optuna's own default is 10,
# but multivariate=True with conditional groups needs more).
N_STARTUP_TRIALS_FLOOR = 20
N_STARTUP_TRIALS_MULT = 5


class _RecordingTrial:
    """Mock Optuna ``Trial`` that records every ``suggest_*`` call.

    Used by :func:`_count_effective_dims` to count search-space dimensions
    along the *longest conditional path* (the deepest trial the search space
    can produce). Values are recorded but never validated, so invalid bounds
    (e.g. ``low > high`` when the user data has zero microbial features) are
    tolerated; only the call count matters.

    The trial steers categorical choices toward the longest branch:

    - ``data_selection`` -> ``"abundance_ith"`` so the dependent
      :func:`_suggest_integer` fires (adding ``data_selection_i``);
    - ``penalty`` -> ``"elasticnet"`` so the conditional ``l1_ratio`` fires;
    - any other categorical -> first non-None choice, so the count reflects
      "user picked an option" rather than the no-op None branch.

    For ``n_hidden_layers``, ``suggest_int`` returns the maximum value, so the
    nn search space's ``for i in range(n_hidden_layers)`` loop adds one
    width-parameter call per possible layer.
    """

    _LONGEST_BRANCH_PICKS = {
        "data_selection": "abundance_ith",
        "penalty": "elasticnet",
    }

    def __init__(self) -> None:
        self.params: Dict[str, Any] = {}

    def _store(self, name: str, value: Any) -> Any:
        self.params[name] = value
        return value

    def suggest_categorical(self, name: str, choices):
        preferred = self._LONGEST_BRANCH_PICKS.get(name)
        if preferred is not None and preferred in choices:
            return self._store(name, preferred)
        for c in choices:
            if c is not None:
                return self._store(name, c)
        return self._store(name, choices[0])

    def suggest_int(self, name: str, low, high, log: bool = False, step: int = 1):
        if name == "n_hidden_layers":
            return self._store(name, high)
        return self._store(name, low)

    def suggest_float(self, name: str, low, high, log: bool = False, step=None):
        return self._store(name, low)


def _count_effective_dims(
    model_type: str,
    train_val: pd.DataFrame,
    tax,
    model_hyperparameters: dict,
) -> int:
    """Count ``trial.suggest_*`` calls along the longest conditional path.

    Runs :func:`static_searchspace.get_search_space` once with a
    :class:`_RecordingTrial`, returning the number of distinct parameters the
    search space would suggest in its deepest branch. Driven by the actual
    search-space code so the count cannot drift if a parameter is added.
    """
    trial = _RecordingTrial()
    ss.get_search_space(
        trial,
        model_type=model_type,
        tax=tax,
        train_val=train_val,
        model_hyperparameters=model_hyperparameters,
    )
    return len(trial.params)


def _adaptive_n_startup_trials(
    model_type: str,
    train_val: pd.DataFrame,
    tax,
    model_hyperparameters: dict,
) -> int:
    """Compute n_startup_trials from the model's effective search-space dim.

    Returns ``max(N_STARTUP_TRIALS_FLOOR, N_STARTUP_TRIALS_MULT * effective_dims)``,
    where ``effective_dims`` is :func:`_count_effective_dims` evaluated on the
    longest conditional path of the search space.

    For nn trainables this longest path is the one with ``n_hidden_layers ==
    max`` of the (possibly user-customised) range, so every per-layer width
    parameter ``n_units_hl{i}`` contributes one dimension. We deliberately use
    the maximum (not the midpoint or expected value) for two reasons:

    1. **Consistency with non-NN models.** Every other model type's count is
       the longest path through its conditional structure; treating
       ``n_hidden_layers`` as just another conditional branch keeps the rule
       uniform.
    2. **Asymmetric cost of being wrong.** Under-budgeting startup trials is
       worse than over-budgeting them: TPE engaging on a poorly-fit kernel
       density estimate produces biased proposals for the rest of the run,
       whereas a few extra random samples are still informative for the KDE
       once it does engage. The deepest configuration is the highest-dim
       group multivariate-TPE has to model, so sizing for it (rather than the
       average shallow case) avoids that asymmetric failure.

    The trade-off is that shallow nn configurations spend more wall time in
    random sampling than strictly necessary; in practice that is dominated by
    nn-trainable wall time per fit, not by the count of warmup trials.
    """
    n_dims = _count_effective_dims(model_type, train_val, tax, model_hyperparameters)
    return max(N_STARTUP_TRIALS_FLOOR, N_STARTUP_TRIALS_MULT * n_dims)


def _get_slurm_resource(resource_name: str, default_value: int = 0) -> int:
    """Retrieve SLURM resource value from environment variables."""
    try:
        return int(os.environ[resource_name])
    except (KeyError, ValueError):
        return default_value


def _check_for_errors_in_trials(
    result: ResultGrid,
    max_trial_failure_rate: float = DEFAULT_MAX_TRIAL_FAILURE_RATE,
) -> dict:
    """Inspect a finished search for trial errors and apply the failure-rate
    policy.

    Raises ``RuntimeError`` when more than ``max_trial_failure_rate`` of
    trials errored, or when the search produced zero trials at all
    (typically a misconfigured ``time_budget_s`` or empty search space).
    Below the threshold, emits a ``RuntimeWarning`` AND prints the
    breakdown -- the print survives the default Python warnings filter
    that dedupes repeats by message+lineno in long-lived processes.

    Returns a breakdown dict (``num_trials``, ``num_errors``,
    ``failure_rate``, ``error_classes``).
    """
    num_trials = len(result)
    num_errors = int(result.num_errors)
    error_classes = sorted({type(err).__name__ for err in (result.errors or [])})

    breakdown = {
        "num_trials": num_trials,
        "num_errors": num_errors,
        "failure_rate": ((num_errors / num_trials) if num_trials > 0 else float("nan")),
        "error_classes": error_classes,
    }

    if num_trials == 0:
        # A successful Ray Tune run always yields at least one trial.
        # Zero means the search aborted before launching anything -- a
        # too-short time_budget_s, empty search space, or scheduler
        # mis-configuration. Silently returning here would hide a real
        # campaign failure (issue_m_strict_abort_policy.md).
        raise RuntimeError(
            "Ray Tune produced 0 trials -- check time_budget_s, search "
            f"space, and scheduler configuration. (Pre-launch errors: "
            f"{num_errors}, classes: {error_classes}.)"
        )

    failure_rate = breakdown["failure_rate"]
    if failure_rate > max_trial_failure_rate:
        raise RuntimeError(
            f"Trial failure rate {failure_rate:.4f} exceeds "
            f"max_trial_failure_rate={max_trial_failure_rate} "
            f"({num_errors}/{num_trials} trials errored: {error_classes}). "
            f"See Ray Tune logs above for details."
        )
    if num_errors > 0:
        msg = (
            f"{num_errors}/{num_trials} trials errored "
            f"(failure_rate={failure_rate:.4f} <= "
            f"max_trial_failure_rate={max_trial_failure_rate}); proceeding "
            f"with surviving trials. Ray Tune error classes: {error_classes}."
        )
        # Print as well as warn: Python's default warnings filter dedupes
        # repeats by (message, module, lineno) so a long-lived process
        # could silently swallow subsequent surfacings.
        print(f"WARNING: {msg}")
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return breakdown


def _validate_run_inputs(
    model_types: list[str],
    task_type: str,
    target: str,
    train_val: pd.DataFrame,
    nn_corn_max_levels: int = DEFAULT_NN_CORN_MAX_LEVELS,
) -> None:
    """Cheap pre-flight validation shared by ``find_best_model_config`` and
    ``run_all_trials``.

    Runs every check that can fire before any Ray Tune work starts:
    ``task_type`` validity, model-type / task-type compatibility, the
    ``nn_corn`` target shape check, and the snapshot + NaN gate. Calling
    this *before* ``_define_experiment_path`` keeps a config-rejection
    failure from leaving a stub experiment directory on disk
    (issue_m_existing_dir.md).
    """
    if task_type not in TASK_METRICS:
        raise ValueError(
            f"Invalid task_type '{task_type}'. Must be one of: "
            f"{list(TASK_METRICS.keys())}."
        )

    # nn_class is dual-task: it uses a classification nn_type internally but
    # reports metrics matching the overall task_type. nn_corn lives only in
    # REGRESSION_MODELS (CORN's rounded-target path is regression-style).
    task_allowed = (
        REGRESSION_MODELS if task_type == "regression" else CLASSIFICATION_MODELS
    )
    task_allowed = task_allowed | {"nn_class"}
    invalid = set(model_types) - task_allowed
    if invalid:
        raise ValueError(
            f"Model types {sorted(invalid)} are not compatible with task_type "
            f"'{task_type}'. Allowed models: {sorted(task_allowed)}."
        )

    # Cap is checked against the full target before splitting -- the
    # trainable only sees per-fold subsets.
    if "nn_corn" in model_types:
        target_series = train_val[target]
        if not pd.api.types.is_numeric_dtype(target_series):
            raise ValueError(
                f"nn_corn requires a numeric target; column '{target}' has "
                f"dtype {target_series.dtype}. nn_corn is regression-only -- "
                f"use logreg / rf_class / xgb_class / nn_class for "
                f"categorical targets."
            )
        n_nan = int(target_series.isna().sum())
        if n_nan > 0:
            raise ValueError(
                f"nn_corn target column '{target}' contains {n_nan} NaN "
                f"values. Drop or impute them before the run; otherwise "
                f"nn_corn_max_levels would be evaluated against a corrupted "
                f"level set."
            )
        rounded_unique = np.unique(np.round(target_series.to_numpy()).astype(int))
        st._check_nn_corn_levels(int(rounded_unique.size), nn_corn_max_levels)

    has_snapshots = any(_PAST_SUFFIX_RE.search(col) for col in train_val.columns)
    all_micro = [c for c in train_val.columns if c.startswith("F")]
    has_snapshot_nans = (
        (pd.isna(train_val[all_micro]).values.any() if all_micro else False)
        if has_snapshots
        else False
    )
    if has_snapshots and has_snapshot_nans:
        # Only xgb/xgb_class support native NaN handling; reject any other request
        xgb_model = "xgb_class" if task_type == "classification" else "xgb"
        incompatible = sorted(set(model_types) - {xgb_model})
        if incompatible:
            raise ValueError(
                f"NaNs in snapshot features detected (missing_mode='nan'); only "
                f"'{xgb_model}' supports native NaN handling. Requested model "
                f"types {incompatible} are incompatible. Either set "
                f"missing_mode='exclude' in split_train_test or restrict "
                f"ls_model_types to ['{xgb_model}']."
            )


def _get_resources(max_concurrent_trials: int) -> dict:
    """Calculate CPU and GPU resources based on SLURM environment variables."""
    all_cpus_avail = _get_slurm_resource("SLURM_CPUS_PER_TASK", 1)
    all_gpus_avail = _get_slurm_resource("SLURM_GPUS_PER_TASK", 0)
    cpus = max(1, all_cpus_avail // max_concurrent_trials)
    gpus = max(0, all_gpus_avail // max_concurrent_trials)
    print(f"Using these resources: CPU {cpus}")
    print(f"Using these resources: GPU {gpus}")
    return {"cpu": cpus, "gpu": gpus}


def _define_scheduler(
    fully_reproducible: bool,
    scheduler_grace_period: int,
    scheduler_max_t: int,
    metric: str,
    mode: str,
    k_folds: int,
):
    # In K-fold mode the running aggregate reports strip the bare ``<metric>``
    # key (see ``_emit_running_fold_aggregate`` / ``issue_eval_class.md``) so
    # that Ray Tune's ``get_best_result(metric=<metric>)`` cannot land on a
    # checkpoint-less row. The scheduler must therefore monitor a key that
    # survives in the running aggregates -- ``<metric>_mean`` -- otherwise it
    # would see no metric mid-trial and never prune. In single-split mode the
    # callbacks emit ``<metric>`` per epoch, so the scheduler reads it
    # directly.
    sched_metric = f"{metric}_mean" if int(k_folds or 1) > 1 else metric

    # Note: Both schedulers might decide to run more trials than allocated
    if not fully_reproducible:
        # AsyncHyperBand enables aggressive early stopping of bad trials.
        # ! efficient & fast BUT
        # ! not fully reproducible with seeds (caused by system load, network
        # ! communication and other factors in env) due to asynchronous mode only
        return AsyncHyperBandScheduler(
            metric=sched_metric,
            mode=mode,
            # Stop trials at least this old in time (measured in training iteration)
            grace_period=scheduler_grace_period,
            # Stopping trials after max_t iterations have passed
            max_t=scheduler_max_t,
        )
    else:
        # ! HyperBandScheduler slower BUT
        # ! improves the reproducibility of experiments by ensuring that all trials
        # ! are evaluated in the same order.
        return HyperBandScheduler(metric=sched_metric, mode=mode, max_t=scheduler_max_t)


def _define_search_algo(
    func_to_get_search_space: Callable,
    exp_name: str,
    tax: pd.DataFrame,
    train_val: pd.DataFrame,
    model_hyperparameters: dict,
    optuna_searchspace_sampler: str,
    seed_model: int,
    metric: str,
    mode: str,
):
    # Partial function needed to pass additional parameters
    define_search_space = partial(
        func_to_get_search_space,
        model_type=exp_name,
        tax=tax,
        train_val=train_val,
        model_hyperparameters=model_hyperparameters,
    )

    # Define sampler to be used with OptunaSearch
    if optuna_searchspace_sampler not in OPTUNA_SAMPLER_CLASSES.keys():
        raise ValueError(
            f"Unrecognized sampler '{optuna_searchspace_sampler}'. "
            f"Available options are: {list(OPTUNA_SAMPLER_CLASSES.keys())}"
        )
    sampler_class = OPTUNA_SAMPLER_CLASSES[optuna_searchspace_sampler]

    sampler_kwargs = {"seed": seed_model}
    if sampler_class in (TPESampler, CmaEsSampler, GPSampler):
        # n_startup_trials sized from the model's effective search-space
        # dimensionality (see _adaptive_n_startup_trials). The previous default
        # of 1000 was a hand-set upper bound that, on most ritme runs,
        # consumed the entire time budget on random sampling and prevented TPE
        # from engaging.
        n_startup = model_hyperparameters.get("n_startup_trials") or (
            _adaptive_n_startup_trials(exp_name, train_val, tax, model_hyperparameters)
        )
        sampler_kwargs["n_startup_trials"] = n_startup
    if sampler_class is TPESampler:
        # handles conditional search spaces well
        sampler_kwargs["multivariate"] = True
        sampler_kwargs["group"] = True
        sampler_kwargs["constant_liar"] = True

    # if provided extract starting points for config
    if "start_points_to_evaluate" in model_hyperparameters.keys():
        # [{"a": 6.5, "b": 5e-4}, {"a": 7.5, "b": 1e-3}]
        start_points = model_hyperparameters["start_points_to_evaluate"]
    else:
        start_points = None

    return OptunaSearch(
        space=define_search_space,
        sampler=sampler_class(**sampler_kwargs),
        metric=metric,
        mode=mode,
        points_to_evaluate=start_points,
    )


def _load_wandb_api_key() -> str:
    """Load WandB API key from .env file."""
    dotenv.load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    if api_key is None:
        raise ValueError("No WANDB_API_KEY found in .env file.")
    return api_key


def _load_wandb_entity() -> str:
    """Load WandB entity from .env file."""
    dotenv.load_dotenv()
    entity = os.getenv("WANDB_ENTITY")
    if entity is None:
        raise ValueError("No WANDB_ENTITY found in .env file.")
    return entity


class _SafeMLflowLoggerCallback(MLflowLoggerCallback):
    """MLflowLoggerCallback that tolerates trials which never started.

    Why: when a Ray worker actor dies before ``log_trial_start`` runs (e.g.
    SIGSEGV during actor bootstrap on busy HPC nodes), Ray Tune still calls
    ``on_trial_error`` → ``log_trial_end``. The upstream implementation does
    a bare ``self._trial_runs[trial]`` lookup and raises ``KeyError``, which
    propagates out of ``Tuner.fit`` and tears down the whole sweep. Skipping
    the call is safe: no MLflow run was ever opened for that trial, so there
    is nothing to finalize.
    """

    def log_trial_end(self, trial, failed: bool = False) -> None:
        if trial not in self._trial_runs:
            return
        super().log_trial_end(trial, failed=failed)


def _define_callbacks(tracking_uri: str, exp_name: str, experiment_tag: str) -> list:
    """Define callbacks based on the tracking URI."""
    callbacks = []

    if tracking_uri.startswith("sqlite:///"):
        callbacks.append(
            _SafeMLflowLoggerCallback(
                tracking_uri=tracking_uri,
                experiment_name=exp_name,
                # Below would be double saving: local_dir as artifact here
                # save_artifact=True,
                tags={"experiment_tag": experiment_tag},
            )
        )
    elif tracking_uri == "wandb":
        # Load WandB API key from .env file
        api_key = _load_wandb_api_key()
        entity = _load_wandb_entity()
        callbacks.append(
            WandbLoggerCallback(
                api_key=api_key,
                entity=entity,
                project=experiment_tag,
                tags={experiment_tag},
            )
        )
    else:
        print("No valid tracking URI provided. Proceeding without logging callbacks.")

    return callbacks


def run_trials(
    tracking_uri: str,
    exp_name: str,
    trainable,
    train_val: pd.DataFrame,
    target: str,
    host_id: str,
    stratify_by: list[str] | None,
    seed_data: int,
    seed_model: int,
    tax: pd.DataFrame,
    tree_phylo: skbio.TreeNode,
    path2exp: str,
    time_budget_s: int,
    max_concurrent_trials: int,
    experiment_tag: str,
    fully_reproducible: bool = False,
    model_hyperparameters: dict = None,
    optuna_searchspace_sampler: str = "TPESampler",
    scheduler_grace_period: int = DEFAULT_SCHEDULER_GRACE_PERIOD,
    scheduler_max_t: int = DEFAULT_SCHEDULER_MAX_T,
    resources: dict = None,
    task_type: str = "regression",
    k_folds: int = 1,
    nn_corn_max_levels: int = DEFAULT_NN_CORN_MAX_LEVELS,
    max_trial_failure_rate: float = DEFAULT_MAX_TRIAL_FAILURE_RATE,
) -> ResultGrid:
    if model_hyperparameters is None:
        model_hyperparameters = {}

    if resources is None:
        # If not a SLURM process, default values are used
        resources = _get_resources(max_concurrent_trials)

    # Trainable parallelization & GPU capabilities:
    # - linreg: not parallelizable, CPU-only
    # - trac: solver Path-Alg not parallelized, CPU-only (Classo)
    # - rf: parallel via n_jobs, CPU-only
    # - xgb: parallel via nthread, GPU via device='cuda' when allocated
    # - nn_reg, nn_class, nn_corn: parallel via torch threads, GPU auto-detected
    #   by Lightning via CUDA_VISIBLE_DEVICES set by Ray

    # Set seed for search algorithms/schedulers
    random.seed(seed_model)
    np.random.seed(seed_model)
    torch.manual_seed(seed_model)

    # Initialize Ray with the runtime environment
    # shutdown()  #can't be used when launching on HPC with externally started
    # ray instance
    # todo: configure dashboard here - see "ray dashboard set up" online once
    # todo: ray (Q2>Py) is updated
    context = init(
        address="local",
        include_dashboard=False,
        ignore_reinit_error=True,
        # #Configure logging if needed
        # logging_level=logging.DEBUG,
        # log_to_driver=True,
    )
    print(f"Ray cluster resources: {ray.cluster_resources()}")
    print(f"Dashboard URL at: {context.dashboard_url}")

    # Define metric and mode to optimize
    metric, mode = TASK_METRICS[task_type]

    # In K-fold mode the running aggregate reports strip the bare ``<metric>``
    # key (see ``_emit_running_fold_aggregate`` / ``issue_eval_class.md``) so
    # that ``get_best_result(metric=<metric>)`` cannot land on a checkpoint-
    # less row. Both the scheduler and the search algorithm must therefore
    # monitor a key that appears in EVERY report (running + final);
    # ``<metric>_mean`` fits. Ray Tune's strict metric validator also enforces
    # the search algo's metric to be present on every report.
    runtime_metric = f"{metric}_mean" if int(k_folds or 1) > 1 else metric

    # Define schedulers
    scheduler = _define_scheduler(
        fully_reproducible,
        scheduler_grace_period,
        scheduler_max_t,
        metric,
        mode,
        k_folds,
    )

    # Define search algorithm with search space
    search_algo = _define_search_algo(
        ss.get_search_space,
        exp_name,
        tax,
        train_val,
        model_hyperparameters,
        optuna_searchspace_sampler,
        seed_model,
        runtime_metric,
        mode,
    )

    storage_path = os.path.abspath(path2exp)

    callbacks = _define_callbacks(tracking_uri, exp_name, experiment_tag)

    # Inject allocated resource counts so trainables can configure parallelism
    cpus_per_trial = resources.get("cpu", 1)
    gpus_per_trial = resources.get("gpu", 0)

    analysis = tune.Tuner(
        # Trainable with input parameters passed and set resources
        tune.with_resources(
            tune.with_parameters(
                trainable,
                train_val=train_val,
                target=target,
                host_id=host_id,
                stratify_by=stratify_by,
                seed_data=seed_data,
                seed_model=seed_model,
                tax=tax,
                tree_phylo=tree_phylo,
                cpus_per_trial=cpus_per_trial,
                gpus_per_trial=gpus_per_trial,
                task_type=task_type,
                k_folds=k_folds,
                nn_corn_max_levels=nn_corn_max_levels,
            ),
            resources,
        ),
        # Logging and checkpoint configuration
        run_config=tune.RunConfig(
            # Complete experiment name with subfolders of trials within
            name=exp_name,
            storage_path=storage_path,
            # Checkpoint: to store best model - is retrieved in evaluate_models.py
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_score_attribute=metric,
                checkpoint_score_order=mode,
                # num_to_keep=3 lets the trainable callbacks write up to 2
                # checkpoints per trial (a first-improvement safety checkpoint
                # plus a final best-state checkpoint) without ever reaching
                # Ray Tune's "force experiment-state snapshot every num_to_keep
                # checkpoints per trial" threshold. Each forced snapshot is a
                # cluster-wide sync; with 5 concurrent trials this used to
                # produce visible bottleneck warnings (see ritme#84).
                num_to_keep=3,
            ),
            failure_config=tune.FailureConfig(max_failures=2),
            callbacks=callbacks,
        ),
        tune_config=tune.TuneConfig(
            # ``metric`` / ``mode`` are intentionally NOT set here: the
            # scheduler holds them (so K-fold can read ``<metric>_mean``
            # while single-split reads ``<metric>``), and Ray Tune raises
            # if both Tuner and scheduler specify them. ``search_alg`` and
            # ``scheduler`` already carry the values they need; downstream
            # ``get_best_result`` calls must pass ``metric`` / ``mode``
            # explicitly (see ``_select_best_with_one_se``).
            scheduler=scheduler,
            # Number of trials to run - schedulers might decide to run more trials
            num_samples=-1,
            # time restriction for the whole experiment
            time_budget_s=time_budget_s,
            # Set max concurrent trials to launch
            max_concurrent_trials=max_concurrent_trials,
            # Define search algorithm
            search_alg=search_algo,
        ),
    )
    # ResultGrid output
    result = analysis.fit()

    # Check all trials & apply failure-rate policy
    _check_for_errors_in_trials(result, max_trial_failure_rate=max_trial_failure_rate)

    return result


def run_all_trials(
    train_val: pd.DataFrame,
    target: str,
    host_id: str,
    stratify_by: list[str] | None,
    seed_data: int,
    seed_model: int,
    tax: pd.DataFrame,
    tree_phylo: skbio.TreeNode,
    mlflow_uri: str,
    path_exp: str,
    time_budget_s: int,
    max_concurrent_trials: int,
    experiment_tag: str,
    model_types: list = [
        "xgb",
        "nn_reg",
        "nn_class",
        "nn_corn",
        "linreg",
        "rf",
        "trac",
    ],
    fully_reproducible: bool = False,
    model_hyperparameters: dict = {},
    optuna_searchspace_sampler: str = "TPESampler",
    task_type: str = "regression",
    k_folds: int = 1,
    nn_corn_max_levels: int = DEFAULT_NN_CORN_MAX_LEVELS,
    max_trial_failure_rate: float = DEFAULT_MAX_TRIAL_FAILURE_RATE,
) -> dict[str, ResultGrid]:
    results_all = {}

    # Defensive: pre-flight checks may already have run in
    # ``find_best_model_config`` (issue_m_existing_dir.md), but this entry
    # point is also tested / called standalone, so re-run them here.
    _validate_run_inputs(
        model_types=model_types,
        task_type=task_type,
        target=target,
        train_val=train_val,
        nn_corn_max_levels=nn_corn_max_levels,
    )

    # First apply snapshot-related constraints for models
    model_types = model_types.copy()

    has_snapshots = any(_PAST_SUFFIX_RE.search(col) for col in train_val.columns)
    if has_snapshots and "trac" in model_types:
        # Remove trac when dynamic snapshots present
        model_types.remove("trac")
        print("Snapshots detected; removing 'trac' from model types.")

    # Now remove trac if taxonomy/phylogeny missing
    if (tax is None or tree_phylo is None) and "trac" in model_types:
        model_types.remove("trac")
        print(
            "Removing trac from model_types since no taxonomy and phylogeny were "
            "provided."
        )

    if not os.path.exists(path_exp):
        os.makedirs(path_exp)
    for model in model_types:
        print(f"Ray Tune training of: {model}...")

        # If there are any, get the range of hyperparameters to check
        if model.startswith("nn"):
            model_hparams_type = model_hyperparameters.get("nn_all_types", {})
        elif model == "rf_class":
            model_hparams_type = model_hyperparameters.get(
                "rf_class", model_hyperparameters.get("rf", {})
            )
        elif model == "xgb_class":
            model_hparams_type = model_hyperparameters.get(
                "xgb_class", model_hyperparameters.get("xgb", {})
            )
        else:
            model_hparams_type = model_hyperparameters.get(model, {})
        # Get data hparam
        model_hparams_type.update(
            {k: v for k, v in model_hyperparameters.items() if k.startswith("data_")}
        )
        model_hparams_type["data_enrich_with"] = model_hparams_type.get(
            "data_enrich_with", None
        )

        # reduce number of concurrent trials in case of trac - requires too much memory
        if model == "trac":
            # todo: implement trac to reduce memory usage
            max_concurrent_trials_launched = max(1, round(max_concurrent_trials / 3))
            print(
                f"Reducing max_concurrent_trials to {max_concurrent_trials_launched} "
                "for trac model due to high memory requirements."
            )
        else:
            max_concurrent_trials_launched = max_concurrent_trials
        result = run_trials(
            mlflow_uri,
            model,
            MODEL_TRAINABLES[model],
            train_val,
            target,
            host_id,
            stratify_by,
            seed_data,
            seed_model,
            tax,
            tree_phylo,
            path_exp,
            time_budget_s,
            max_concurrent_trials_launched,
            experiment_tag,
            fully_reproducible=fully_reproducible,
            model_hyperparameters=model_hparams_type,
            optuna_searchspace_sampler=optuna_searchspace_sampler,
            task_type=task_type,
            k_folds=k_folds,
            nn_corn_max_levels=nn_corn_max_levels,
            max_trial_failure_rate=max_trial_failure_rate,
        )
        results_all[model] = result
    return results_all
