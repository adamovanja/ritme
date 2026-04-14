import os
import random
from functools import partial
from typing import Callable

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

REGRESSION_MODELS = {"xgb", "nn_reg", "linreg", "rf", "trac"}
CLASSIFICATION_MODELS = {"xgb_class", "nn_class", "nn_corn", "logreg", "rf_class"}

TASK_METRICS = {
    "regression": ("rmse_val", "min"),
    "classification": ("accuracy_val", "max"),
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


def _get_slurm_resource(resource_name: str, default_value: int = 0) -> int:
    """Retrieve SLURM resource value from environment variables."""
    try:
        return int(os.environ[resource_name])
    except (KeyError, ValueError):
        return default_value


def _check_for_errors_in_trials(result: ResultGrid) -> None:
    """Check if any trials encountered errors and raise an exception if so."""
    if result.num_errors > 0:
        raise RuntimeError(
            "Some trials encountered errors. See above for reported Ray Tune errors."
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
    fully_reproducible: bool, scheduler_grace_period: int, scheduler_max_t: int
):
    # Note: Both schedulers might decide to run more trials than allocated
    if not fully_reproducible:
        # AsyncHyperBand enables aggressive early stopping of bad trials.
        # ! efficient & fast BUT
        # ! not fully reproducible with seeds (caused by system load, network
        # ! communication and other factors in env) due to asynchronous mode only
        return AsyncHyperBandScheduler(
            # Stop trials at least this old in time (measured in training iteration)
            grace_period=scheduler_grace_period,
            # Stopping trials after max_t iterations have passed
            max_t=scheduler_max_t,
        )
    else:
        # ! HyperBandScheduler slower BUT
        # ! improves the reproducibility of experiments by ensuring that all trials
        # ! are evaluated in the same order.
        return HyperBandScheduler(max_t=scheduler_max_t)


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
        # These samplers can use n_startup_trials to better explore the space
        # todo: expose this paraemter to user such that it can be configured
        sampler_kwargs["n_startup_trials"] = 1000
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


def _define_callbacks(tracking_uri: str, exp_name: str, experiment_tag: str) -> list:
    """Define callbacks based on the tracking URI."""
    callbacks = []

    if tracking_uri.startswith("sqlite:///"):
        callbacks.append(
            MLflowLoggerCallback(
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
    fully_reproducible: bool = False,
    model_hyperparameters: dict = None,
    optuna_searchspace_sampler: str = "TPESampler",
    scheduler_grace_period: int = DEFAULT_SCHEDULER_GRACE_PERIOD,
    scheduler_max_t: int = DEFAULT_SCHEDULER_MAX_T,
    resources: dict = None,
    task_type: str = "regression",
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

    # Define schedulers
    scheduler = _define_scheduler(
        fully_reproducible, scheduler_grace_period, scheduler_max_t
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
        metric,
        mode,
    )

    storage_path = os.path.abspath(path2exp)
    experiment_tag = os.path.basename(path2exp)

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
            callbacks=callbacks,
        ),
        tune_config=tune.TuneConfig(
            metric=metric,
            mode=mode,
            # Define the scheduler
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

    # Check all trials & check for error status
    _check_for_errors_in_trials(result)

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
) -> dict[str, ResultGrid]:
    results_all = {}

    # Validate task_type
    if task_type not in TASK_METRICS:
        raise ValueError(
            f"Invalid task_type '{task_type}'. Must be one of: "
            f"{list(TASK_METRICS.keys())}."
        )

    # Validate model types against task_type
    # nn_class and nn_corn support both regression and classification tasks
    allowed = REGRESSION_MODELS if task_type == "regression" else CLASSIFICATION_MODELS
    allowed = allowed | {"nn_class", "nn_corn"}
    invalid = set(model_types) - allowed
    if invalid:
        raise ValueError(
            f"Model types {sorted(invalid)} are not compatible with task_type "
            f"'{task_type}'. Allowed models: {sorted(allowed)}."
        )

    # First apply snapshot-related constraints for models
    model_types = model_types.copy()

    has_snapshots = any(_PAST_SUFFIX_RE.search(col) for col in train_val.columns)
    all_micro = [c for c in train_val.columns if c.startswith("F")]
    has_snapshot_nans = (
        (pd.isna(train_val[all_micro]).values.any() if all_micro else False)
        if has_snapshots
        else False
    )
    if has_snapshots and has_snapshot_nans:
        # Restrict to xgb/xgb_class only when NaNs in snapshot feature tables
        xgb_model = "xgb_class" if task_type == "classification" else "xgb"
        if xgb_model not in model_types:
            print(f"NaNs in snapshot features detected. Using only '{xgb_model}'.")
        else:
            if len(model_types) != 1:
                print(
                    f"NaNs in snapshot features detected. Restricting to "
                    f"'{xgb_model}'."
                )
        model_types = [xgb_model]
    elif has_snapshots and "trac" in model_types:
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
            fully_reproducible=fully_reproducible,
            model_hyperparameters=model_hparams_type,
            optuna_searchspace_sampler=optuna_searchspace_sampler,
            task_type=task_type,
        )
        results_all[model] = result
    return results_all
