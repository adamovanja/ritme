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
from ray import air, init, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune import ResultGrid
from ray.tune.schedulers import AsyncHyperBandScheduler, HyperBandScheduler
from ray.tune.search.optuna import OptunaSearch

from q2_ritme.model_space import static_searchspace as ss
from q2_ritme.model_space import static_trainables as st

# Set environment variable
os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"

# Constants
MODEL_TRAINABLES = {
    # model_type: trainable
    "xgb": st.train_xgb,
    "nn_reg": st.train_nn_reg,
    "nn_class": st.train_nn_class,
    "nn_corn": st.train_nn_corn,
    "linreg": st.train_linreg,
    "rf": st.train_rf,
    "trac": st.train_trac,
}

DEFAULT_SCHEDULER_GRACE_PERIOD = 10
DEFAULT_SCHEDULER_MAX_T = 100


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
    test_mode: bool,
    model_hyperparameters: dict,
    seed_model: int,
    metric: str,
    mode: str,
):
    # Partial function needed to pass additional parameters
    define_search_space = partial(
        func_to_get_search_space,
        model_type=exp_name,
        tax=tax,
        test_mode=test_mode,
        model_hyperparameters=model_hyperparameters,
    )

    return OptunaSearch(
        space=define_search_space, seed=seed_model, metric=metric, mode=mode
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

    if tracking_uri.endswith("mlruns"):
        if not os.path.exists(tracking_uri):
            os.makedirs(tracking_uri)
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
    test_mode: bool,
    train_val: pd.DataFrame,
    target: str,
    host_id: str,
    seed_data: int,
    seed_model: int,
    tax: pd.DataFrame,
    tree_phylo: skbio.TreeNode,
    path2exp: str,
    num_trials: int,
    max_concurrent_trials: int,
    fully_reproducible: bool = False,
    model_hyperparameters: dict = None,
    scheduler_grace_period: int = DEFAULT_SCHEDULER_GRACE_PERIOD,
    scheduler_max_t: int = DEFAULT_SCHEDULER_MAX_T,
    resources: dict = None,
) -> ResultGrid:
    if model_hyperparameters is None:
        model_hyperparameters = {}

    # Since each trial starts its own threads, this should not be set too high
    max_concurrent_trials = min(num_trials, max_concurrent_trials)
    if resources is None:
        # If not a SLURM process, default values are used
        resources = _get_resources(max_concurrent_trials)

    # Fun facts about trainables and their parallelization/GPU capabilities:
    # - linreg: not parallelizable + CPU-based
    # - trac: solver Path-Alg not parallelized by default + Classo is a
    #   CPU-based library
    # - rf: parallel processing supported but no GPU support
    # - xgb, nn_reg, nn_class, nn_corn: parallel processing supported with GPU support

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
    metric = "rmse_val"
    mode = "min"

    # Define schedulers
    scheduler = _define_scheduler(
        fully_reproducible, scheduler_grace_period, scheduler_max_t
    )

    # Define search algorithm with search space
    search_algo = _define_search_algo(
        ss.get_search_space,
        exp_name,
        tax,
        test_mode,
        model_hyperparameters,
        seed_model,
        metric,
        mode,
    )

    storage_path = os.path.abspath(path2exp)
    experiment_tag = os.path.basename(path2exp)

    callbacks = _define_callbacks(tracking_uri, exp_name, experiment_tag)

    analysis = tune.Tuner(
        # Trainable with input parameters passed and set resources
        tune.with_resources(
            tune.with_parameters(
                trainable,
                train_val=train_val,
                target=target,
                host_id=host_id,
                seed_data=seed_data,
                seed_model=seed_model,
                tax=tax,
                tree_phylo=tree_phylo,
            ),
            resources,
        ),
        # Logging and checkpoint configuration
        run_config=air.RunConfig(
            # Complete experiment name with subfolders of trials within
            name=exp_name,
            storage_path=storage_path,
            # Checkpoint: to store best model - is retrieved in evaluate_models.py
            checkpoint_config=air.CheckpointConfig(
                checkpoint_score_attribute=metric,
                checkpoint_score_order=mode,
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
            num_samples=num_trials,
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
    seed_data: int,
    seed_model: int,
    tax: pd.DataFrame,
    tree_phylo: skbio.TreeNode,
    mlflow_uri: str,
    path_exp: str,
    num_trials: int,
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
    test_mode: bool = False,
    model_hyperparameters: dict = {},
) -> dict[str, ResultGrid]:
    results_all = {}

    # If tax + phylogeny empty, we can't run trac
    model_types = model_types.copy()
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
        else:
            model_hparams_type = model_hyperparameters.get(model, {})

        result = run_trials(
            mlflow_uri,
            model,
            MODEL_TRAINABLES[model],
            test_mode,
            train_val,
            target,
            host_id,
            seed_data,
            seed_model,
            tax,
            tree_phylo,
            path_exp,
            num_trials,
            max_concurrent_trials,
            fully_reproducible=fully_reproducible,
            model_hyperparameters=model_hparams_type,
        )
        results_all[model] = result
    return results_all
