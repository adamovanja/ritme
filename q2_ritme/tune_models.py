import multiprocessing
import os
import random

import numpy as np
import pandas as pd
import skbio
import torch
from ray import air, init, shutdown, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import AsyncHyperBandScheduler, HyperBandScheduler

from q2_ritme.model_space import static_searchspace as ss
from q2_ritme.model_space import static_trainables as st

model_trainables = {
    # model_type: trainable
    "xgb": st.train_xgb,
    "nn_reg": st.train_nn_reg,
    "nn_class": st.train_nn_class,
    "nn_corn": st.train_nn_corn,
    "linreg": st.train_linreg,
    "rf": st.train_rf,
    "trac": st.train_trac,
}


def get_slurm_resource(resource_name, default_value=0):
    try:
        return int(os.environ[resource_name])
    except (KeyError, ValueError):
        return default_value


def run_trials(
    mlflow_tracking_uri,  # MLflow with MLflowLoggerCallback
    exp_name,
    trainable,
    search_space,
    train_val,
    target,
    host_id,
    seed_data,
    seed_model,
    tax,
    tree_phylo,
    path2exp,
    num_trials,
    resources,
    fully_reproducible=False,  # if True hyperband instead of ASHA scheduler is used
    scheduler_grace_period=5,
    scheduler_max_t=100,
):
    if not os.path.exists(mlflow_tracking_uri):
        os.makedirs(mlflow_tracking_uri)

    # set seed for search algorithms/schedulers
    random.seed(seed_model)
    np.random.seed(seed_model)
    torch.manual_seed(seed_model)

    # Initialize Ray with the runtime environment
    shutdown()
    # todo: configure dashboard here - see "ray dashboard set up" online once
    # todo: ray (Q2>Py) is updated
    context = init(include_dashboard=False, ignore_reinit_error=True)
    print(context.dashboard_url)
    # note: both schedulers might decide to run more trials than allocated
    if not fully_reproducible:
        # AsyncHyperBand enables aggressive early stopping of bad trials.
        # ! efficient & fast BUT
        # ! not fully reproducible with seeds (caused by system load, network
        # ! communication and other factors in env) due to asynchronous mode only
        scheduler = AsyncHyperBandScheduler(
            # stop trials at least this old in time (measured in training iteration)
            grace_period=scheduler_grace_period,
            # stopping trials after max_t iterations have passed
            max_t=scheduler_max_t,
        )
    else:
        # ! slower BUT
        # ! improves the reproducibility of experiments by ensuring that all trials
        # ! are evaluated in the same order.
        scheduler = HyperBandScheduler(max_t=scheduler_max_t)

    storage_path = os.path.abspath(path2exp)
    experiment_tag = os.path.basename(path2exp)
    analysis = tune.Tuner(
        # trainable with input parameters passed
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
            resources=resources,
        ),
        # mlflow
        run_config=air.RunConfig(
            # complete experiment name with subfolders of trials within
            name=exp_name,
            local_dir=storage_path,
            # ! checkpoint: to store best model - is retrieved in
            # evaluate_models.py
            checkpoint_config=air.CheckpointConfig(
                checkpoint_score_attribute="rmse_val",
                num_to_keep=3,
            ),
            # ! callback: executing specific tasks (e.g. logging) at specific
            # points in training - used in MLflow browser interface
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri=mlflow_tracking_uri,
                    experiment_name=exp_name,
                    # below would be double saving: local_dir as artifact here
                    # save_artifact=True,
                    tags={"experiment_tag": experiment_tag},
                ),
            ],
        ),
        # hyperparameter space: passes config used in trainables
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="rmse_val",
            mode="min",
            # define the scheduler
            scheduler=scheduler,
            # number of trials to run - schedulers might decide to run more trials
            num_samples=num_trials,
            # ! set seed
            # todo: set advanced search algo -> here default random
            search_alg=tune.search.BasicVariantGenerator(),
        ),
    )
    # ResultGrid output
    return analysis.fit()


def launch_model_trials(
    model,
    path_exp,
    mlflow_uri,
    model_trainables,
    model_search_space,
    train_val,
    target,
    host_id,
    seed_data,
    seed_model,
    tax,
    tree_phylo,
    num_trials,
    resources,
    fully_reproducible,
):
    if not os.path.exists(path_exp):
        os.makedirs(path_exp)

    print(f"Ray tune training of: {model}...")
    result_grid = run_trials(
        mlflow_uri,
        model,
        model_trainables[model],
        model_search_space[model],
        train_val,
        target,
        host_id,
        seed_data,
        seed_model,
        tax,
        tree_phylo,
        path_exp,
        num_trials,
        resources,
        fully_reproducible=fully_reproducible,
    )
    return model, result_grid


def run_all_trials_parallel(
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
) -> dict:
    model_search_space = ss.get_search_space(train_val)

    # if tax + phylogeny empty we can't run trac
    if (tax.empty or tree_phylo.children == []) and "trac" in model_types:
        model_types.remove("trac")
        print(
            "Removing trac from model_types since no taxonomy and phylogeny were "
            "provided."
        )

    # ! define resources
    # per default ray tune uses 1 CPU per trial and all GPU/#trials - but this
    # does not work with slurm.
    # configure such that if not a slurm process: default values are used
    num_cpus_avail = get_slurm_resource("SLURM_CPUS_PER_TASK", 1)
    num_gpus_avail = get_slurm_resource("SLURM_GPUS_PER_TASK", 0)
    # TODO: make gird_search options dependent on number of grid search
    # TODO: occurrences in static_searchspace.py
    nb_grid_search_options = 4
    num_actual_trials = num_trials * len(model_types) * nb_grid_search_options

    # resource per 1 trial
    resources = {
        "cpu": max(1, num_cpus_avail // num_actual_trials),
        "gpu": max(0, num_gpus_avail // num_actual_trials),
    }

    # funfacts about trainables and their parallelisation/GPU capabilities:
    # - linreg: not parallelisable + CPU based
    # - trac: solver Path-Alg not parallelized by default + Classo is a
    #   CPU-based library
    # - rf: parallel processing supported but no GPU support
    # - xgb, nn_reg, nn_class, nn_corn: parallel processing supported with GPU support

    pool = multiprocessing.Pool()
    ls_async_results = []
    # launch models in parallel way
    for model in model_types:
        async_result = pool.apply_async(
            launch_model_trials,
            args=(
                model,
                path_exp,
                mlflow_uri,
                model_trainables,
                model_search_space,
                train_val,
                target,
                host_id,
                seed_data,
                seed_model,
                tax,
                tree_phylo,
                num_trials,
                resources,
                fully_reproducible,
            ),
        )
        ls_async_results.append(async_result)

    pool.close()
    pool.join()

    results_all = {}
    # ls_async_results is a list of AsyncResult objects
    for result in ls_async_results:
        model, result_grid = result.get()
        results_all[model] = result_grid

    return results_all
