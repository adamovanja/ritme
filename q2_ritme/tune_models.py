import os
import random
import warnings
from functools import partial

import numpy as np
import pandas as pd
import skbio
import torch
from ray import air, init, shutdown, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import AsyncHyperBandScheduler, HyperBandScheduler
from ray.tune.search.optuna import OptunaSearch

from q2_ritme.model_space import static_searchspace as ss
from q2_ritme.model_space import static_trainables as st

os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"

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
    test_mode,
    train_val,
    target,
    host_id,
    seed_data,
    seed_model,
    tax,
    tree_phylo,
    path2exp,
    num_trials,
    fully_reproducible=False,
    scheduler_grace_period=10,
    scheduler_max_t=100,
    resources=None,
):
    # todo: this 10 is imposed by my HPC system - should be made flexible (20
    # could also be possible)
    max_concurrent_trials = 10
    if resources is None:
        # if not a slurm process: default values are used
        all_cpus_avail = get_slurm_resource("SLURM_CPUS_PER_TASK", 1)
        all_gpus_avail = get_slurm_resource("SLURM_GPUS_PER_TASK", 0)
        cpus = int(max(1, (all_cpus_avail // max_concurrent_trials)))
        gpus = max(0, (all_gpus_avail // max_concurrent_trials))
        print(f"Using these resources: CPU {cpus}")
        print(f"Using these resources: GPU {gpus}")
        resources = {
            "cpu": cpus,
            "gpu": gpus,
        }
    # funfacts about trainables and their parallelisation/GPU capabilities:
    # - linreg: not parallelisable + CPU based
    # - trac: solver Path-Alg not parallelized by default + Classo is a
    #   CPU-based library
    # - rf: parallel processing supported but no GPU support
    # - xgb, nn_reg, nn_class, nn_corn: parallel processing supported with GPU
    #   support

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
        metric = "rmse_val"
        mode = "min"
        # AsyncHyperBand enables aggressive early stopping of bad trials.
        # ! efficient & fast BUT
        # ! not fully reproducible with seeds (caused by system load, network
        # ! communication and other factors in env) due to asynchronous mode only
        scheduler = AsyncHyperBandScheduler(
            # stop trials at least this old in time (measured in training iteration)
            grace_period=scheduler_grace_period,
            # stopping trials after max_t iterations have passed
            max_t=scheduler_max_t,
            metric=metric,
            mode=mode,
        )
        # partial function needed to pass additional parameters
        define_search_space = partial(
            ss.get_search_space, model_type=exp_name, tax=tax, test_mode=test_mode
        )
        # define a set of search_space options to evaluate for sure before
        # continuing with optimal search of optuna
        # ! currently it is 225 points which might need to be subsampled
        points_to_evaluate = ss.get_optuna_points_to_evaluate()
        if len(points_to_evaluate) > num_trials:
            raise warnings.Warning(
                f"Number of points to evaluate with optuna is larger than number of "
                f"trials. Not all points can be evaluated. Consider increasing the "
                f"number of trials to at least {len(points_to_evaluate)}."
            )
        search_algo = OptunaSearch(
            space=define_search_space,
            points_to_evaluate=points_to_evaluate,
            metric=metric,
            mode=mode,
            seed=seed_model,
        )

        # todo: potentially add this
        # search_algo = ConcurrencyLimiter(search_algo,
        # max_concurrent=max_concurrent_trials)
    else:
        # ! HyperBandScheduler slower BUT
        # ! improves the reproducibility of experiments by ensuring that all trials
        # ! are evaluated in the same order.
        scheduler = HyperBandScheduler(max_t=scheduler_max_t)

        # BasicVariantGenerator: tries out all hyperparams in search space as
        # they are defined
        search_algo = tune.search.BasicVariantGenerator()
        # ! requires different definition of search space - more like in github
        # commit 2874c70
        raise ValueError("Fully reproducible mode is no longer implemented.")

    storage_path = os.path.abspath(path2exp)
    experiment_tag = os.path.basename(path2exp)
    analysis = tune.Tuner(
        # trainable with input parameters passed and set resources
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
        # mlflow
        run_config=air.RunConfig(
            # complete experiment name with subfolders of trials within
            name=exp_name,
            storage_path=storage_path,
            # ! checkpoint: to store best model - is retrieved in
            # evaluate_models.py
            checkpoint_config=air.CheckpointConfig(
                checkpoint_score_attribute="rmse_val",
                checkpoint_score_order="min",
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
        tune_config=tune.TuneConfig(
            search_alg=search_algo,
            scheduler=scheduler,
            # number of trials to run - schedulers might decide to run more trials
            num_samples=num_trials,
        ),
    )
    # ResultGrid output
    result = analysis.fit()

    # Check all trials & check for error status
    if result.num_errors > 0:
        raise RuntimeError(
            "Some trials encountered errors see above for reported ray tune errors"
        )

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
) -> dict:
    results_all = {}

    # if tax + phylogeny empty we can't run trac
    if (tax.empty or tree_phylo.children == []) and "trac" in model_types:
        model_types.remove("trac")
        print(
            "Removing trac from model_types since no taxonomy and phylogeny were "
            "provided."
        )

    for model in model_types:
        if not os.path.exists(path_exp):
            os.makedirs(path_exp)
        print(f"Ray tune training of: {model}...")
        result = run_trials(
            mlflow_uri,
            model,
            model_trainables[model],
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
            fully_reproducible=fully_reproducible,
        )
        results_all[model] = result
    return results_all
