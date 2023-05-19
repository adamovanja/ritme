import random

import numpy as np
import tensorflow as tf
from ray import air, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import AsyncHyperBandScheduler, HyperBandScheduler


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
    fully_reproducible=False,  # if True hyperband instead of ASHA scheduler is used
    num_trials=2,  # todo: increase default num_trials
    scheduler_grace_period=5,
    scheduler_max_t=100,
    resources={"cpu": 1},
):
    # set seed for search algorithms/schedulers
    random.seed(seed_model)
    np.random.seed(seed_model)
    tf.random.set_seed(seed_model)

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
            ),
            resources,
        ),
        # mlflow
        run_config=air.RunConfig(
            # complete experiment name with subfolders of trials within
            name=exp_name,
            local_dir="ray_results",
            # checkpoint: to store best model
            checkpoint_config=air.CheckpointConfig(
                checkpoint_score_attribute="rmse_val",
                num_to_keep=3,
            ),
            # callback: executing specific tasks (e.g. logging)
            # at specific points in training
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri=mlflow_tracking_uri,
                    experiment_name=exp_name,
                    # # todo: double saving: Tune local_dir as artifact here
                    # save_artifact=True,
                ),
            ],
        ),
        # hyperparameter space
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="rmse_val",
            mode="min",
            # define the scheduler
            scheduler=scheduler,
            # number of trials to run
            num_samples=num_trials,
            # ! set seed
            search_alg=tune.search.BasicVariantGenerator(),
        ),
    )
    # ResultGrid output
    return analysis.fit()


# def tune_models(run_config):

#     train_val, test = load_n_split_data()

#     results_xgb = run_trials(
#     MLFLOW_TRACKING_URI,
#     "xgb",
#     st.train_xgb,
#     ss.xgb_space,
#     train_val,
#     TARGET,
#     HOST_ID,
#     SEED_DATA,
#     SEED_MODEL,
#     fully_reproducible=False,
#     )
