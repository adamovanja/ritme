import argparse
import json
import os
import shutil

import pandas as pd

from q2_ritme.evaluate_models import (
    aggregate_best_models_metrics_and_configs,
    get_predictions,
    plot_best_models_comparison,
    plot_model_training_over_iterations,
    plot_rmse_over_experiments,
    plot_rmse_over_time,
    retrieve_best_models,
)
from q2_ritme.process_data import load_n_split_data
from q2_ritme.tune_models import run_all_trials


def parse_args():
    parser = argparse.ArgumentParser(description="Run configuration.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the run configuration JSON file.",
    )
    return parser.parse_args()


def run_n_eval_tune(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    # ! Define needed paths
    base_path = os.path.join("experiments", "models")

    exp_comparison_output = os.path.join(base_path, config["experiment_tag"])
    if os.path.exists(exp_comparison_output):
        raise ValueError(
            f"This experiment tag already exists: {config['experiment_tag']}."
            "Please use another one."
        )
    if config["tracking_uri"] == "mlruns":
        path_tracker = os.path.join("experiments", config["tracking_uri"])
    elif config["tracking_uri"] == "wandb":
        path_tracker = "wandb"
    else:
        raise ValueError(
            f"Invalid tracking_uri: {config['tracking_uri']}. Must be "
            f"'mlruns' or 'wandb'."
        )
    path_exp = os.path.join(base_path, config["experiment_tag"])

    # ! Load and split data
    train_val, test, tax, tree_phylo = load_n_split_data(
        config["path_to_md"],
        config["path_to_ft"],
        config["path_to_tax"],
        config["path_to_phylo"],
        config["host_id"],
        config["target"],
        config["train_size"],
        config["seed_data"],
    )

    # ! Run all experiments
    result_dic = run_all_trials(
        train_val,
        config["target"],
        config["host_id"],
        config["seed_data"],
        config["seed_model"],
        tax,
        tree_phylo,
        path_tracker,
        path_exp,
        # number of trials to run per model type * grid_search parameters in
        # @_static_searchspace
        config["num_trials"],
        config["max_cuncurrent_trials"],
        model_types=config["ls_model_types"],
        fully_reproducible=False,
        test_mode=config["test_mode"],
    )

    # ! Save run config
    config_output_path = os.path.join(exp_comparison_output, "run_config.json")
    shutil.copy(config_path, config_output_path)

    # ! Evaluate best models of this experiment
    # Eval1: train_val vs. test -> performance
    # TODO: move THIS eval1 part to model training
    best_model_dic = retrieve_best_models(result_dic)
    # todo: allow for more flexibility -> see _process_train.py
    features = [x for x in train_val if x.startswith("F")]

    preds_dic = {}
    for model_type, tmodel in best_model_dic.items():
        train_pred = get_predictions(
            train_val, tmodel, config["target"], features, "train"
        )
        test_pred = get_predictions(test, tmodel, config["target"], features, "test")
        all_pred = pd.concat([train_pred, test_pred])

        # Save all predictions to model file
        path2save = os.path.join(tmodel.path, "predictions.csv")
        all_pred.to_csv(path2save, index=True)
        preds_dic[model_type] = all_pred
    # TODO END: move THIS eval1 part to model training

    # TODO: eval2 could be done post training for particular usecases if not
    # TODO: already possible in WandB/MLflow
    # remove: redundant to model tracker
    plot_rmse_over_experiments(preds_dic, exp_comparison_output)
    # remove here - but keep for age prediction
    plot_rmse_over_time(preds_dic, config["ls_model_types"], exp_comparison_output)

    # Eval2: train vs. val -> performance and config
    # remove: redundant to model tracker
    metrics_all, best_configs = aggregate_best_models_metrics_and_configs(result_dic)
    best_configs.to_csv(
        os.path.join(exp_comparison_output, "best_trial_config.csv"), index=True
    )
    # remove: but keep plotting code for later use
    plot_best_models_comparison(metrics_all, exp_comparison_output)

    # ! Evaluate one model over training iterations
    # remove: redundant to model tracker
    for m in config["models_to_evaluate_separately"]:
        plot_model_training_over_iterations(
            m, result_dic, labels=["data_transform"], save_loc=exp_comparison_output
        )


if __name__ == "__main__":
    args = parse_args()
    run_n_eval_tune(args.config)
