import glob
import os
from pathlib import Path

import pandas as pd
from ray.tune.analysis import ExperimentAnalysis

from q2_ritme.evaluate_models import plot_rmse_over_experiments


def _min_comparison(current, best):
    return current < best


def _max_comparison(current, best):
    return current > best


def best_trial_name(analyses_ls, metric_to_evaluate, mode="min"):
    # note: experiment > trial
    best_trial_overall = None

    if mode == "min":
        best_metric = float("inf")
        comparison_operator = _min_comparison
    else:
        best_metric = -float("inf")
        comparison_operator = _max_comparison

    for analysis in analyses_ls:
        # Get the best trial for the current analysis based on the metric
        best_trial = analysis.get_best_trial(metric_to_evaluate, mode, "last")

        # Retrieve the best metric for this trial
        best_trial_metric = best_trial.metric_analysis[metric_to_evaluate][mode]

        # Update the overall best trial if this trial has a better "trial_metric"
        if comparison_operator(best_trial_metric, best_metric):
            best_trial_overall = best_trial
            best_metric = best_trial_metric

    return best_trial_overall


def get_all_exp_analyses(experiment_dir):
    state_files = glob.glob(os.path.join(experiment_dir, "experiment_state-*.json"))
    analyses_ls = []
    for f in state_files:
        absolute_path = os.path.abspath(f)
        analyses_ls.append(ExperimentAnalysis(experiment_checkpoint_path=absolute_path))
    return analyses_ls


def read_predictions_for_trial(trial_tag, path_to_models):
    # read predictions for this trial
    base_path = Path(path_to_models)
    target_path = [p for p in base_path.rglob(f"{trial_tag}*") if p.is_dir()]

    if len(target_path) > 1:
        raise ValueError(f"Retrieved experiment tag is not unique: {trial_tag}")
    rel_path = target_path[0]
    path_to_pred = os.path.join(rel_path, "predictions.csv")

    return pd.read_csv(path_to_pred, index_col=0)


def verify_indices(models_dict, pred_value):
    # Extract indices for the specified split ("train" or "test") from each
    # DataFrame
    indices = [df[df["split"] == pred_value].index for df in models_dict.values()]

    # Use all() to check if all indices are equal by comparing each to the first
    return all(index.equals(indices[0]) for index in indices)


def compare_trials(dic_trials_to_check, path_to_models, path_to_save):
    # get predictions for each best trial
    pred_dic = {}
    config_dic = {}
    for v in dic_trials_to_check.values():
        pred_dic[v] = read_predictions_for_trial(v, path_to_models)
        config_dic[v] = v.config

    # Verify that IDs are identical for both splits
    for split in ["train", "test"]:
        idx_identical = verify_indices(pred_dic, split)
        if not idx_identical:
            raise ValueError(f"Indices for {split} are not identical")

    # display RMSE overall best trials
    plot_rmse_over_experiments(pred_dic, path_to_save)

    # display config differences
    config_df = pd.DataFrame(config_dic)
    config_df.to_csv(os.path.join(path_to_save, "best_trial_config.csv"), index=True)
