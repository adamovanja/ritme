import os
import re

import pandas as pd
import typer
from sklearn.metrics import r2_score, root_mean_squared_error

from ritme._decorators import helper_function, main_function
from ritme.evaluate_models import (
    TunedModel,
    get_predictions,
    load_best_model,
    load_experiment_config,
)


# ----------------------------------------------------------------------------
@helper_function
def _predict_w_tuned_model(
    tuned_model: TunedModel,
    exp_config: dict,
    train_val: pd.DataFrame,
    test: pd.DataFrame,
):
    # define
    target = exp_config["target"]
    features = [x for x in train_val.columns if x.startswith("F")]

    # create predictions on train_val and test set - note: ft aggregation,
    # selection and transformation are originally also performed on train_val
    # directly before splitting to train-val for hyperparameter search - so no
    # problem with doing this again here
    train_pred = get_predictions(train_val, tuned_model, target, features, "train")
    test_pred = get_predictions(test, tuned_model, target, features, "test")
    all_pred = pd.concat([train_pred, test_pred])

    return all_pred


@helper_function
def _calculate_metrics(all_preds: pd.DataFrame, model_type: str) -> pd.DataFrame:
    metrics = pd.DataFrame()
    for split in ["train", "test"]:
        pred_split = all_preds[all_preds["split"] == split].copy()

        metrics.loc[model_type, f"rmse_{split}"] = root_mean_squared_error(
            pred_split["true"], pred_split["pred"]
        )
        metrics.loc[model_type, f"r2_{split}"] = r2_score(
            pred_split["true"], pred_split["pred"]
        )
    return metrics


@helper_function
def _load_best_tuned_models(path_to_exp):
    # Get model types that yielded tuned models for this experiment
    pattern = re.compile(r"(.*)_best_model\.pkl$")
    tmodel_files = [
        pattern.match(filename).group(1)
        for filename in os.listdir(path_to_exp)
        if pattern.match(filename)
    ]
    if len(tmodel_files) == 0:
        raise ValueError(
            f"No best tuned models found in {path_to_exp}. "
            "Please run cli_find_best_model_config first or evaluate in the "
            "Python API with the function evaluate_tuned_models."
        )
    return tmodel_files


# ----------------------------------------------------------------------------
@main_function
def evaluate_tuned_models(
    dic_tuned_models: dict[str:TunedModel],
    exp_config: dict,
    train_val: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate tuned models in dic_tuned_models on train_val and test data using
    the target and feature settings defined in exp_config.

    Args:
        dic_tuned_models (dict): Dictionary with model_type as keys and
        TunedModel as values.
        exp_config (dict): Dictionary containing the experiment configuration
        incl. target and feature settings.
        train_val (pd.DataFrame): Train and validation set used to infer best
        tuned models before.
        test (pd.DataFrame): Test set.

    Returns:
        pd.DataFrame: R2 and RMSE metrics for train and test split.
    """
    df_metrics = pd.DataFrame()
    for model_type, model in dic_tuned_models.items():
        # create predictions on train_val & test
        preds_model_type = _predict_w_tuned_model(model, exp_config, train_val, test)

        # calculate metrics for models and append
        metrics_split = _calculate_metrics(preds_model_type, model_type)
        df_metrics = pd.concat([df_metrics, metrics_split])

    # create plots for comparison:
    # TODO: add plot_rmse_over_target plot - à la plot_rmse_over_target_bins
    # TODO: create scatterplot true vs. predicted
    # TODO: create metric plot à la plot_rmse_over_experiments

    return df_metrics


@main_function
def cli_evaluate_tuned_models(
    path_to_exp: str,
    path_to_train_val: str,
    path_to_test: str,
) -> None:
    """
    Evaluate best tuned models in experiment path_to_exp on train_val and test data.

    Args:
        dic_tuned_models (dict): Dictionary with model_type as keys and
        TunedModel as values.
        exp_config (dict): Dictionary containing the experiment configuration
        incl. target and feature settings.
        train_val (str): Path to train and validation set used to infer
        best tuned models before.
        test (str): Path to test set.

    Side Effects:
        Writes the best tuned model evaluation metrics (R2, RMSE) to a file
        "best_metrics.csv" in path_to_exp.
    """
    # ! load data
    train_val = pd.read_pickle(path_to_train_val)
    test = pd.read_pickle(path_to_test)

    # ! load best tuned models of this experiment
    tmodel_files = _load_best_tuned_models(path_to_exp)

    best_model_dict = {}
    for model_type in tmodel_files:
        best_model_dict[model_type] = load_best_model(model_type, path_to_exp)

    # load experiment config
    config = load_experiment_config(path_to_exp)

    # ! calculate metrics
    metrics = evaluate_tuned_models(best_model_dict, config, train_val, test)

    # ! save metrics
    path_to_save = os.path.join(path_to_exp, "best_metrics.csv")
    metrics.to_csv(path_to_save)
    print(f"Metrics were saved in {path_to_save}.")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    typer.run(cli_evaluate_tuned_models)
