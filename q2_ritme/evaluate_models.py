import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from joblib import load
from ray.air.result import Result
from sklearn.metrics import mean_squared_error

from q2_ritme.feature_space.transform_features import transform_features
from q2_ritme.model_space._static_trainables import NeuralNet

# 30.437 is avg. number of days per month
DAYS_PER_MONTH = 30.437


def _get_checkpoint_path(result: Result) -> str:
    """
    Get the checkpoint path for a given result.

    :param result: The result object containing the checkpoint information.
    :return: The checkpoint path as a string.
    """
    checkpoint_dir = result.checkpoint.to_directory()
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
    return checkpoint_path


def load_sklearn_model(result: Result) -> Any:
    """
    Load an sklearn model from a given result object.

    :param result: The result object containing the model path.
    :return: The loaded sklearn model.
    """
    return load(result.metrics["model_path"])


def load_xgb_model(result: Result) -> xgb.Booster:
    """
    Load an XGBoost model from a given result object.

    :param result: The result object containing the checkpoint information.
    :return: The loaded XGBoost model.
    """
    ckpt_path = _get_checkpoint_path(result)
    return xgb.Booster(model_file=ckpt_path)


def load_nn_model(result: Result) -> Any:
    """
    Load a neural network model from a given result object.

    :param result: The result object containing the checkpoint information.
    :return: The loaded neural network model.
    """
    ckpt_path = _get_checkpoint_path(result)
    model = NeuralNet.load_from_checkpoint(checkpoint_path=ckpt_path)
    return model


def get_model(model_type: str, result) -> Any:
    """
    Get the model of a specified type from the result object.
    """
    model_loaders = {
        "linreg": load_sklearn_model,
        "rf": load_sklearn_model,
        "xgb": load_xgb_model,
        "nn": load_nn_model,
    }

    model = model_loaders[model_type](result)

    return model


def get_data_processing(result) -> str:
    config_data = {k: v for k, v in result.config.items() if k.startswith("data_")}
    return config_data


class TunedModel:
    def __init__(self, model, data_config, path):
        self.model = model
        self.data_config = data_config
        self.path = path

    def transform(self, data):
        transformed = transform_features(
            data,
            self.data_config["data_transform"],
            self.data_config["data_alr_denom_idx"],
        )
        if isinstance(self.model, xgb.core.Booster):
            return xgb.DMatrix(transformed)
        elif isinstance(self.model, NeuralNet):
            return torch.tensor(transformed.values, dtype=torch.float32)
        else:
            return transformed.values

    def predict(self, data):
        transformed = self.transform(data)
        if isinstance(self.model, NeuralNet):
            with torch.no_grad():
                predicted = self.model(transformed).numpy()
        else:
            predicted = self.model.predict(transformed)
        return predicted.flatten()


def retrieve_best_models(result_dic):
    best_model_dic = {}
    for model_type, result_grid in result_dic.items():
        best_result = result_grid.get_best_result()

        best_model = get_model(model_type, best_result)
        best_data_proc = get_data_processing(best_result)
        best_path = best_result.path

        best_model_dic[model_type] = TunedModel(best_model, best_data_proc, best_path)
    return best_model_dic


def get_predictions(data, tmodel, target, features, split=None):
    # id, true
    saved_pred = data[[target]].copy()
    saved_pred.rename(columns={target: "true"}, inplace=True)
    # pred, split
    saved_pred["pred"] = tmodel.predict(data[features])
    saved_pred["split"] = split
    # todo: actually save the performed predictions into best_models folder
    return saved_pred


def calculate_rmse(pred_df):
    rmse_scores = {}
    for split in pred_df["split"].unique():
        pred_split = pred_df[pred_df["split"] == split].copy()
        rmse = mean_squared_error(
            pred_split["true"].values, pred_split["pred"].values, squared=False
        )
        rmse_scores[split] = rmse
    return rmse_scores


def plot_rmse_over_experiments(preds_dic, save_loc, dpi=400):
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    rmse_dic = {}
    for model_type, pred_df in preds_dic.items():
        rmse_dic[model_type] = calculate_rmse(pred_df)

    plt.figure(dpi=dpi)  # Increase the resolution by setting a higher dpi
    rmse_df = pd.DataFrame(rmse_dic).T
    rmse_df.plot(kind="bar", title="Overall", ylabel="RMSE")
    path_to_save = os.path.join(save_loc, "rmse_over_experiments_train_test.png")
    plt.tight_layout()
    plt.savefig(path_to_save, dpi=dpi)
    plt.show()


def plot_rmse_over_time(
    preds_dic, ls_model_types, save_loc, days_per_month=DAYS_PER_MONTH, dpi=300
):
    """
    Plot RMSE over true time bins for the first model type in ls_model_types.

    Parameters:
    preds_dic (dict): Dictionary containing predictions for each model type.
    ls_model_types (list): List of model types.
    days_per_month (float): Average number of days per month to use for binning.
    dpi (int): Resolution of the plot.
    """
    for model_type in ls_model_types:
        pred_df = preds_dic[model_type]
        split = None

        # Bin true columns by months
        pred_df["group"] = np.round(pred_df["true"] / days_per_month, 0).astype(int)

        # Calculate RMSE for each group
        grouped_ser = pred_df.groupby(["group"]).apply(calculate_rmse)
        grouped_df = grouped_ser.apply(pd.Series)
        if split is not None:
            grouped_df = grouped_df[[split]].copy()

        # Plot
        plt.figure(dpi=dpi)
        grouped_df.plot(
            kind="bar", title=f"Model: {model_type}", ylabel="RMSE", figsize=(10, 5)
        )
        path_to_save = os.path.join(
            save_loc, f"rmse_over_time_train_test_{model_type}.png"
        )
        plt.savefig(path_to_save, dpi=dpi)
        plt.show()


def get_best_model_metrics_and_config(
    trial_result, metric_ls=["rmse_train", "rmse_val"]
):
    """
    Retrieve metrics and configuration for the best model from a trial result.

    Parameters:
    trial_result (ResultGrid): The result grid from a Ray Tune trial.
    metric_ls (list): List of metrics to retrieve.

    Returns:
    tuple: A tuple containing a DataFrame of metrics and a dictionary of the
    best model's configuration.
    """
    best_result = trial_result.get_best_result()
    config = best_result.config
    metrics_ser = best_result.metrics_dataframe[metric_ls].iloc[-1]
    metrics_df = pd.DataFrame(metrics_ser).transpose()
    return metrics_df, config


def aggregate_best_models_metrics_and_configs(dic_trials):
    """
    Aggregate metrics and configurations of best models from multiple trials.

    Parameters:
    dic_trials (dict): Dictionary of trials.

    Returns:
    tuple: A tuple containing a DataFrame of aggregated metrics and a DataFrame
    of configurations.
    """
    df_metrics = pd.DataFrame()
    dic_config = {}
    for key, value in dic_trials.items():
        df_best, config = get_best_model_metrics_and_config(value)
        df_metrics = pd.concat(
            [df_metrics, df_best.rename(index={df_best.index[0]: key})]
        )
        dic_config[key] = config
    return df_metrics, pd.DataFrame(dic_config)


def plot_best_models_comparison(
    df_metrics, save_loc, title="Model Performance Comparison: train vs. val"
):
    """
    Plot a comparison of best models based on their metrics.

    Parameters:
    df_metrics (pd.DataFrame): DataFrame containing the metrics of the best models.
    title (str): Title of the plot.
    dpi (int): Resolution of the plot.
    """
    plt.style.use("seaborn-v0_8-colorblind")
    df2plot = df_metrics.sort_values(by="rmse_val", ascending=True)
    df2plot.plot(kind="bar", figsize=(12, 6))
    plt.xticks(rotation=45)
    plt.ylabel("RMSE")
    plt.xlabel("Model type (order: increasing val score)")
    plt.title(title)
    plt.tight_layout()
    path_to_save = os.path.join(save_loc, "rmse_over_experiments_train_val.png")
    plt.savefig(path_to_save, dpi=400)
    plt.show()


def plot_model_training_over_iterations(model_type, result_dic, labels, save_loc):
    ax = None
    for result in result_dic[model_type]:
        label_str = ""
        for lab in labels:
            label_str += f"{lab}={result.config[lab]}, "
        if ax is None:
            ax = result.metrics_dataframe.plot(
                "training_iteration", "rmse_val", label=label_str
            )
        else:
            result.metrics_dataframe.plot(
                "training_iteration", "rmse_val", ax=ax, label=label_str
            )
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    ax.set_title(f"RMSE_val vs. training iteration for all trials of {model_type}")
    ax.set_ylabel("RMSE_val")
    plt.tight_layout()
    path_to_save = os.path.join(
        save_loc, f"rmse_best_{model_type}_over_training_iteration.png"
    )
    plt.savefig(path_to_save, dpi=400)
