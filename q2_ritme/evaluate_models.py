import json
import os
import pickle
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from joblib import load
from ray.air.result import Result
from sklearn.metrics import root_mean_squared_error

from q2_ritme.feature_space._process_trac_specific import (
    _preprocess_taxonomy_aggregation,
)
from q2_ritme.feature_space.aggregate_features import aggregate_microbial_features
from q2_ritme.feature_space.select_features import select_microbial_features
from q2_ritme.feature_space.transform_features import transform_microbial_features
from q2_ritme.model_space.static_trainables import NeuralNet

plt.rcParams.update({"font.family": "DejaVu Sans"})
plt.style.use("seaborn-v0_8-pastel")

# custom color map
color_map = {
    "train": "lightskyblue",
    "test": "peachpuff",
    "rmse_train": "lightskyblue",
    "rmse_val": "plum",
}


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


def load_trac_model(result: Result) -> Any:
    """
    Load a TRAC model from a given result object.

    :param result: The result object containing the model path.
    :return: The loaded TRAC model.
    """
    with open(result.metrics["model_path"], "rb") as file:
        model = pickle.load(file)

    return model


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
        "trac": load_trac_model,
        "rf": load_sklearn_model,
        "xgb": load_xgb_model,
        "nn_reg": load_nn_model,
        "nn_class": load_nn_model,
        "nn_corn": load_nn_model,
    }

    model = model_loaders[model_type](result)

    return model


def get_taxonomy(result) -> pd.DataFrame:
    trial_dir = result.path
    taxonomy_path = os.path.join(trial_dir, "taxonomy.pkl")
    return load(taxonomy_path)


def get_data_processing(result) -> str:
    config_data = {k: v for k, v in result.config.items() if k.startswith("data_")}
    return config_data


class TunedModel:
    def __init__(self, model, data_config, tax, path):
        self.model = model
        self.data_config = data_config
        self.tax = tax
        self.path = path
        self.train_selected_fts = []

    # todo: add refit method!
    def aggregate(self, data):
        return aggregate_microbial_features(
            data, self.data_config["data_aggregation"], self.tax
        )

    def select(self, data, split):
        # selection needs to be performed based on train set values -> not on
        # test set!
        ft_prefix = "F"

        if split == "train":
            # assign self.train_selected_fts to be able to run select on test set later
            train_selected = select_microbial_features(
                data,
                self.data_config,
                ft_prefix,
            )
            self.train_selected_fts = train_selected.columns
            return train_selected
        elif len(self.train_selected_fts) == 0:
            raise ValueError(
                "To run tmodel.predict on the test set it has to be run on the train "
                "set first."
            )
        # SPLIT = "test"
        ft_to_sum = [x for x in data.columns if x not in self.train_selected_fts]
        if len(ft_to_sum) > 0:
            group_name = (
                "_low_abun"
                if self.data_config["data_selection"].startswith("abundance")
                else "_low_var"
            )
            data[f"{ft_prefix}{group_name}"] = data[ft_to_sum].sum(axis=1)

        return data[self.train_selected_fts]

    def transform(self, data):
        transformed = transform_microbial_features(
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

    def predict(self, data, split):
        aggregated = self.aggregate(data)
        # selection only possible if it was run on test set before!
        selected = self.select(aggregated, split)
        transformed = self.transform(selected)
        if isinstance(self.model, NeuralNet):
            with torch.no_grad():
                X_t = torch.tensor(transformed, dtype=torch.float32)
                predicted = self.model(X_t)
                predicted = self.model._prepare_predictions(predicted)
        elif isinstance(self.model, dict):
            # trac model
            log_geom, _ = _preprocess_taxonomy_aggregation(
                transformed, self.model["matrix_a"].values
            )
            alpha = self.model["model"].values
            predicted = log_geom.dot(alpha[1:]) + alpha[0]
        else:
            predicted = self.model.predict(transformed).flatten()
        return predicted


def retrieve_best_models(result_dic):
    best_model_dic = {}
    for model_type, result_grid in result_dic.items():
        best_result = result_grid.get_best_result(scope="all")

        best_model = get_model(model_type, best_result)
        best_data_proc = get_data_processing(best_result)
        best_tax = get_taxonomy(best_result)
        best_path = best_result.path

        best_model_dic[model_type] = TunedModel(
            best_model, best_data_proc, best_tax, best_path
        )
    return best_model_dic


def save_best_models(best_model_dic: dict[str:TunedModel], output_dir: str):
    for model_type, tuned_model in best_model_dic.items():
        path_to_save = os.path.join(output_dir, f"{model_type}_best_model.pkl")
        with open(path_to_save, "wb") as file:
            pickle.dump(tuned_model, file)


def load_best_model(model_type: str, path_to_exp: str) -> TunedModel:
    file_path = os.path.join(path_to_exp, f"{model_type}_best_model.pkl")
    with open(file_path, "rb") as file:
        return pickle.load(file)


def load_experiment_config(path_to_exp: str) -> dict:
    file_path = os.path.join(path_to_exp, "experiment_config.json")

    with open(file_path, "rb") as f:
        return json.load(f)


def get_predictions(data, tmodel, target, features, split=None):
    # id, true
    saved_pred = data[[target]].copy()
    saved_pred.rename(columns={target: "true"}, inplace=True)
    # pred, split
    saved_pred["pred"] = tmodel.predict(data[features], split)
    saved_pred["split"] = split
    return saved_pred


def calculate_rmse(pred_df):
    rmse_scores = {}
    for split in pred_df["split"].unique():
        pred_split = pred_df[pred_df["split"] == split].copy()
        rmse = root_mean_squared_error(
            pred_split["true"].values, pred_split["pred"].values
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
    rmse_df = rmse_df[
        sorted(rmse_df.columns, key=lambda x: 0 if "train" in x else 1)
    ]  # Enforce column order
    rmse_df.plot(
        kind="bar",
        title="Overall",
        ylabel="RMSE",
        color=[color_map.get(col, "gray") for col in rmse_df.columns],
    )
    path_to_save = os.path.join(save_loc, "rmse_over_experiments_train_test.png")
    plt.tight_layout()
    plt.savefig(path_to_save, dpi=dpi)


def plot_rmse_over_time(preds_dic, ls_model_types, save_loc, dpi=300):
    """
    Plot RMSE over true time bins for the first model type in ls_model_types.

    Parameters:
    preds_dic (dict): Dictionary containing predictions for each model type.
    ls_model_types (list): List of model types.
    dpi (int): Resolution of the plot.
    """
    for model_type in ls_model_types:
        pred_df = preds_dic[model_type]
        split = None

        # Bin true columns
        pred_df["group"] = np.round(pred_df["true"], 0).astype(int)

        # Calculate RMSE for each group
        grouped_ser = pred_df.groupby(["group"]).apply(calculate_rmse)
        grouped_df = grouped_ser.apply(pd.Series)
        if split is not None:
            grouped_df = grouped_df[[split]].copy()

        # Enforce column order
        grouped_df = grouped_df[
            sorted(grouped_df.columns, key=lambda x: 0 if "train" in x else 1)
        ]

        # Plot
        plt.figure(dpi=dpi)
        grouped_df.plot(
            kind="bar",
            title=f"Model: {model_type}",
            ylabel="RMSE",
            figsize=(10, 5),
            color=[color_map.get(col, "gray") for col in grouped_df.columns],
        )
        path_to_save = os.path.join(
            save_loc, f"rmse_over_time_train_test_{model_type}.png"
        )
        plt.savefig(path_to_save, dpi=dpi)
