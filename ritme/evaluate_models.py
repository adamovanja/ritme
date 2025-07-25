import json
import os
import pickle
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import torch
import xgboost as xgb
from joblib import load
from ray.air.result import Result

from ritme.feature_space._process_trac_specific import _preprocess_taxonomy_aggregation
from ritme.feature_space.aggregate_features import aggregate_microbial_features
from ritme.feature_space.enrich_features import enrich_features
from ritme.feature_space.select_features import select_microbial_features
from ritme.feature_space.transform_features import transform_microbial_features
from ritme.model_space.static_trainables import NeuralNet

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
    """
    checkpoint_dir = result.checkpoint.to_directory()
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
    return checkpoint_path


def load_sklearn_model(result: Result) -> Any:
    """
    Load an sklearn model from a given result object.
    """
    return load(result.metrics["model_path"])


def load_trac_model(result: Result) -> dict:
    """
    Load a TRAC model from a given result object.
    """
    with open(result.metrics["model_path"], "rb") as file:
        model = pickle.load(file)

    return model


def load_xgb_model(result: Result) -> xgb.Booster:
    """
    Load an XGBoost model from a given result object.
    """
    ckpt_path = _get_checkpoint_path(result)
    return xgb.Booster(model_file=ckpt_path)


def load_nn_model(result: Result) -> NeuralNet:
    """
    Load a neural network model from a given result object.
    """
    ckpt_path = _get_checkpoint_path(result)
    model = NeuralNet.load_from_checkpoint(checkpoint_path=ckpt_path)
    return model


def get_model(model_type: str, result: Result) -> Any:
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


def get_taxonomy(result: Result) -> pd.DataFrame:
    """
    Retrieve the taxonomy DataFrame from the result object.
    """
    trial_dir = result.path
    taxonomy_path = os.path.join(trial_dir, "taxonomy.pkl")
    return load(taxonomy_path)


def get_data_processing(result: Result) -> Dict[str, str]:
    """
    Retrieve the data processing configuration from the result object.
    """
    config_data = {k: v for k, v in result.config.items() if k.startswith("data_")}
    return config_data


class TunedModel:
    def __init__(
        self, model: Any, data_config: Dict[str, str], tax: pd.DataFrame, path: str
    ):
        self.model = model
        self.data_config = data_config
        self.tax = tax
        self.path = path
        self.train_selected_fts = []
        self.ft_prefix = "F"
        self.enriched_train_cols = []
        self.enriched_train_other_ft_ls = []

    def aggregate(self, data: pd.DataFrame) -> pd.DataFrame:
        return aggregate_microbial_features(
            data, self.data_config["data_aggregation"], self.tax
        )

    def select(self, data: pd.DataFrame, split: str) -> pd.DataFrame:
        # selection needs to be performed based on train set values -> not on
        # test set!
        if split == "train" and len(self.train_selected_fts) == 0:
            # assign self.train_selected_fts to be able to run select on test set later
            train_selected = select_microbial_features(
                data,
                self.data_config,
                self.ft_prefix,
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
            data[f"{self.ft_prefix}{group_name}"] = data[ft_to_sum].sum(axis=1)

        return data[self.train_selected_fts]

    def transform(self, data: pd.DataFrame) -> Any:
        return transform_microbial_features(
            data,
            self.data_config["data_transform"],
            self.data_config["data_alr_denom_idx"],
        )

    def enrich(
        self,
        data: pd.DataFrame,
        micro_ft_ls: list,
        transformed: pd.DataFrame,
        config: dict,
        split: str = "train",
    ) -> tuple[list, pd.DataFrame]:
        other_ft_ls, enriched = enrich_features(data, micro_ft_ls, transformed, config)
        if split == "train":
            # save columns for test set later on
            self.enriched_train_cols = enriched.columns.tolist()
            self.enriched_train_other_ft_ls = other_ft_ls
        elif len(self.enriched_train_cols) == 0:
            raise ValueError(
                "To run tmodel.enrich on the test set it has to be run on the train "
                "set first."
            )
        else:
            # reindex dummies to match the train set
            enriched = enriched.reindex(
                columns=self.enriched_train_cols, fill_value=0
            ).copy()
            other_ft_ls = self.enriched_train_other_ft_ls
        return other_ft_ls, enriched

    def predict(self, data: pd.DataFrame, split: str) -> Any:
        micro_ft_raw = [x for x in data.columns if x.startswith(self.ft_prefix)]
        # below feature engineering only contains microbial features
        aggregated = self.aggregate(data[micro_ft_raw])
        # selection only possible if it was run on test set before!
        selected = self.select(aggregated, split)
        transformed = self.transform(selected)
        micro_ft_transf = transformed.columns.tolist()
        # enrichment adds metadata to feature table
        # enrichment only possible if it was run on test set before!
        other_ft_ls, enriched = self.enrich(
            data, micro_ft_raw, transformed, self.data_config, split
        )

        X = enriched[micro_ft_transf + other_ft_ls]

        if isinstance(self.model, NeuralNet):
            self.model.eval()
            with torch.no_grad():
                X_t = torch.tensor(X.values, dtype=torch.float32).clone().detach()
                predicted = self.model(X_t)
                predicted = self.model._prepare_predictions(predicted)
        elif isinstance(self.model, dict):
            # trac model
            log_geom, _ = _preprocess_taxonomy_aggregation(
                X.values, self.model["matrix_a"].values
            )
            alpha = self.model["model"].values
            predicted = log_geom.dot(alpha[1:]) + alpha[0]
        elif isinstance(self.model, xgb.core.Booster):
            predicted = self.model.predict(xgb.DMatrix(X)).flatten()
        else:
            predicted = self.model.predict(X.values).flatten()
        return predicted


def retrieve_n_init_best_models(
    result_dic: Dict[str, Result], train_val: pd.DataFrame
) -> Dict[str, TunedModel]:
    """
    Retrieve and initialize the best models from the result dictionary.

    Args:
        result_dic (Dict[str, Result]): Dictionary with model types as keys and
        result grids as values.
        train_val (pd.DataFrame): The training and validation data used to set
        all feature engineering parameters in TunedModel.

    Returns:
        Dict[str, TunedModel]: Dictionary with model types as keys and
        TunedModel instances as values.
    """
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
        # init all model's feature engineering approaches in TunedModel
        _ = best_model_dic[model_type].predict(train_val, "train")
    return best_model_dic


def save_best_models(best_model_dic: Dict[str, TunedModel], output_dir: str) -> None:
    """
    Save the best models to the specified output directory.

    Args:
        best_model_dic (Dict[str, TunedModel]): Dictionary with model types as keys and
        TunedModel instances as values.
        output_dir (str): Directory to save the best models.
    """
    for model_type, tuned_model in best_model_dic.items():
        path_to_save = os.path.join(output_dir, f"{model_type}_best_model.pkl")
        with open(path_to_save, "wb") as file:
            pickle.dump(tuned_model, file)


def load_best_model(model_type: str, path_to_exp: str) -> TunedModel:
    """
    Load the best model of a specified type from the experiment directory.

    Args:
        model_type (str): The type of the model to load.
        path_to_exp (str): The path to the experiment directory.

    Returns:
        TunedModel: The loaded best model.
    """
    file_path = os.path.join(path_to_exp, f"{model_type}_best_model.pkl")
    with open(file_path, "rb") as file:
        return pickle.load(file)


def load_experiment_config(path_to_exp: str) -> Dict[str, Any]:
    """
    Load the experiment configuration from the specified directory.

    Args:
        path_to_exp (str): The path to the experiment directory.

    Returns:
        Dict[str, Any]: The experiment configuration.
    """
    file_path = os.path.join(path_to_exp, "experiment_config.json")
    with open(file_path, "rb") as f:
        return json.load(f)


def get_predictions(
    data: pd.DataFrame,
    tmodel: TunedModel,
    target: str,
    split: str = None,
) -> pd.DataFrame:
    """
    Get predictions from the tuned model for the given data.

    Args:
        data (pd.DataFrame): The input data for prediction.
        tmodel (TunedModel): The tuned model to use for prediction.
        target (str): The target column name in the data.
        split (str, optional): The data split type (e.g., 'train', 'test').
        Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing true and predicted values along with
        the split type.
    """
    # id, true
    saved_pred = data[[target]].copy()
    saved_pred.rename(columns={target: "true"}, inplace=True)
    # pred, split
    saved_pred["pred"] = tmodel.predict(data, split)
    saved_pred["split"] = split
    return saved_pred
