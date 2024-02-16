import json
import os
from typing import Any

import torch
import xgboost as xgb
from joblib import load
from ray.air.result import Result
from ray.tune.result_grid import ResultGrid

from q2_ritme.model_space._static_trainables import NeuralNet, _determine_device


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
    ckpt_path = _get_checkpoint_path(result) + ".pth"
    # ! n_units is needed - which needs to be saved
    unit_path = ckpt_path.replace("checkpoint.pth", "n_units.json")
    n_units = json.load(open(unit_path, "r"))
    model = NeuralNet(n_units).to(_determine_device())
    model.load_state_dict(torch.load(ckpt_path))

    return model


def get_best_model(model_type: str, result_grid: ResultGrid) -> Any:
    """
    Get the best model of a specified type from the result grid.

    :param model_type: The type of the model to load: "linreg", "rf", "xgb", "nn".
    :param result_grid: The result grid containing the results for each model.
    :return: The best model of the specified type.
    """
    model_loaders = {
        "linreg": load_sklearn_model,
        "rf": load_sklearn_model,
        "xgb": load_xgb_model,
        "nn": load_nn_model,
    }

    best_result = result_grid.get_best_result()
    best_model = model_loaders[model_type](best_result)

    return best_model


def get_best_data_processing(result_grid: ResultGrid) -> str:
    best_result = result_grid.get_best_result()
    config_data = {k: v for k, v in best_result.config.items() if k.startswith("data_")}
    return config_data
