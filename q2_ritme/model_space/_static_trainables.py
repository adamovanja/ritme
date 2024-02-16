"""Module with tune trainables of all static models"""
import json
import os
import random
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from ray import tune
from ray.air import session
from ray.tune.integration.xgboost import TuneReportCheckpointCallback as xgb_cc
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from q2_ritme.feature_space._process_train import process_train


def _predict_rmse(model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the root mean squared error of the model's predictions.

    Parameters:
    model (BaseEstimator): The trained model.
    X (np.ndarray): The input data.
    y (np.ndarray): The target values.

    Returns:
    float: The root mean squared error of the model's predictions.
    """
    y_pred = model.predict(X)
    return mean_squared_error(y, y_pred, squared=False)


def _save_sklearn_model(model: BaseEstimator) -> str:
    """
    Save a Scikit-learn model to a file.

    Parameters:
    model (BaseEstimator): The model to save.

    Returns:
    str: The path to the saved model file.
    """
    model_path = os.path.join(tune.get_trial_dir(), "model.pkl")
    joblib.dump(model, model_path)
    return model_path


def _report_results_manually(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> None:
    """
    Manually report results and model to Ray Tune. This function is used for
    Scikit-learn models which do not have built-in support for Ray Tune.

    Parameters:
    model (BaseEstimator): The trained Scikit-learn model.
    X_train (np.ndarray): The training data.
    y_train (np.ndarray): The training labels.
    X_val (np.ndarray): The validation data.
    y_val (np.ndarray): The validation labels.

    Returns:
    None
    """
    model_path = _save_sklearn_model(model)

    score_train = _predict_rmse(model, X_train, y_train)
    score_val = _predict_rmse(model, X_val, y_val)

    session.report(
        metrics={
            "rmse_val": score_val,
            "rmse_train": score_train,
            "model_path": model_path,
        }
    )
    return None


def train_linreg(
    config: Dict[str, Any],
    train_val: pd.DataFrame,
    target: str,
    host_id: str,
    seed_data: int,
    seed_model: int,
) -> None:
    """
    Train a linear regression model and report the results to Ray Tune.

    Parameters:
    config (Dict[str, Any]): The configuration for the training.
    train_val (DataFrame): The training and validation data.
    target (str): The target variable.
    host_id (str): The host ID.
    seed_data (int): The seed for the data.
    seed_model (int): The seed for the model.

    Returns:
    None
    """
    # ! process dataset: X with features & y with host_id
    X_train, y_train, X_val, y_val = process_train(
        config, train_val, target, host_id, seed_data
    )

    # ! model
    np.random.seed(seed_model)
    linreg = LinearRegression(fit_intercept=config["fit_intercept"])
    linreg.fit(X_train, y_train)

    _report_results_manually(linreg, X_train, y_train, X_val, y_val)


def train_rf(
    config: Dict[str, Any],
    train_val: pd.DataFrame,
    target: str,
    host_id: str,
    seed_data: int,
    seed_model: int,
) -> None:
    """
    Train a random forest model and report the results to Ray Tune.

    Parameters:
    config (Dict[str, Any]): The configuration for the training.
    train_val (DataFrame): The training and validation data.
    target (str): The target variable.
    host_id (str): The host ID.
    seed_data (int): The seed for the data.
    seed_model (int): The seed for the model.

    Returns:
    None
    """
    # ! process dataset
    X_train, y_train, X_val, y_val = process_train(
        config, train_val, target, host_id, seed_data
    )

    # ! model
    # setting seed for scikit library
    np.random.seed(seed_model)
    rf = RandomForestRegressor(
        n_estimators=config["n_estimators"], max_depth=config["max_depth"]
    )
    rf.fit(X_train, y_train)

    _report_results_manually(rf, X_train, y_train, X_val, y_val)


class NeuralNet(nn.Module):
    def __init__(self, n_units):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList()
        n_layers = len(n_units)
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_units[i], n_units[i + 1]))
            if i != len(n_units) - 2:  # No activation after the last layer
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def load_data(X_train, y_train, X_val, y_val, config):
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    return train_loader, val_loader


def _determine_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_nn(
    config: Dict[str, Any],
    train_val: pd.DataFrame,
    target: str,
    host_id: str,
    seed_data: int,
    seed_model: int,
) -> None:
    """
    Train a neural network model and report the results to Ray Tune.

    Parameters:
    config (Dict[str, Any]): The configuration for the training.
    train_val (DataFrame): The training and validation data.
    target (str): The target variable.
    host_id (str): The host ID.
    seed_data (int): The seed for the data.
    seed_model (int): The seed for the model.

    Returns:
    None
    """
    # ! process dataset
    X_train, y_train, X_val, y_val = process_train(
        config, train_val, target, host_id, seed_data
    )

    # ! model
    # set seeds
    torch.manual_seed(seed_model)
    np.random.seed(seed_model)
    random.seed(seed_model)

    # Get cpu or gpu device for training.
    device = _determine_device()
    # print(f"Using {device} device")

    # load data
    train_loader, val_loader = load_data(X_train, y_train, X_val, y_val, config)

    # define neural network
    n_layers = config["n_hidden_layers"]
    n_units = (
        # input layer
        [X_train.shape[1]]
        # hidden layers
        + [config[f"n_units_hl{i}"] for i in range(0, n_layers)]
        # output layer defined by target: cont. regression
        + [1]
    )
    assert len(n_units) == n_layers + 2
    model = NeuralNet(n_units).to(device)
    # print(model)

    # define optimizer and training loss
    train_loss = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])

    # fit model
    for epoch in range(config["epochs"]):
        # Training
        t_rmse = 0
        n_t = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # model output: [batch_size, 1] and target [batch_size] hence
            # using squeeze to match dimensions
            pred = model(inputs).squeeze()
            loss = train_loss(pred, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            t_rmse += torch.sqrt(loss).item()
            n_t += 1
        t_rmse /= n_t

        # Validation
        with torch.no_grad():
            val_loss = 0
            val_rmse = 0
            n = 0
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                pred = model(inputs).squeeze()
                loss = train_loss(pred, targets)
                val_loss += loss.item()
                val_rmse += torch.sqrt(loss).item()
                n += 1
            val_loss /= n
            val_rmse /= n

        # Save checkpoint at last epoch
        if epoch == config["epochs"] - 1:
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint.pth")
                torch.save(model.state_dict(), path)

                # Serialize and save n_units
                n_units_path = os.path.join(checkpoint_dir, "n_units.json")
                with open(n_units_path, "w") as f:
                    json.dump(n_units, f)

        # report results to Ray Tune
        # print(epoch, val_loss, val_rmse)
        tune.report(rmse_val=val_rmse, rmse_train=t_rmse)


def train_xgb(
    config: Dict[str, Any],
    train_val: pd.DataFrame,
    target: str,
    host_id: str,
    seed_data: int,
    seed_model: int,
) -> None:
    """
    Train an XGBoost model and report the results to Ray Tune.

    Parameters:
    config (Dict[str, Any]): The configuration for the training.
    train_val (DataFrame): The training and validation data.
    target (str): The target variable.
    host_id (str): The host ID.
    seed_data (int): The seed for the data.
    seed_model (int): The seed for the model.

    Returns:
    None
    """
    # ! process dataset
    X_train, y_train, X_val, y_val = process_train(
        config, train_val, target, host_id, seed_data
    )
    # Set seeds
    np.random.seed(seed_model)
    random.seed(seed_model)

    # Build input matrices for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # ! model
    # Initialize the checkpoint callback - which is built-in support for xgb
    # models in Ray Tune
    checkpoint_callback = xgb_cc(
        # tune:xgboost
        metrics={"rmse_train": "train-rmse", "rmse_val": "val-rmse"},
        filename="checkpoint",
    )
    # todo: add test set here to be tracked as well

    xgb.train(
        config,
        dtrain,
        evals=[(dtrain, "train"), (dval, "val")],
        callbacks=[checkpoint_callback],
    )
