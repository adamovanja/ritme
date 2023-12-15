"""Module with tune trainables of all static models"""
import os
import random
from typing import Any, Dict

import joblib
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from ray import tune
from ray.air import session
from ray.tune.integration.keras import TuneReportCheckpointCallback as k_cc
from ray.tune.integration.xgboost import TuneReportCheckpointCallback as xgb_cc
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers, models, optimizers

from q2_time.feature_space._process_train import process_train


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
    random.seed(seed_model)
    np.random.seed(seed_model)
    tf.random.set_seed(seed_model)
    tf.compat.v1.set_random_seed(seed_model)

    # define neural network
    model = models.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))

    n_layers = config["n_layers"]
    for i in range(n_layers):
        num_hidden = config[f"n_units_l{i}"]
        model.add(layers.Dense(num_hidden, activation="relu"))

    model.add(layers.Dense(1))

    # define learning
    learning_rate = config["learning_rate"]
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    # todo: reconsider adding early stopping
    # early_stopping = callbacks.EarlyStopping(
    #   patience=10, restore_best_weights=True)
    mlflow.tensorflow.autolog()

    # Add TuneReportCallback to report metrics for each epoch - which is
    # built-in support for keras models in Ray Tune
    checkpoint_callback = k_cc(
        # tune: keras
        {"rmse_val": "val_rmse", "rmse_train": "rmse"},
        on="epoch_end",
        filename="checkpoint",
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=config["batch_size"],
        callbacks=[checkpoint_callback],
        verbose=0,
    )


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
