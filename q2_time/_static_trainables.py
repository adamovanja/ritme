"""Module with tune trainables of all static models"""
import math
import random

import mlflow
import numpy as np
import tensorflow as tf
import xgboost as xgb
from ray.air import session
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers, models, optimizers

from q2_time._process_train import process_train


def _predict_rmse(model, X, y):
    y_pred = model.predict(X)
    return math.sqrt(mean_squared_error(y, y_pred))


# Linear Regression (for consistency with other training)
def train_linreg(config, train_val, target, host_id, seed_data, seed_model):
    # ! process dataset
    X_train, y_train, X_val, y_val = process_train(
        config, train_val, target, host_id, seed_data
    )

    # ! model
    np.random.seed(seed_model)
    linreg = LinearRegression(fit_intercept=config["fit_intercept"])
    linreg.fit(X_train, y_train)

    score_train = _predict_rmse(linreg, X_train, y_train)
    score_val = _predict_rmse(linreg, X_val, y_val)
    session.report({"rmse_val": score_val, "rmse_train": score_train})


# Define a training function for RandomForest
def train_rf(config, train_val, target, host_id, seed_data, seed_model):
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

    score_train = _predict_rmse(rf, X_train, y_train)
    score_val = _predict_rmse(rf, X_val, y_val)
    session.report({"rmse_val": score_val, "rmse_train": score_train})


# Define a training function for Keras neural network
def train_nn(config, train_val, target, host_id, seed_data, seed_model):
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
    # early_stopping = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    mlflow.tensorflow.autolog()

    # Add TuneReportCallback to report metrics for each epoch
    report_callback = TuneReportCallback(
        # tune: keras
        {"rmse_val": "val_rmse", "rmse_train": "rmse"},
        on="epoch_end",
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=config["batch_size"],
        callbacks=[report_callback],
        verbose=0,
    )


def train_xgb(config, train_val, target, host_id, seed_data, seed_model):
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
    # Initialize the callback
    tune_callback = TuneReportCheckpointCallback(
        # tune:xgboost
        metrics={"rmse_train": "train-rmse", "rmse_val": "val-rmse"},
    )
    # todo: add test here to be tracked as well

    xgb.train(
        config,
        dtrain,
        evals=[(dtrain, "train"), (dval, "val")],
        callbacks=[tune_callback],
    )
