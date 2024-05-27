"""Module with tune trainables of all static models"""

import os
import pickle
import random
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import ray
import skbio
import torch
import xgboost as xgb
from classo import Classo
from coral_pytorch.dataset import corn_label_from_logits
from coral_pytorch.losses import corn_loss
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from ray import tune
from ray.air import session
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.integration.xgboost import TuneReportCheckpointCallback as xgb_cc
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from q2_ritme.feature_space._process_trac_specific import (
    _preprocess_taxonomy_aggregation,
    create_matrix_from_tree,
)
from q2_ritme.feature_space._process_train import process_train
from q2_ritme.model_space._model_trac_calc import min_least_squares_solution


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
    model_path = os.path.join(ray.train.get_context().get_trial_dir(), "model.pkl")
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
    tax: pd.DataFrame = pd.DataFrame(),
    tree_phylo: skbio.TreeNode = skbio.TreeNode(),
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
    X_train, y_train, X_val, y_val, ft_col = process_train(
        config, train_val, target, host_id, seed_data
    )

    # ! model
    np.random.seed(seed_model)
    linreg = ElasticNet(
        alpha=config["alpha"],
        l1_ratio=config["l1_ratio"],
        fit_intercept=config["fit_intercept"],
    )
    linreg.fit(X_train, y_train)

    _report_results_manually(linreg, X_train, y_train, X_val, y_val)


def _predict_rmse_trac(alpha, log_geom_X, y):
    y_pred = log_geom_X.dot(alpha[1:]) + alpha[0]
    return mean_squared_error(y, y_pred, squared=False)


def _report_results_manually_trac(
    alpha, A_df, log_geom_train, y_train, log_geom_val, y_val
):
    # save coefficients w labels & matrix A with labels -> model_path
    idx_alpha = ["intercept"] + A_df.columns.tolist()
    df_alpha_with_labels = pd.DataFrame(alpha, columns=["alpha"], index=idx_alpha)

    model = {"model": df_alpha_with_labels, "matrix_a": A_df}

    # save model
    path_to_save = ray.train.get_context().get_trial_dir()
    model_path = os.path.join(path_to_save, "model.pkl")
    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    # calculate RMSE
    score_train = _predict_rmse_trac(alpha, log_geom_train, y_train)
    score_val = _predict_rmse_trac(alpha, log_geom_val, y_val)

    session.report(
        metrics={
            "rmse_val": score_val,
            "rmse_train": score_train,
            "model_path": model_path,
        }
    )
    return None


def train_trac(
    config: Dict[str, Any],
    train_val: pd.DataFrame,
    target: str,
    host_id: str,
    seed_data: int,
    seed_model: int,
    tax: pd.DataFrame,
    tree_phylo: skbio.TreeNode,
) -> None:
    """
    Train a trac model and report the results to Ray Tune.

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
    X_train, y_train, X_val, y_val, ft_col = process_train(
        config, train_val, target, host_id, seed_data
    )
    # ! derive matrix A
    a_df = create_matrix_from_tree(tree_phylo, tax)

    # ! get log_geom
    log_geom_train, nleaves = _preprocess_taxonomy_aggregation(X_train, a_df.values)
    log_geom_val, _ = _preprocess_taxonomy_aggregation(X_val, a_df.values)

    # ! model
    np.random.seed(seed_model)
    matrices_train = (log_geom_train, np.ones((1, len(log_geom_train[0]))), y_train)
    intercept = True
    alpha_norefit = Classo(
        matrix=matrices_train,
        lam=config["lambda"],
        typ="R1",
        meth="Path-Alg",
        w=1 / nleaves,
        intercept=intercept,
    )
    selected_param = abs(alpha_norefit) > 1e-5
    alpha = min_least_squares_solution(
        matrices_train, selected_param, intercept=intercept
    )

    _report_results_manually_trac(
        alpha, a_df, log_geom_train, y_train, log_geom_val, y_val
    )


def train_rf(
    config: Dict[str, Any],
    train_val: pd.DataFrame,
    target: str,
    host_id: str,
    seed_data: int,
    seed_model: int,
    tax: pd.DataFrame = pd.DataFrame(),
    tree_phylo: skbio.TreeNode = skbio.TreeNode(),
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
    X_train, y_train, X_val, y_val, ft_col = process_train(
        config, train_val, target, host_id, seed_data
    )

    # ! model
    # setting seed for scikit library
    np.random.seed(seed_model)
    rf = RandomForestRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        # ray tune uses joblib for parallelization - so this makes sure all
        # available resources are used
        n_jobs=None,
    )
    rf.fit(X_train, y_train)

    _report_results_manually(rf, X_train, y_train, X_val, y_val)


class NeuralNet(LightningModule):
    # TODO: adjust to have option of NNcorn also within
    def __init__(self, n_units, learning_rate, nn_type="regression"):
        super(NeuralNet, self).__init__()
        self.save_hyperparameters()  # This saves all passed arguments to self.hparams
        self.layers = nn.ModuleList()
        n_layers = len(n_units)
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_units[i], n_units[i + 1]))
            if i != len(n_units) - 2:  # No activation after the last layer
                self.layers.append(nn.ReLU())
        self.learning_rate = learning_rate
        self.nn_type = nn_type
        self.num_classes = n_units[-1]
        if self.nn_type == "regression":
            self.loss_fn = nn.MSELoss()
        elif self.nn_type == "classification":
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.nn_type == "ordinal_regression":
            self.loss_fn = None

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def _prepare_predictions(self, predictions):
        # todo: add ordinal regression option
        if self.nn_type == "regression":
            return predictions
        elif self.nn_type == "classification":
            return torch.argmax(predictions, dim=1).float()
        elif self.nn_type == "ordinal_regression":
            return corn_label_from_logits(predictions).float()

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs).squeeze()

        # loss: corn_loss, cross-entropy or mse
        # todo: clean up and remove redundancy with val step
        if self.nn_type == "ordinal_regression":
            loss = corn_loss(predictions, targets, self.num_classes)
            # loss = self.loss_fn(predictions, targets, self.num_classes)
        else:
            loss = self.loss_fn(predictions, targets)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        # rmse
        pred_rmse = self._prepare_predictions(predictions)
        rmse = torch.sqrt(nn.functional.mse_loss(pred_rmse, targets))
        self.log(
            "train_rmse", rmse, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs).squeeze()

        # loss: corn_loss, cross-entropy or mse
        if self.nn_type == "ordinal_regression":
            loss = corn_loss(predictions, targets, self.num_classes)
            # loss = self.loss_fn(predictions, targets, self.num_classes)
        else:
            loss = self.loss_fn(predictions, targets)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        # rmse
        pred_rmse = self._prepare_predictions(predictions)
        rmse = torch.sqrt(nn.functional.mse_loss(pred_rmse, targets))
        self.log(
            "val_rmse", rmse, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def load_data(X_train, y_train, X_val, y_val, y_type, config):
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=y_type),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=y_type),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=3
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=3)
    return train_loader, val_loader


def train_nn(
    config,
    train_val,
    target,
    host_id,
    seed_data,
    seed_model,
    nn_type="regression",
):
    # Set the seed for reproducibility
    seed_everything(seed_model, workers=True)

    # Process dataset
    X_train, y_train, X_val, y_val, ft_col = process_train(
        config, train_val, target, host_id, seed_data
    )

    # round target to monthly classes in case of ordinal regression and load
    # data with data loaders
    if nn_type in ["ordinal_regression", "classification"]:
        y_train = np.round(y_train)
        y_val = np.round(y_val)
        train_loader, val_loader = load_data(
            X_train, y_train, X_val, y_val, torch.long, config
        )

    else:
        train_loader, val_loader = load_data(
            X_train, y_train, X_val, y_val, torch.float32, config
        )

    # Model
    n_layers = config["n_hidden_layers"]
    # output layer defined by target
    n_target_classes = len(np.unique(y_train))
    if nn_type == "regression":
        output_layer = [1]
    elif nn_type == "classification":
        output_layer = [n_target_classes]
    elif nn_type == "ordinal_regression":
        # CORN reduces number of classes by 1
        output_layer = [n_target_classes - 1]

    n_units = (
        # input layer
        [X_train.shape[1]]
        # hidden layers
        + [config[f"n_units_hl{i}"] for i in range(0, n_layers)]
        # output layer defined by nn_type
        + output_layer
    )
    assert len(n_units) == n_layers + 2
    model = NeuralNet(
        n_units=n_units, learning_rate=config["learning_rate"], nn_type=nn_type
    )

    # Callbacks
    checkpoint_dir = (
        tune.get_trial_dir() if tune.is_session_enabled() else "checkpoints"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            monitor="val_rmse",
            mode="min",
            save_top_k=1,
            save_weights_only=False,
            dirpath=checkpoint_dir,  # Automatically set dirpath
            filename="{epoch}-{val_rmse:.2f}",
        ),
        TuneReportCheckpointCallback(
            metrics={
                "rmse_val": "val_rmse",
                "rmse_train": "train_rmse",
                "loss_val": "val_loss",
                "loss_train": "train_loss",
            },
            filename="checkpoint",
            on="validation_end",
        ),
    ]

    # Trainer
    trainer = Trainer(
        max_epochs=config["epochs"],
        callbacks=callbacks,
        deterministic=True,
        enable_progress_bar=False,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def train_nn_reg(
    config, train_val, target, host_id, seed_data, seed_model, tax, tree_phylo
):
    train_nn(
        config, train_val, target, host_id, seed_data, seed_model, nn_type="regression"
    )


def train_nn_class(
    config, train_val, target, host_id, seed_data, seed_model, tax, tree_phylo
):
    train_nn(
        config,
        train_val,
        target,
        host_id,
        seed_data,
        seed_model,
        nn_type="classification",
    )


def train_nn_corn(
    config, train_val, target, host_id, seed_data, seed_model, tax, tree_phylo
):
    # corn model from https://github.com/Raschka-research-group/coral-pytorch
    train_nn(
        config,
        train_val,
        target,
        host_id,
        seed_data,
        seed_model,
        nn_type="ordinal_regression",
    )


def train_xgb(
    config: Dict[str, Any],
    train_val: pd.DataFrame,
    target: str,
    host_id: str,
    seed_data: int,
    seed_model: int,
    tax: pd.DataFrame = pd.DataFrame(),
    tree_phylo: skbio.TreeNode = skbio.TreeNode(),
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
    X_train, y_train, X_val, y_val, ft_col = process_train(
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
    # nthread defaults to max available threads available if not set
    xgb.train(
        config,
        dtrain,
        evals=[(dtrain, "train"), (dval, "val")],
        callbacks=[checkpoint_callback],
    )