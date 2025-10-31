"""Module with tune trainables of all static models"""

import os
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import ray
import skbio
import torch
import torchmetrics
import xgboost as xgb
from classo import Classo
from coral_pytorch.dataset import corn_label_from_logits
from coral_pytorch.losses import corn_loss
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from ray import train
from ray.air import session
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.integration.xgboost import TuneReportCheckpointCallback as xgb_cc
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ritme.feature_space._process_trac_specific import (
    _preprocess_taxonomy_aggregation,
    create_matrix_from_tree,
)
from ritme.feature_space._process_train import process_train
from ritme.model_space._model_trac_calc import min_least_squares_solution


def _predict_rmse_r2(model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Compute the root mean squared error and R2 score of the model's predictions.

    Parameters:
    model (BaseEstimator): The trained model.
    X (np.ndarray): The input data.
    y (np.ndarray): The target values.

    Returns:
    tuple: The root mean squared error and R2 score of the model's predictions.
    """
    y_pred = model.predict(X)
    return root_mean_squared_error(y, y_pred), r2_score(y, y_pred)


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


def _save_taxonomy(tax: pd.DataFrame) -> None:
    taxonomy_path = os.path.join(
        ray.train.get_context().get_trial_dir(), "taxonomy.pkl"
    )
    joblib.dump(tax, taxonomy_path)


def _report_results_manually(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    tax: pd.DataFrame,
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

    _save_taxonomy(tax)

    rmse_train, r2_train = _predict_rmse_r2(model, X_train, y_train)
    rmse_val, r2_val = _predict_rmse_r2(model, X_val, y_val)

    session.report(
        metrics={
            "rmse_val": rmse_val,
            "rmse_train": rmse_train,
            "r2_val": r2_val,
            "r2_train": r2_train,
            "model_path": model_path,
            "nb_features": X_train.shape[1],
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
    X_train, y_train, X_val, y_val = process_train(
        config, train_val, target, host_id, tax, seed_data
    )

    # ! model
    np.random.seed(seed_model)
    linreg = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "linreg",
                ElasticNet(
                    alpha=config["alpha"],
                    l1_ratio=config["l1_ratio"],
                    fit_intercept=True,
                ),
            ),
        ]
    )
    linreg.fit(X_train, y_train)

    _report_results_manually(linreg, X_train, y_train, X_val, y_val, tax)


def _predict_rmse_r2_trac(alpha, log_geom_X, y):
    y_pred = log_geom_X.dot(alpha[1:]) + alpha[0]
    return root_mean_squared_error(y, y_pred), r2_score(y, y_pred)


def _bundle_trac_model(alpha, A_df):
    # get coefficients w labels & matrix A with labels
    idx_alpha = ["intercept"] + A_df.columns.tolist()
    df_alpha_with_labels = pd.DataFrame(alpha, columns=["alpha"], index=idx_alpha)

    model = {"model": df_alpha_with_labels, "matrix_a": A_df}
    return model


def _report_results_manually_trac(
    model, log_geom_train, y_train, log_geom_val, y_val, tax
):
    # save model in a compressed way
    path_to_save = ray.train.get_context().get_trial_dir()
    model_path = os.path.join(path_to_save, "model.pkl")
    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    # calculate RMSE and R2
    df_alpha_with_labels = model["model"]
    alpha = model["model"]["alpha"].values
    rmse_train, r2_train = _predict_rmse_r2_trac(alpha, log_geom_train, y_train)
    rmse_val, r2_val = _predict_rmse_r2_trac(alpha, log_geom_val, y_val)

    # taxonomy
    _save_taxonomy(tax)
    session.report(
        metrics={
            "rmse_val": rmse_val,
            "rmse_train": rmse_train,
            "r2_val": r2_val,
            "r2_train": r2_train,
            "model_path": model_path,
            "nb_features": df_alpha_with_labels[
                df_alpha_with_labels["alpha"] != 0.0
            ].shape[0],
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
    X_train, y_train, X_val, y_val = process_train(
        config, train_val, target, host_id, tax, seed_data
    )
    # ! derive matrix A
    # todo: adjust A_df here already
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

    model = _bundle_trac_model(alpha, a_df)

    _report_results_manually_trac(
        model, log_geom_train, y_train, log_geom_val, y_val, tax
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
    X_train, y_train, X_val, y_val = process_train(
        config, train_val, target, host_id, tax, seed_data
    )

    # ! model
    # setting seed for scikit library
    np.random.seed(seed_model)
    rf = RandomForestRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"],
        min_weight_fraction_leaf=config["min_weight_fraction_leaf"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        min_impurity_decrease=config["min_impurity_decrease"],
        bootstrap=config["bootstrap"],
        # ray tune uses joblib for parallelization - so this makes sure all
        # available resources are used
        n_jobs=None,
        # set randomness
        random_state=seed_model,
    )
    rf.fit(X_train, y_train)

    _report_results_manually(rf, X_train, y_train, X_val, y_val, tax)


class NeuralNet(LightningModule):
    def __init__(
        self,
        n_units,
        learning_rate,
        nn_type="regression",
        dropout_rate=0.0,
        weight_decay=0.0,
        classes: Optional[list] = None,
    ):
        super(NeuralNet, self).__init__()
        self.save_hyperparameters()  # This saves all passed arguments to self.hparams
        self.learning_rate = learning_rate
        self.nn_type = nn_type
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay

        self.input_norm = nn.BatchNorm1d(n_units[0])

        self.classes = classes
        if nn_type in ["classification", "ordinal_regression"]:
            self.class_to_index = {c: i for i, c in enumerate(classes)}
            self.index_to_class = {i: c for i, c in enumerate(classes)}
            self.num_classes = len(classes)
        self.layers = nn.ModuleList()
        n_layers = len(n_units)
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_units[i], n_units[i + 1]))
            if i != len(n_units) - 2:  # No activation after the last layer
                self.layers.append(nn.ReLU())
                if self.dropout_rate > 0:
                    self.layers.append(nn.Dropout(self.dropout_rate))

        self.train_loss = 0
        self.val_loss = 0
        self.train_predictions = []
        self.train_targets = []
        self.validation_predictions = []
        self.validation_targets = []

    def forward(self, x):
        x = self.input_norm(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def _prepare_predictions(self, predictions):
        if self.nn_type == "regression":
            return predictions
        elif self.nn_type == "classification":
            idx = torch.argmax(predictions, dim=1)
            # map back to original labels
            mapped = [self.index_to_class[int(i)] for i in idx.detach().cpu().numpy()]
            return torch.tensor(mapped, device=predictions.device, dtype=torch.float32)
        elif self.nn_type == "ordinal_regression":
            if predictions.ndim == 1:
                # [num_samples] must be [num_samples, 1] for corn
                predictions = predictions.unsqueeze(1)
            corn_label = corn_label_from_logits(predictions).float()
            # map back to original labels
            mapped = [
                self.index_to_class[int(i)] for i in corn_label.detach().cpu().numpy()
            ]
            return torch.tensor(mapped, device=predictions.device, dtype=torch.float32)

    def _calculate_metrics(self, predictions, targets):
        preds = self._prepare_predictions(predictions)

        rmse = torch.sqrt(nn.functional.mse_loss(preds, targets))

        r2score = torchmetrics.regression.R2Score()
        r2 = r2score(preds, targets)

        return rmse, r2

    def _calculate_loss(self, predictions, targets):
        # loss: corn_loss, cross-entropy or mse
        # calculated on rounded classes as targets for ordinal regression and
        # classification
        targets_rounded = torch.round(targets).long()
        if self.nn_type == "ordinal_regression":
            # re-index to 0...C-1 for loss
            t = targets_rounded.detach().cpu().numpy().astype(int)
            t_idx = [self.class_to_index[v] for v in t]
            targets_rounded = torch.tensor(
                t_idx, device=targets.device, dtype=torch.long
            )
            # predictions = logits
            if predictions.ndim == 1:
                # [num_samples] must be [num_samples, 1] for corn_loss
                predictions = predictions.unsqueeze(1)
            return corn_loss(predictions, targets_rounded, self.num_classes)
        elif self.nn_type == "classification":
            # re-index to 0...C-1 for cross-entropy loss
            t = targets_rounded.detach().cpu().numpy().astype(int)
            t_idx = [self.class_to_index[v] for v in t]
            targets_rounded = torch.tensor(
                t_idx, device=targets.device, dtype=torch.long
            )

            loss_fn = nn.CrossEntropyLoss()
            # predictions = logits
            return loss_fn(predictions, targets_rounded)
        loss_fn = nn.MSELoss()
        return loss_fn(predictions, targets)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self.forward(inputs).squeeze()

        # Store predictions and targets
        self.train_predictions.append(predictions.detach())
        self.train_targets.append(targets.detach())

        self.train_loss = self._calculate_loss(predictions, targets)

        return self.train_loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self.forward(inputs).squeeze()

        self.validation_predictions.append(predictions.detach())
        self.validation_targets.append(targets.detach())

        self.val_loss = self._calculate_loss(predictions, targets)
        # nb_features log
        self.log("nb_features", inputs.shape[1])
        return {"val_loss": self.val_loss}

    def on_train_epoch_end(self):
        all_preds = torch.cat(self.train_predictions)
        all_targets = torch.cat(self.train_targets)
        loss = self._calculate_loss(all_preds, all_targets)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        rmse, r2 = self._calculate_metrics(all_preds, all_targets)
        self.log("train_rmse", rmse, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_r2", r2, on_epoch=True, prog_bar=True, logger=True)
        self.train_predictions.clear()
        self.train_targets.clear()

    def on_validation_epoch_end(self):
        # make use of all outputs from each validation_step()
        all_preds = torch.cat(self.validation_predictions)
        all_targets = torch.cat(self.validation_targets)

        # do something with all preds and targets
        loss = self._calculate_loss(all_preds, all_targets)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        # rmse and r2
        rmse, r2 = self._calculate_metrics(all_preds, all_targets)
        self.log("val_rmse", rmse, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_r2", r2, on_epoch=True, prog_bar=True, logger=True)

        self.validation_predictions.clear()
        self.validation_targets.clear()

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_data(X_train, y_train, X_val, y_val, config, seed_model):
    # fixed data loader - for reference on reproducibility:
    # https://docs.pytorch.org/docs/stable/notes/randomness.html

    # a Generator for shuffling
    g = torch.Generator()
    g.manual_seed(seed_model)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return train_loader, val_loader


class NNTuneReportCheckpointCallback(TuneReportCheckpointCallback):
    def __init__(
        self,
        metrics: Optional[Union[str, List[str], Dict[str, str]]] = None,
        filename: str = "checkpoint",
        save_checkpoints: bool = True,
        on: Union[str, List[str]] = "validation_end",
        nb_features: int = None,
    ):
        super().__init__(
            metrics=metrics, filename=filename, save_checkpoints=save_checkpoints, on=on
        )
        self.nb_features = nb_features

    def _handle(self, trainer: Trainer, pl_module: LightningModule):
        # CUSTOM: includes also nb_features in report
        if trainer.sanity_checking:
            return

        report_dict = self._get_report_dict(trainer, pl_module)
        report_dict["nb_features"] = self.nb_features
        if not report_dict:
            return

        with self._get_checkpoint(trainer) as checkpoint:
            train.report(report_dict, checkpoint=checkpoint)


def train_nn(
    config,
    train_val,
    target,
    host_id,
    tax,
    seed_data,
    seed_model,
    nn_type="regression",
):
    # Force deterministic algorithms and disable benchmark
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set the seed for reproducibility
    seed_everything(seed_model, workers=True)
    torch.manual_seed(seed_model)
    random.seed(seed_model)
    np.random.seed(seed_model)

    # Process dataset
    X_train, y_train, X_val, y_val = process_train(
        config, train_val, target, host_id, tax, seed_data
    )

    train_loader, val_loader = load_data(
        X_train, y_train, X_val, y_val, config, seed_model
    )

    # Model
    n_layers = config["n_hidden_layers"]
    # output layer defined by target
    if nn_type == "regression":
        output_layer = [1]
        classes = None
    else:
        # nn_type == "classification" or nn_type == "ordinal_regression"

        # this rounds the targets in a the torch-way for consistency (np rounds
        # differently)
        y_tr_t = torch.from_numpy(y_train).float()
        y_val_t = torch.from_numpy(y_val).float()

        classes_train = torch.round(y_tr_t).long().unique().cpu().numpy()
        classes_val = torch.round(y_val_t).long().unique().cpu().numpy()
        classes = sorted(set(classes_train) | set(classes_val))

        if nn_type == "classification":
            output_layer = [len(classes)]
        else:  # nn_type == "ordinal_regression"
            # CORN reduces number of classes by 1
            output_layer = [len(classes) - 1]

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
        n_units=n_units,
        learning_rate=config["learning_rate"],
        nn_type=nn_type,
        dropout_rate=config["dropout_rate"],
        weight_decay=config["weight_decay"],
        classes=classes,
    )

    _save_taxonomy(tax)
    # Callbacks
    checkpoint_dir = ray.train.get_context().get_trial_dir()

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
        NNTuneReportCheckpointCallback(
            metrics={
                "rmse_val": "val_rmse",
                "rmse_train": "train_rmse",
                "r2_val": "val_r2",
                "r2_train": "train_r2",
                "loss_val": "val_loss",
                "loss_train": "train_loss",
            },
            filename="checkpoint",
            on="train_end",
            nb_features=X_train.shape[1],
        ),
        EarlyStopping(
            monitor="val_loss",
            min_delta=config["early_stopping_min_delta"],
            patience=config["early_stopping_patience"],
            mode="min",
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
        config,
        train_val,
        target,
        host_id,
        tax,
        seed_data,
        seed_model,
        nn_type="regression",
    )


def train_nn_class(
    config, train_val, target, host_id, seed_data, seed_model, tax, tree_phylo
):
    train_nn(
        config,
        train_val,
        target,
        host_id,
        tax,
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
        tax,
        seed_data,
        seed_model,
        nn_type="ordinal_regression",
    )


def add_nb_features_to_results(results, nb_features):
    results["nb_features"] = nb_features
    return results


def custom_xgb_metric(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> List[Tuple[str, float]]:
    """
    Custom metric function for XGBoost to calculate RMSE and R2 score.

    Parameters:
    predt (np.ndarray): Predictions.
    dtrain (xgb.DMatrix): DMatrix containing the labels.

    Returns:
    List[Tuple[str, float]]: List of tuples containing metric names and values.
    """
    y = dtrain.get_label()
    return [("r2", r2_score(y, predt)), ("rmse", np.sqrt(np.mean((predt - y) ** 2)))]


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
    tax (pd.DataFrame): Taxonomy data.
    tree_phylo (skbio.TreeNode): Phylogenetic tree.

    Returns:
    None
    """
    # ! process dataset
    X_train, y_train, X_val, y_val = process_train(
        config, train_val, target, host_id, tax, seed_data
    )
    # Set seeds
    np.random.seed(seed_model)
    random.seed(seed_model)

    # Build input matrices for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    _save_taxonomy(tax)
    # ! model
    # Initialize the checkpoint callback - which is built-in support for xgb
    # models in Ray Tune
    checkpoint_callback = xgb_cc(
        # tune:xgboost
        metrics={
            "r2_train": "train-r2",
            "r2_val": "val-r2",
            "rmse_train": "train-rmse",
            "rmse_val": "val-rmse",
        },
        filename="checkpoint",
        results_postprocessing_fn=lambda results: add_nb_features_to_results(
            results, X_train.shape[1]
        ),
        frequency=1,  # Save checkpoint every iteration
    )
    patience = max(10, int(0.1 * config["n_estimators"]))
    xgb.train(
        config,
        dtrain,
        num_boost_round=config[
            "n_estimators"
        ],  # num_boost_round is the number of boosting iterations,
        # equal to n_estimators in scikit-learn
        evals=[(dtrain, "train"), (dval, "val")],
        callbacks=[checkpoint_callback],
        custom_metric=custom_xgb_metric,
        early_stopping_rounds=patience,
    )
