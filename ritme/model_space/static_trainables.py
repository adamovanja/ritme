"""Module with tune trainables of all static models"""

import math
import os
import pickle
import random
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import joblib
import numpy as np

# classo uses np.infty which was removed in NumPy 2.0
if not hasattr(np, "infty"):
    np.infty = np.inf

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
from lightning.pytorch.callbacks import EarlyStopping
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.integration.xgboost import TuneReportCheckpointCallback as xgb_cc
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ritme.feature_space._process_trac_specific import (
    _preprocess_taxonomy_aggregation,
    create_matrix_from_tree,
)
from ritme.feature_space._process_train import process_train, process_train_kfold
from ritme.model_space._model_trac_calc import min_least_squares_solution


def _aggregate_fold_metrics(per_fold_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-fold metric dicts into mean/std/standard-error fields.

    For each metric key found in any fold dict, emits ``<key>``, ``<key>_mean``,
    ``<key>_std`` (sample std, ddof=1, NaN folds excluded), and
    ``<key>_se`` (``std / sqrt(n_valid)``). The bare ``<key>`` is set to the
    mean so existing single-split callers (and Ray Tune metric configuration)
    keep working without rename. Adds ``n_folds`` for downstream auditing.

    Notes:
        The K fold scores are not independent: their training sets overlap by
        ``(K-2)/K * N`` samples, so ``std / sqrt(n_valid)`` is an optimistic
        (too narrow) estimate of the true SE. The formula is retained because
        it is what the downstream 1-SE rule in :mod:`ritme.evaluate_models`
        consumes.

        When a metric has fewer than two non-NaN folds, ``<key>_std`` and
        ``<key>_se`` are set to NaN. The downstream 1-SE rule treats trials
        with NaN SE as unreliable (the mean is a single-fold point estimate
        masquerading as a K-fold result) and excludes them from selection.
    """
    metrics: Dict[str, float] = {}
    keys = sorted({k for d in per_fold_dicts for k in d.keys()})
    for k in keys:
        vals = [d[k] for d in per_fold_dicts if k in d and d[k] is not None]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        if np.isnan(arr).all():
            continue
        mean = float(np.nanmean(arr))
        n_valid = int(np.sum(~np.isnan(arr)))
        if n_valid <= 1:
            # A single surviving fold cannot support a meaningful SE; emit NaN
            # so the downstream 1-SE rule can mark the trial unreliable and
            # exclude it, rather than treating it as a zero-noise winner.
            std = float("nan")
            se = float("nan")
        else:
            std = float(np.nanstd(arr, ddof=1))
            se = std / math.sqrt(n_valid)
        metrics[k] = mean
        metrics[f"{k}_mean"] = mean
        metrics[f"{k}_std"] = std
        metrics[f"{k}_se"] = se
    metrics["n_folds"] = len(per_fold_dicts)
    return metrics


def _allocate_fold_resources(n_splits: int, cpus_per_trial: int) -> tuple[int, int]:
    """Split a trial's CPU budget between parallel folds and the inner model.

    Picks ``n_workers = min(n_splits, cpus_per_trial)`` parallel folds and
    ``cpus_per_fold = floor(cpus_per_trial / n_workers)`` for each fold's
    inner fit (e.g. RandomForest n_jobs). When folds outnumber CPUs, joblib
    queues them across workers automatically.
    """
    cpus_avail = max(1, int(cpus_per_trial))
    n_workers = max(1, min(int(n_splits), cpus_avail))
    cpus_per_fold = max(1, cpus_avail // n_workers)
    return n_workers, cpus_per_fold


# --------------------------------------------------------------------------
# Module-level estimator builders
# --------------------------------------------------------------------------
# Each ``_build_<model>`` is a top-level factory that returns a fresh,
# unfitted estimator from explicit hyperparameter arguments. They replace the
# previous nested ``_make`` closures (which captured ``config`` from the
# trainable scope) so that Ray dispatches estimator construction by function
# reference plus a small explicit kwargs dict — not by cloudpickling an
# enclosing closure together with the full ``config``. Each builder accepts
# ``seed_model`` and ``n_jobs`` for uniform invocation by the fold workers;
# models that don't consume them absorb them via the keyword signature.


def _build_linreg(
    alpha: float,
    l1_ratio: float,
    seed_model: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> Pipeline:
    """Build a fresh linear regression pipeline (StandardScaler + ElasticNet).

    ``seed_model`` and ``n_jobs`` are accepted for factory-signature parity
    with the other ``_build_*`` helpers but are unused: ElasticNet's default
    cyclic coordinate descent is deterministic without a seed, and the
    Pipeline wrapper does not expose n_jobs.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "linreg",
                ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True),
            ),
        ]
    )


def _build_logreg(
    C: float,
    penalty: Literal["l1", "l2", "elasticnet"],
    l1_ratio: Optional[float],
    seed_model: int,
    n_jobs: Optional[int] = None,
) -> Pipeline:
    """Build a fresh logistic regression pipeline.

    ``n_jobs`` is accepted for factory-signature parity but not exposed at the
    pipeline level for the saga solver used here.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=C,
                    penalty=penalty,
                    l1_ratio=l1_ratio,
                    solver="saga",
                    max_iter=2000,
                    random_state=seed_model,
                ),
            ),
        ]
    )


def _build_rf(
    n_estimators: int,
    max_depth: Optional[int],
    min_samples_split: float,
    min_weight_fraction_leaf: float,
    min_samples_leaf: float,
    max_features,
    min_impurity_decrease: float,
    bootstrap: bool,
    seed_model: int,
    n_jobs: int,
) -> RandomForestRegressor:
    """Build a fresh RandomForestRegressor from explicit hyperparameters."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        min_impurity_decrease=min_impurity_decrease,
        bootstrap=bootstrap,
        n_jobs=n_jobs,
        random_state=seed_model,
    )


def _build_rf_class(
    n_estimators: int,
    max_depth: Optional[int],
    min_samples_split: float,
    min_weight_fraction_leaf: float,
    min_samples_leaf: float,
    max_features,
    min_impurity_decrease: float,
    bootstrap: bool,
    seed_model: int,
    n_jobs: int,
) -> RandomForestClassifier:
    """Build a fresh RandomForestClassifier from explicit hyperparameters."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        min_impurity_decrease=min_impurity_decrease,
        bootstrap=bootstrap,
        n_jobs=n_jobs,
        random_state=seed_model,
    )


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


def _save_label_encoder(config: dict) -> None:
    """Save label encoder from process_train config to the trial directory."""
    le = config.pop("_label_encoder", None)
    if le is not None:
        le_path = os.path.join(
            ray.tune.get_context().get_trial_dir(),
            "label_encoder.pkl",
        )
        joblib.dump(le, le_path)


def _save_sklearn_model(model: BaseEstimator) -> str:
    """
    Save a Scikit-learn model to a file.

    Parameters:
    model (BaseEstimator): The model to save.

    Returns:
    str: The path to the saved model file.
    """
    model_path = os.path.join(ray.tune.get_context().get_trial_dir(), "model.pkl")
    joblib.dump(model, model_path)
    return model_path


def _save_taxonomy(tax: pd.DataFrame) -> None:
    taxonomy_path = os.path.join(ray.tune.get_context().get_trial_dir(), "taxonomy.pkl")
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

    tune.report(
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


def _classification_metrics_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: List,
) -> Dict[str, float]:
    """Compute the standard ritme classification metric set.

    f1_macro / balanced_accuracy / MCC are recorded at the model's argmax
    decision (= 0.5 on the positive-class probability for binary);
    roc_auc_macro_ovr and log_loss are threshold-free.
    """
    classes = list(classes)
    if len(classes) == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        auc = roc_auc_score(
            y_true,
            y_proba,
            multi_class="ovr",
            average="macro",
            labels=classes,
        )
    return {
        "roc_auc_macro_ovr": float(auc),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_proba, labels=classes)),
    }


def _predict_classification_metrics(
    model: BaseEstimator, X: np.ndarray, y: np.ndarray
) -> Dict[str, float]:
    """Compute classification metrics for an sklearn-compatible classifier."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    classes = list(model.classes_)
    return _classification_metrics_dict(y, y_pred, y_proba, classes)


def _report_classification_results_manually(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    tax: pd.DataFrame,
) -> None:
    model_path = _save_sklearn_model(model)
    _save_taxonomy(tax)

    train_metrics = _predict_classification_metrics(model, X_train, y_train)
    val_metrics = _predict_classification_metrics(model, X_val, y_val)

    metrics = {
        "model_path": model_path,
        "nb_features": X_train.shape[1],
    }
    for name, value in train_metrics.items():
        metrics[f"{name}_train"] = value
    for name, value in val_metrics.items():
        metrics[f"{name}_val"] = value

    tune.report(metrics=metrics)
    return None


def _fit_one_fold_sklearn_regression(
    X_full: np.ndarray,
    y_full: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    estimator_builder: Callable[..., Any],
    builder_kwargs: Dict[str, Any],
    seed_model: int,
    cpus_per_fold: int,
) -> Dict[str, float]:
    """Fit one sklearn regression fold and return per-fold metrics.

    Runs inside a ``ray.remote`` task. ``X_full`` / ``y_full`` arrive as
    materialized numpy arrays (zero-copy view of the Ray plasma object), and
    the train/val slices are taken inside the task so the parent trainable
    never K-way pickles the design matrix. The estimator is built inside the
    worker from a top-level builder function plus an explicit hyperparameter
    dict — Ray pickles the builder by reference and the kwargs dict is plain
    data, so the worker does not carry an implicit closure over the parent's
    ``config``. Seeds are reset at function entry to preserve deterministic
    per-fold initialization.
    """
    np.random.seed(seed_model)
    random.seed(seed_model)
    X_tr = X_full[train_idx]
    y_tr = y_full[train_idx]
    X_va = X_full[val_idx]
    y_va = y_full[val_idx]
    model = estimator_builder(
        **builder_kwargs, seed_model=seed_model, n_jobs=cpus_per_fold
    )
    model.fit(X_tr, y_tr)
    rmse_train, r2_train = _predict_rmse_r2(model, X_tr, y_tr)
    rmse_val, r2_val = _predict_rmse_r2(model, X_va, y_va)
    return {
        "rmse_val": rmse_val,
        "rmse_train": rmse_train,
        "r2_val": r2_val,
        "r2_train": r2_train,
    }


def _fit_one_fold_sklearn_classification(
    X_full: np.ndarray,
    y_full: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    estimator_builder: Callable[..., Any],
    builder_kwargs: Dict[str, Any],
    seed_model: int,
    cpus_per_fold: int,
) -> Dict[str, float]:
    """Fit one sklearn classification fold and return per-fold metrics.

    Counterpart of :func:`_fit_one_fold_sklearn_regression` for classifiers.
    The targets are rounded to integers (matching the single-split path) and
    the standard ritme classification metric set is computed on both train
    and val slices.
    """
    np.random.seed(seed_model)
    random.seed(seed_model)
    X_tr = X_full[train_idx]
    y_tr = np.round(y_full[train_idx]).astype(int)
    X_va = X_full[val_idx]
    y_va = np.round(y_full[val_idx]).astype(int)
    model = estimator_builder(
        **builder_kwargs, seed_model=seed_model, n_jobs=cpus_per_fold
    )
    model.fit(X_tr, y_tr)
    train_metrics = _predict_classification_metrics(model, X_tr, y_tr)
    val_metrics = _predict_classification_metrics(model, X_va, y_va)
    out = {f"{k}_train": v for k, v in train_metrics.items()}
    out.update({f"{k}_val": v for k, v in val_metrics.items()})
    return out


def _fit_full_data_sklearn(
    X_full: np.ndarray,
    y_full: np.ndarray,
    estimator_builder: Callable[..., Any],
    builder_kwargs: Dict[str, Any],
    seed_model: int,
    cpus_per_trial: int,
    classification: bool,
) -> BaseEstimator:
    """Refit an sklearn estimator on the entire design matrix.

    Runs inside a ``ray.remote`` task so that the refit happens in parallel
    with the K fold fits rather than sequentially after them. ``X_full`` /
    ``y_full`` arrive as materialized numpy arrays (zero-copy view of the
    Ray plasma object). The estimator is built from the same module-level
    builder + kwargs pair used by the fold tasks. Seeds are reset at function
    entry so the refit is deterministic and mirrors the previous in-process
    refit. For ``classification=True`` the targets are rounded to integers
    (matching the per-fold classification path). Returns the fitted
    estimator, which the caller pickles to disk as the deployable checkpoint.
    """
    np.random.seed(seed_model)
    random.seed(seed_model)
    model = estimator_builder(
        **builder_kwargs, seed_model=seed_model, n_jobs=cpus_per_trial
    )
    if classification:
        model.fit(X_full, np.round(y_full).astype(int))
    else:
        model.fit(X_full, y_full)
    return model


def _dispatch_folds_then_refit(
    submit_fold: Callable[[int], Any],
    n_folds: int,
    submit_refit: Callable[[], Any],
    n_workers: int,
) -> Tuple[List[Any], Any]:
    """Dispatch K fold tasks (throttled) and then the refit task.

    ``submit_fold(i)`` and ``submit_refit()`` submit a Ray task and return
    its ObjectRef; this helper drives the scheduling: it keeps at most
    ``n_workers`` fold tasks in flight at any time, collects their results
    in submission order, and only after every fold has returned does it
    submit the refit task. This bounds the peak per-node thread count to
    roughly ``n_workers * cpus_per_fold`` during the fold phase and to
    ``cpus_per_trial`` during the refit -- assuming the caller built the
    submit callables with those per-task CPU budgets.

    Earlier the K folds and the refit fanned out simultaneously with
    ``num_cpus=0`` on every task, which let the actual thread count on the
    parent's node climb to ~``K * cpus_per_fold + cpus_per_trial`` (~2x the
    trial's reservation) and silently oversubscribed the CPU budget.
    """
    fold_results: List[Any] = [None] * n_folds
    in_flight: Dict[Any, int] = {}
    next_idx = 0
    while next_idx < n_folds or in_flight:
        while next_idx < n_folds and len(in_flight) < n_workers:
            ref = submit_fold(next_idx)
            in_flight[ref] = next_idx
            next_idx += 1
        if in_flight:
            done, _ = ray.wait(list(in_flight.keys()), num_returns=1)
            ref = done[0]
            idx = in_flight.pop(ref)
            fold_results[idx] = ray.get(ref)
    refit_result = ray.get(submit_refit())
    return fold_results, refit_result


def _dispatch_kfold_and_refit_sklearn(
    folds_idx: List[Tuple[np.ndarray, np.ndarray]],
    X_full: np.ndarray,
    y_full: np.ndarray,
    estimator_builder: Callable[..., Any],
    builder_kwargs: Dict[str, Any],
    seed_model: int,
    cpus_per_fold: int,
    cpus_per_trial: int,
    classification: bool,
    n_workers: int,
) -> Tuple[List[Dict[str, float]], BaseEstimator]:
    """Dispatch K-fold fits and the full-data refit as Ray tasks.

    Places the full design matrix in Ray's object store once via ``ray.put``
    and reuses the resulting ObjectRefs across all K+1 tasks. Selects the
    regression/classification fold runner, then submits up to ``n_workers``
    per-fold tasks at a time via :func:`_dispatch_folds_then_refit`. The
    refit task is submitted only after every fold has returned, so peak
    per-node thread count stays within the parent trial's CPU reservation.

    Tasks are pinned to the parent trial's node via
    ``NodeAffinitySchedulingStrategy`` with ``soft=False``, so they share
    the trial actor's CPU reservation on that node rather than floating to
    other nodes (which would bypass the per-trial CPU cap). ``num_cpus=0``
    keeps the scheduler from asking for additional CPUs on top of the
    parent reservation -- throttling instead happens explicitly via
    ``n_workers`` in :func:`_dispatch_folds_then_refit`.

    Returns
    -------
    (fold_metrics, full_model)
        ``fold_metrics`` is the list of K per-fold metric dicts in fold
        order; ``full_model`` is the estimator refit on the full design
        matrix.
    """
    X_ref = ray.put(X_full)
    y_ref = ray.put(y_full)
    node_id = ray.get_runtime_context().get_node_id()
    strategy = NodeAffinitySchedulingStrategy(node_id, soft=False)

    fold_fn = (
        _fit_one_fold_sklearn_classification
        if classification
        else _fit_one_fold_sklearn_regression
    )
    remote_fold_fn = ray.remote(num_cpus=0)(fold_fn)
    remote_refit_fn = ray.remote(num_cpus=0)(_fit_full_data_sklearn)

    def submit_fold(i: int) -> Any:
        tr_idx, va_idx = folds_idx[i]
        return remote_fold_fn.options(scheduling_strategy=strategy).remote(
            X_ref,
            y_ref,
            tr_idx,
            va_idx,
            estimator_builder,
            builder_kwargs,
            seed_model,
            cpus_per_fold,
        )

    def submit_refit() -> Any:
        return remote_refit_fn.options(scheduling_strategy=strategy).remote(
            X_ref,
            y_ref,
            estimator_builder,
            builder_kwargs,
            seed_model,
            cpus_per_trial,
            classification,
        )

    return _dispatch_folds_then_refit(
        submit_fold, len(folds_idx), submit_refit, n_workers
    )


def _finalize_and_report_sklearn(
    X_full: np.ndarray,
    full_model: BaseEstimator,
    fold_metrics: List[Dict[str, float]],
    classification: bool,
    tax: pd.DataFrame,
    config: Dict[str, Any],
) -> None:
    """Persist artifacts and report a K-fold trainable's metrics to Tune.

    Accepts the already-refit ``full_model`` (produced in parallel with the
    fold fits by :func:`_dispatch_kfold_and_refit_sklearn`); the refit is
    therefore no longer on the critical path of this function. For
    classification trainables saves the label encoder first so the trial
    directory holds it alongside the model — this must happen after all
    parallel tasks have returned because the encoder is stashed in
    ``config`` by the feature-engineering step. Then persists the model
    pickle and taxonomy, aggregates the per-fold metrics into
    mean/std/standard-error fields, augments with ``model_path`` /
    ``nb_features``, and finally calls ``tune.report``.
    """
    if classification:
        _save_label_encoder(config)

    metrics = _aggregate_fold_metrics(fold_metrics)
    metrics["model_path"] = _save_sklearn_model(full_model)
    metrics["nb_features"] = X_full.shape[1]
    _save_taxonomy(tax)
    tune.report(metrics=metrics)


def _run_kfold_sklearn(
    config: Dict[str, Any],
    train_val: pd.DataFrame,
    target: str,
    host_id: str,
    stratify_by: List[str] | None,
    seed_data: int,
    seed_model: int,
    tax: pd.DataFrame,
    n_splits: int,
    cpus_per_trial: int,
    estimator_builder: Callable[..., Any],
    builder_kwargs: Dict[str, Any],
    classification: bool,
) -> None:
    """Run K-fold cross-validation for an sklearn-style trainable.

    ``estimator_builder`` is a module-level factory function (e.g.
    :func:`_build_linreg`) and ``builder_kwargs`` is the explicit
    hyperparameter dict it consumes. The worker calls
    ``estimator_builder(**builder_kwargs, seed_model=..., n_jobs=...)`` inside
    each fold task. This replaces an earlier nested-closure factory pattern
    that implicitly captured ``config`` and therefore cloudpickled the full
    config dict to every Ray worker.

    Orchestrates four steps:
      1. Engineer features once on full ``train_val`` (same pipeline as
         single-split path) and yield K group-/stratify-aware fold pairs.
      2. Allocate the trial's CPU budget between parallel fold workers and
         each fold's inner fit.
      3. Dispatch the K per-fold fits plus the full-data refit in parallel
         via ``_dispatch_kfold_and_refit_sklearn``.
      4. Persist artifacts and report aggregated metrics to Ray Tune via
         ``_finalize_and_report_sklearn``.
    """
    engineered = process_train_kfold(
        config,
        train_val,
        target,
        host_id,
        tax,
        seed_data,
        n_splits,
        stratify_by=stratify_by,
    )

    n_workers, cpus_per_fold = _allocate_fold_resources(n_splits, cpus_per_trial)

    fold_metrics, full_model = _dispatch_kfold_and_refit_sklearn(
        engineered.fold_indices,
        engineered.X_full,
        engineered.y_full,
        estimator_builder,
        builder_kwargs,
        seed_model,
        cpus_per_fold,
        cpus_per_trial,
        classification,
        n_workers,
    )

    _finalize_and_report_sklearn(
        engineered.X_full,
        full_model,
        fold_metrics,
        classification,
        tax,
        config,
    )


def train_linreg(
    config: Dict[str, Any],
    train_val: pd.DataFrame,
    target: str,
    host_id: str,
    stratify_by: List[str] | None,
    seed_data: int,
    seed_model: int,
    tax: pd.DataFrame = pd.DataFrame(),
    tree_phylo: skbio.TreeNode = skbio.TreeNode(),
    cpus_per_trial: int = 1,
    gpus_per_trial: int = 0,
    task_type: str = "regression",
    k_folds: int = 1,
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
    n_splits = int(k_folds or 1)
    builder_kwargs: Dict[str, Any] = {
        "alpha": config["alpha"],
        "l1_ratio": config["l1_ratio"],
    }

    if n_splits > 1:
        _run_kfold_sklearn(
            config,
            train_val,
            target,
            host_id,
            stratify_by,
            seed_data,
            seed_model,
            tax,
            n_splits,
            cpus_per_trial,
            estimator_builder=_build_linreg,
            builder_kwargs=builder_kwargs,
            classification=False,
        )
        return

    # ! process dataset: X with features & y with host_id
    X_train, y_train, X_val, y_val = process_train(
        config, train_val, target, host_id, tax, seed_data, stratify_by=stratify_by
    )

    # ! model
    np.random.seed(seed_model)
    linreg = _build_linreg(
        **builder_kwargs, seed_model=seed_model, n_jobs=cpus_per_trial
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
    path_to_save = ray.tune.get_context().get_trial_dir()
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
    tune.report(
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


def _fit_trac_single(
    log_geom_train: np.ndarray,
    nleaves: np.ndarray,
    y_train: np.ndarray,
    a_df: pd.DataFrame,
    lam: float,
    seed: int,
):
    """Fit one TRAC model from a log-geom training matrix and return its bundle."""
    np.random.seed(seed)
    matrices_train = (log_geom_train, np.ones((1, len(log_geom_train[0]))), y_train)
    intercept = True
    alpha_norefit = Classo(
        matrix=matrices_train,
        lam=lam,
        typ="R1",
        meth="Path-Alg",
        w=1 / nleaves,
        intercept=intercept,
    )
    selected_param = abs(alpha_norefit) > 1e-5
    alpha = min_least_squares_solution(
        matrices_train, selected_param, intercept=intercept
    )
    return _bundle_trac_model(alpha, a_df)


def _fit_one_fold_trac(
    log_geom_full: np.ndarray,
    y_full: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    nleaves: np.ndarray,
    a_df: pd.DataFrame,
    lam: float,
    seed_model: int,
) -> Dict[str, float]:
    """Fit one TRAC fold and return per-fold RMSE / R2 on train and val.

    Runs inside a ``ray.remote`` task. The log-geom design matrix is sliced
    inside the task from the put-once ``log_geom_full`` array, so the parent
    trainable does not K-way pickle the (potentially large) design matrix.
    Seeds are reset at function entry to preserve deterministic per-fold
    behavior (mirrors the previous joblib closure).
    """
    lg_tr = log_geom_full[train_idx]
    y_tr = y_full[train_idx]
    lg_va = log_geom_full[val_idx]
    y_va = y_full[val_idx]
    model = _fit_trac_single(lg_tr, nleaves, y_tr, a_df, lam, seed_model)
    alpha = model["model"]["alpha"].values
    rmse_tr, r2_tr = _predict_rmse_r2_trac(alpha, lg_tr, y_tr)
    rmse_va, r2_va = _predict_rmse_r2_trac(alpha, lg_va, y_va)
    return {
        "rmse_val": rmse_va,
        "rmse_train": rmse_tr,
        "r2_val": r2_va,
        "r2_train": r2_tr,
    }


def _fit_full_data_trac(
    log_geom_full: np.ndarray,
    nleaves: np.ndarray,
    y_full: np.ndarray,
    a_df: pd.DataFrame,
    lam: float,
    seed: int,
) -> Dict[str, Any]:
    """Refit a TRAC model on the entire log-geom design matrix.

    Runs inside a ``ray.remote`` task so the refit happens in parallel with
    the K fold fits rather than sequentially after them. The arguments
    mirror ``_fit_trac_single`` (``log_geom_full`` arrives as a zero-copy
    view of the Ray plasma object); returns the same model-bundle dict
    shape ({"model": <alpha DataFrame>, "matrix_a": <A>}) that
    ``_fit_trac_single`` already returns, which the caller pickles to disk
    as the deployable checkpoint.
    """
    return _fit_trac_single(log_geom_full, nleaves, y_full, a_df, lam, seed)


def _dispatch_kfold_and_refit_trac(
    folds_idx: List[Tuple[np.ndarray, np.ndarray]],
    log_geom_full: np.ndarray,
    y_full: np.ndarray,
    nleaves: np.ndarray,
    a_df: pd.DataFrame,
    lam: float,
    seed_model: int,
    n_workers: int,
) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    """TRAC counterpart of :func:`_dispatch_kfold_and_refit_sklearn`.

    Shares the same throttle-then-refit dispatch shape via
    :func:`_dispatch_folds_then_refit`; differs only in the per-task args
    (log-geom design matrix, nleaves, A) and the refit return type
    (TRAC model bundle dict).
    """
    lg_ref = ray.put(log_geom_full)
    y_ref = ray.put(y_full)
    a_df_ref = ray.put(a_df)
    node_id = ray.get_runtime_context().get_node_id()
    strategy = NodeAffinitySchedulingStrategy(node_id, soft=False)

    remote_fold_fn = ray.remote(num_cpus=0)(_fit_one_fold_trac)
    remote_refit_fn = ray.remote(num_cpus=0)(_fit_full_data_trac)

    def submit_fold(i: int) -> Any:
        tr_idx, va_idx = folds_idx[i]
        return remote_fold_fn.options(scheduling_strategy=strategy).remote(
            lg_ref,
            y_ref,
            tr_idx,
            va_idx,
            nleaves,
            a_df_ref,
            lam,
            seed_model,
        )

    def submit_refit() -> Any:
        return remote_refit_fn.options(scheduling_strategy=strategy).remote(
            lg_ref,
            nleaves,
            y_ref,
            a_df_ref,
            lam,
            seed_model,
        )

    return _dispatch_folds_then_refit(
        submit_fold, len(folds_idx), submit_refit, n_workers
    )


def train_trac(
    config: Dict[str, Any],
    train_val: pd.DataFrame,
    target: str,
    host_id: str,
    stratify_by: List[str] | None,
    seed_data: int,
    seed_model: int,
    tax: pd.DataFrame,
    tree_phylo: skbio.TreeNode,
    cpus_per_trial: int = 1,
    gpus_per_trial: int = 0,
    task_type: str = "regression",
    k_folds: int = 1,
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
    # ! derive matrix A (same for every fold; depends only on the phylogeny)
    a_df = create_matrix_from_tree(tree_phylo, tax)

    n_splits = int(k_folds or 1)

    if n_splits > 1:
        engineered = process_train_kfold(
            config,
            train_val,
            target,
            host_id,
            tax,
            seed_data,
            n_splits,
            stratify_by=stratify_by,
        )
        # Precompute log-geom transform on the full design matrix once; each
        # fold task slices its train/val views from this shared array.
        log_geom_full, nleaves = _preprocess_taxonomy_aggregation(
            engineered.X_full, a_df
        )

        n_workers, _ = _allocate_fold_resources(n_splits, cpus_per_trial)
        fold_metrics, full_model = _dispatch_kfold_and_refit_trac(
            engineered.fold_indices,
            log_geom_full,
            engineered.y_full,
            nleaves,
            a_df,
            config["lambda"],
            seed_model,
            n_workers,
        )

        df_alpha_with_labels = full_model["model"]
        path_to_save = ray.tune.get_context().get_trial_dir()
        model_path = os.path.join(path_to_save, "model.pkl")
        with open(model_path, "wb") as file:
            pickle.dump(full_model, file)
        _save_taxonomy(tax)

        metrics = _aggregate_fold_metrics(fold_metrics)
        metrics["model_path"] = model_path
        metrics["nb_features"] = df_alpha_with_labels[
            df_alpha_with_labels["alpha"] != 0.0
        ].shape[0]
        tune.report(metrics=metrics)
        return

    # Single-split path (preserved for backwards compatibility and tests)
    X_train, y_train, X_val, y_val = process_train(
        config, train_val, target, host_id, tax, seed_data, stratify_by=stratify_by
    )

    # ! get log_geom
    # pass a_df directly so the sparse representation is not densified.
    log_geom_train, nleaves = _preprocess_taxonomy_aggregation(X_train, a_df)
    log_geom_val, _ = _preprocess_taxonomy_aggregation(X_val, a_df)

    # ! model
    model = _fit_trac_single(
        log_geom_train, nleaves, y_train, a_df, config["lambda"], seed_model
    )

    _report_results_manually_trac(
        model, log_geom_train, y_train, log_geom_val, y_val, tax
    )


def train_rf(
    config: Dict[str, Any],
    train_val: pd.DataFrame,
    target: str,
    host_id: str,
    stratify_by: List[str] | None,
    seed_data: int,
    seed_model: int,
    tax: pd.DataFrame = pd.DataFrame(),
    tree_phylo: skbio.TreeNode = skbio.TreeNode(),
    cpus_per_trial: int = 1,
    gpus_per_trial: int = 0,
    task_type: str = "regression",
    k_folds: int = 1,
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
    cpus_per_trial (int): Number of CPUs allocated by Ray Tune for this trial.

    Returns:
    None
    """
    n_splits = int(k_folds or 1)
    builder_kwargs: Dict[str, Any] = {
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "min_samples_split": config["min_samples_split"],
        "min_weight_fraction_leaf": config["min_weight_fraction_leaf"],
        "min_samples_leaf": config["min_samples_leaf"],
        "max_features": config["max_features"],
        "min_impurity_decrease": config["min_impurity_decrease"],
        "bootstrap": config["bootstrap"],
    }

    if n_splits > 1:
        _run_kfold_sklearn(
            config,
            train_val,
            target,
            host_id,
            stratify_by,
            seed_data,
            seed_model,
            tax,
            n_splits,
            cpus_per_trial,
            estimator_builder=_build_rf,
            builder_kwargs=builder_kwargs,
            classification=False,
        )
        return

    # ! process dataset
    X_train, y_train, X_val, y_val = process_train(
        config, train_val, target, host_id, tax, seed_data, stratify_by=stratify_by
    )

    # ! model
    # setting seed for scikit library
    np.random.seed(seed_model)
    rf = _build_rf(**builder_kwargs, seed_model=seed_model, n_jobs=cpus_per_trial)
    rf.fit(X_train, y_train)

    _report_results_manually(rf, X_train, y_train, X_val, y_val, tax)


def train_logreg(
    config: Dict[str, Any],
    train_val: pd.DataFrame,
    target: str,
    host_id: str,
    stratify_by: List[str] | None,
    seed_data: int,
    seed_model: int,
    tax: pd.DataFrame = pd.DataFrame(),
    tree_phylo: skbio.TreeNode = skbio.TreeNode(),
    cpus_per_trial: int = 1,
    gpus_per_trial: int = 0,
    task_type: str = "classification",
    k_folds: int = 1,
) -> None:
    n_splits = int(k_folds or 1)

    builder_kwargs: Dict[str, Any] = {
        "C": config["C"],
        "penalty": config["penalty"],
        "l1_ratio": config.get("l1_ratio"),
    }

    if n_splits > 1:
        _run_kfold_sklearn(
            config,
            train_val,
            target,
            host_id,
            stratify_by,
            seed_data,
            seed_model,
            tax,
            n_splits,
            cpus_per_trial,
            estimator_builder=_build_logreg,
            builder_kwargs=builder_kwargs,
            classification=True,
        )
        return

    X_train, y_train, X_val, y_val = process_train(
        config, train_val, target, host_id, tax, seed_data, stratify_by=stratify_by
    )
    _save_label_encoder(config)
    y_train = np.round(y_train).astype(int)
    y_val = np.round(y_val).astype(int)

    np.random.seed(seed_model)
    logreg = _build_logreg(
        **builder_kwargs, seed_model=seed_model, n_jobs=cpus_per_trial
    )
    logreg.fit(X_train, y_train)

    _report_classification_results_manually(logreg, X_train, y_train, X_val, y_val, tax)


def train_rf_class(
    config: Dict[str, Any],
    train_val: pd.DataFrame,
    target: str,
    host_id: str,
    stratify_by: List[str] | None,
    seed_data: int,
    seed_model: int,
    tax: pd.DataFrame = pd.DataFrame(),
    tree_phylo: skbio.TreeNode = skbio.TreeNode(),
    cpus_per_trial: int = 1,
    gpus_per_trial: int = 0,
    task_type: str = "classification",
    k_folds: int = 1,
) -> None:
    n_splits = int(k_folds or 1)
    builder_kwargs: Dict[str, Any] = {
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "min_samples_split": config["min_samples_split"],
        "min_weight_fraction_leaf": config["min_weight_fraction_leaf"],
        "min_samples_leaf": config["min_samples_leaf"],
        "max_features": config["max_features"],
        "min_impurity_decrease": config["min_impurity_decrease"],
        "bootstrap": config["bootstrap"],
    }

    if n_splits > 1:
        _run_kfold_sklearn(
            config,
            train_val,
            target,
            host_id,
            stratify_by,
            seed_data,
            seed_model,
            tax,
            n_splits,
            cpus_per_trial,
            estimator_builder=_build_rf_class,
            builder_kwargs=builder_kwargs,
            classification=True,
        )
        return

    X_train, y_train, X_val, y_val = process_train(
        config, train_val, target, host_id, tax, seed_data, stratify_by=stratify_by
    )
    _save_label_encoder(config)
    y_train = np.round(y_train).astype(int)
    y_val = np.round(y_val).astype(int)

    np.random.seed(seed_model)
    rf_cls = _build_rf_class(
        **builder_kwargs, seed_model=seed_model, n_jobs=cpus_per_trial
    )
    rf_cls.fit(X_train, y_train)

    _report_classification_results_manually(rf_cls, X_train, y_train, X_val, y_val, tax)


class NeuralNet(LightningModule):
    def __init__(
        self,
        n_units,
        learning_rate,
        nn_type="regression",
        dropout_rate=0.0,
        weight_decay=0.0,
        classes: Optional[list] = None,
        task_type: str = "regression",
    ):
        super(NeuralNet, self).__init__()
        self.save_hyperparameters()  # This saves all passed arguments to self.hparams
        self.learning_rate = learning_rate
        self.nn_type = nn_type
        self.task_type = task_type
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

    def _predict_proba(self, predictions: torch.Tensor) -> np.ndarray:
        """Per-class probabilities for classification / CORN heads."""
        if self.nn_type == "classification":
            return torch.softmax(predictions, dim=1).detach().cpu().numpy()
        # CORN: conditional sigmoids -> cumulative P(y>k) -> per-class probs.
        if predictions.ndim == 1:
            predictions = predictions.unsqueeze(1)
        cum = torch.cumprod(torch.sigmoid(predictions), dim=1)
        proba = torch.cat(
            [1.0 - cum[:, :1], cum[:, :-1] - cum[:, 1:], cum[:, -1:]], dim=1
        )
        return proba.detach().cpu().numpy()

    def _calculate_metrics(self, predictions, targets):
        preds = self._prepare_predictions(predictions)

        if self.task_type == "regression":
            rmse = torch.sqrt(nn.functional.mse_loss(preds, targets))
            r2score = torchmetrics.regression.R2Score().to(preds.device)
            r2 = r2score(preds, targets)
            return {"rmse": rmse, "r2": r2}

        y_pred_np = preds.detach().cpu().numpy().astype(int)
        y_true_np = targets.detach().cpu().numpy().astype(int)
        y_proba_np = self._predict_proba(predictions)
        return _classification_metrics_dict(
            y_true_np, y_pred_np, y_proba_np, list(self.classes)
        )

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

        metrics = self._calculate_metrics(all_preds, all_targets)
        for name, value in metrics.items():
            self.log(f"train_{name}", value, on_epoch=True, prog_bar=True, logger=True)
        self.train_predictions.clear()
        self.train_targets.clear()

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_predictions)
        all_targets = torch.cat(self.validation_targets)

        loss = self._calculate_loss(all_preds, all_targets)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        metrics = self._calculate_metrics(all_preds, all_targets)
        for name, value in metrics.items():
            self.log(f"val_{name}", value, on_epoch=True, prog_bar=True, logger=True)

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


def load_data(X_train, y_train, X_val, y_val, config, seed_model, num_workers=2):
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
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return train_loader, val_loader


class NNTuneReportCheckpointCallback(TuneReportCheckpointCallback):
    """PyTorch Lightning callback that decouples metric reports from checkpoint writes.

    Reports metrics to Ray Tune after every validation epoch (so the ASHA
    scheduler can prune trials and intermediate progress is logged), and writes
    **at most two** Ray Tune checkpoints per trial:

    1. A safety write on the *first* validation improvement, so trials that
       are paused-then-killed by HyperBand still have at least one checkpoint
       on disk (otherwise ``result.checkpoint`` is ``None`` and downstream
       retrieval crashes).
    2. A final write in ``on_train_end`` containing the best validation state
       seen during the run -- score-based retention by ``CheckpointConfig``
       keeps the better of the two.

    Improvements *between* the first and the last are saved to a per-trial
    scratch directory only (no ``tune.report(checkpoint=...)`` call), so the
    experiment-state snapshotter is not triggered for them. Bounding writes to
    two per trial keeps it from being saturated by concurrent trials.

    Also injects ``nb_features`` into every reported metric dict (used by
    ``evaluate_models.py``).
    """

    def __init__(
        self,
        metrics: Optional[Union[str, List[str], Dict[str, str]]] = None,
        filename: str = "checkpoint",
        save_checkpoints: bool = True,
        on: Union[str, List[str]] = "validation_end",
        nb_features: int = None,
        score_attr: str = "rmse_val",
        score_mode: str = "min",
    ):
        super().__init__(
            metrics=metrics, filename=filename, save_checkpoints=save_checkpoints, on=on
        )
        self.nb_features = nb_features
        if score_mode not in ("min", "max"):
            raise ValueError(f"score_mode must be 'min' or 'max', got {score_mode!r}")
        self._score_attr = score_attr
        self._score_mode = score_mode
        self._best_score = float("inf") if score_mode == "min" else float("-inf")
        # Per-trial scratch dir holding the best Lightning checkpoint seen so
        # far. Lazily created on first improvement; cleaned up at on_train_end
        # after the contents have been reported to Ray Tune.
        self._best_scratch_dir: Optional[str] = None
        self._best_report_dict: Optional[Dict] = None
        # Tracks whether the first-improvement safety checkpoint has been
        # written. Used so we only pay the Ray Tune write cost once during
        # training (before the second write at on_train_end).
        self._wrote_safety_checkpoint = False

    def _is_improvement(self, score) -> bool:
        if score is None:
            return False
        try:
            score = float(score)
        except (TypeError, ValueError):
            return False
        if math.isnan(score):
            return False
        if self._score_mode == "min":
            return score < self._best_score
        return score > self._best_score

    def _build_report_dict(self, trainer, pl_module):
        report_dict = self._get_report_dict(trainer, pl_module)
        if not report_dict:
            return None
        report_dict["nb_features"] = self.nb_features
        return report_dict

    def _ensure_scratch_dir(self) -> str:
        if self._best_scratch_dir is None:
            self._best_scratch_dir = tempfile.mkdtemp(prefix="ritme_nn_best_")
        return self._best_scratch_dir

    def _handle(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.sanity_checking:
            return

        report_dict = self._build_report_dict(trainer, pl_module)
        if report_dict is None:
            return

        score = report_dict.get(self._score_attr)
        improved = self._is_improvement(score)
        if improved:
            self._best_score = score
            scratch_dir = self._ensure_scratch_dir()
            # Overwrite previous best on local disk (no Ray Tune visibility).
            trainer.save_checkpoint(os.path.join(scratch_dir, self._filename))
            self._best_report_dict = dict(report_dict)

        if improved and not self._wrote_safety_checkpoint:
            # Safety write: ensure paused-then-killed trials have a checkpoint.
            self._wrote_safety_checkpoint = True
            checkpoint = ray.train.Checkpoint.from_directory(self._best_scratch_dir)
            tune.report(report_dict, checkpoint=checkpoint)
        else:
            # Cheap metric-only report so ASHA can prune.
            tune.report(report_dict)

    def on_train_end(self, trainer, pl_module):
        # Final write of the best validation state seen during training.
        if self._best_scratch_dir is not None and self._best_report_dict is not None:
            checkpoint = ray.train.Checkpoint.from_directory(self._best_scratch_dir)
            tune.report(self._best_report_dict, checkpoint=checkpoint)
            # Ray Tune copies the checkpoint contents synchronously into its
            # storage during tune.report, so the scratch dir is safe to remove.
            shutil.rmtree(self._best_scratch_dir, ignore_errors=True)
            self._best_scratch_dir = None
            return

        # Fallback: no improvement was ever recorded (e.g. NaN loss every
        # epoch). Persist the current trainer state so downstream retrieval
        # still works.
        report_dict = self._build_report_dict(trainer, pl_module) or {
            "nb_features": self.nb_features
        }
        with self._get_checkpoint(trainer) as checkpoint:
            tune.report(report_dict, checkpoint=checkpoint)


def train_nn(
    config,
    train_val,
    target,
    host_id,
    tax,
    seed_data,
    seed_model,
    stratify_by,
    nn_type="regression",
    cpus_per_trial=1,
    gpus_per_trial=0,
    task_type="regression",
):
    # Limit PyTorch threads to Ray-allocated CPUs to avoid oversubscription
    torch.set_num_threads(cpus_per_trial)

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
        config, train_val, target, host_id, tax, seed_data, stratify_by=stratify_by
    )
    if nn_type != "regression":
        _save_label_encoder(config)

    # Scale DataLoader workers to allocated CPUs (reserve at least 1 for training)
    num_workers = max(0, cpus_per_trial - 1)
    train_loader, val_loader = load_data(
        X_train, y_train, X_val, y_val, config, seed_model, num_workers=num_workers
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
        task_type=task_type,
    )

    _save_taxonomy(tax)
    # Callbacks
    checkpoint_dir = ray.tune.get_context().get_trial_dir()

    os.makedirs(checkpoint_dir, exist_ok=True)

    if task_type == "regression":
        nn_metrics = {
            "rmse_val": "val_rmse",
            "rmse_train": "train_rmse",
            "r2_val": "val_r2",
            "r2_train": "train_r2",
            "loss_val": "val_loss",
            "loss_train": "train_loss",
        }
        nn_score_attr = "rmse_val"
        nn_score_mode = "min"
    else:
        nn_metrics = {
            "roc_auc_macro_ovr_val": "val_roc_auc_macro_ovr",
            "roc_auc_macro_ovr_train": "train_roc_auc_macro_ovr",
            "log_loss_val": "val_log_loss",
            "log_loss_train": "train_log_loss",
            "f1_macro_val": "val_f1_macro",
            "f1_macro_train": "train_f1_macro",
            "balanced_accuracy_val": "val_balanced_accuracy",
            "balanced_accuracy_train": "train_balanced_accuracy",
            "mcc_val": "val_mcc",
            "mcc_train": "train_mcc",
            "loss_val": "val_loss",
            "loss_train": "train_loss",
        }
        nn_score_attr = "roc_auc_macro_ovr_val"
        nn_score_mode = "max"

    callbacks = [
        NNTuneReportCheckpointCallback(
            metrics=nn_metrics,
            filename="checkpoint",
            on="validation_end",
            nb_features=X_train.shape[1],
            score_attr=nn_score_attr,
            score_mode=nn_score_mode,
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
    config,
    train_val,
    target,
    host_id,
    stratify_by,
    seed_data,
    seed_model,
    tax,
    tree_phylo,
    cpus_per_trial=1,
    gpus_per_trial=0,
    task_type="regression",
    k_folds: int = 1,
):
    # k_folds accepted for signature parity; iterative trainable still uses
    # single-split inside ritme. K-fold support for nn is Phase 2 work.
    train_nn(
        config,
        train_val,
        target,
        host_id,
        tax,
        seed_data,
        seed_model,
        stratify_by,
        nn_type="regression",
        cpus_per_trial=cpus_per_trial,
        gpus_per_trial=gpus_per_trial,
        task_type=task_type,
    )


def train_nn_class(
    config,
    train_val,
    target,
    host_id,
    stratify_by,
    seed_data,
    seed_model,
    tax,
    tree_phylo,
    cpus_per_trial=1,
    gpus_per_trial=0,
    task_type="classification",
    k_folds: int = 1,
):
    # k_folds accepted for signature parity; iterative trainable still uses
    # single-split inside ritme. K-fold support for nn is Phase 2 work.
    train_nn(
        config,
        train_val,
        target,
        host_id,
        tax,
        seed_data,
        seed_model,
        stratify_by,
        nn_type="classification",
        cpus_per_trial=cpus_per_trial,
        gpus_per_trial=gpus_per_trial,
        task_type=task_type,
    )


def train_nn_corn(
    config,
    train_val,
    target,
    host_id,
    stratify_by,
    seed_data,
    seed_model,
    tax,
    tree_phylo,
    cpus_per_trial=1,
    gpus_per_trial=0,
    task_type="classification",
    k_folds: int = 1,
):
    # k_folds accepted for signature parity; iterative trainable still uses
    # single-split inside ritme. K-fold support for nn is Phase 2 work.
    # corn model from https://github.com/Raschka-research-group/coral-pytorch
    train_nn(
        config,
        train_val,
        target,
        host_id,
        tax,
        seed_data,
        seed_model,
        stratify_by,
        nn_type="ordinal_regression",
        cpus_per_trial=cpus_per_trial,
        gpus_per_trial=gpus_per_trial,
        task_type=task_type,
    )


def add_nb_features_to_results(results, nb_features):
    results["nb_features"] = nb_features
    return results


class _RitmeXGBCheckpointCallback(xgb_cc):
    """XGBoost callback that decouples metric reporting from checkpoint writes.

    Reports metrics to Ray Tune on every boosting iteration so the ASHA
    scheduler always has fresh data to prune trials, and writes **at most two**
    Ray Tune checkpoints per trial:

    1. A safety write on the *first* validation improvement, so trials that
       are paused-then-killed by HyperBand still have at least one checkpoint
       on disk (otherwise ``result.checkpoint`` is ``None`` and downstream
       retrieval crashes).
    2. A final write in ``after_training`` containing the best booster seen
       during the run -- score-based retention by ``CheckpointConfig`` keeps
       the better of the two.

    Improvements *between* the first and the last are held in memory only
    (``Booster.save_raw`` returns a compact UBJSON bytearray), so per-iteration
    disk I/O stays at zero. Bounding writes to two per trial keeps the
    experiment-state snapshotter from being saturated by concurrent trials.

    Parent's ``checkpoint_at_end`` handling is disabled because it would
    persist the *last* iteration's state, which for trials that overfit early
    is worse than the best historical state we track in-memory.
    """

    def __init__(
        self,
        metrics,
        filename,
        results_postprocessing_fn,
        score_attr,
        score_mode,
    ):
        super().__init__(
            metrics=metrics,
            filename=filename,
            frequency=0,
            checkpoint_at_end=False,
            results_postprocessing_fn=results_postprocessing_fn,
        )
        if score_mode not in ("min", "max"):
            raise ValueError(f"score_mode must be 'min' or 'max', got {score_mode!r}")
        self._score_attr = score_attr
        self._score_mode = score_mode
        self._best_score = float("inf") if score_mode == "min" else float("-inf")
        # In-memory snapshot of the best booster seen so far. ``save_raw`` is
        # cheap (a few KB to a few MB) and avoids per-iteration disk I/O.
        self._best_model_bytes: Optional[bytearray] = None
        self._best_report_dict: Optional[Dict] = None
        # Tracks whether the first-improvement safety checkpoint has been
        # written. Used so we only pay the Ray Tune write cost once during
        # training (before the second write at after_training).
        self._wrote_safety_checkpoint = False

    def _is_improvement(self, score) -> bool:
        if score is None:
            return False
        try:
            score = float(score)
        except (TypeError, ValueError):
            return False
        if math.isnan(score):
            return False
        if self._score_mode == "min":
            return score < self._best_score
        return score > self._best_score

    def after_iteration(self, model, epoch, evals_log):
        self._evals_log = evals_log
        report_dict = self._get_report_dict(evals_log)
        score = report_dict.get(self._score_attr)
        improved = self._is_improvement(score)
        if improved:
            self._best_score = score
            # Snapshot the booster state in-memory (no disk I/O).
            self._best_model_bytes = model.save_raw(raw_format="ubj")
            self._best_report_dict = dict(report_dict)

        if improved and not self._wrote_safety_checkpoint:
            # Safety write: ensure paused-then-killed trials have a checkpoint.
            self._wrote_safety_checkpoint = True
            self._save_and_report_checkpoint(report_dict, model)
        else:
            # Cheap metric-only report so ASHA can prune.
            self._report_metrics(report_dict)

    def after_training(self, model):
        # Final write of the best booster seen during training.
        if self._best_model_bytes is not None:
            best_booster = xgb.Booster()
            best_booster.load_model(bytearray(self._best_model_bytes))
            self._save_and_report_checkpoint(self._best_report_dict, best_booster)
            return model

        # Fallback: no improvement was ever recorded (e.g. NaN val-rmse for
        # every iteration). Persist the current model state so downstream
        # retrieval still works.
        report_dict = self._get_report_dict(self._evals_log) if self._evals_log else {}
        self._save_and_report_checkpoint(report_dict, model)
        return model


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


# --- K-fold path for train_xgb ---------------------------------------------
#
# These helpers implement the sequential K-fold + full-data refit branch of
# ``train_xgb``. In K-fold mode all ``cpus_per_trial`` are routed into each
# fold's ``xgb.train`` call (via ``nthread``), so folds are run serially in
# the trainable's own process -- no Ray-remote fan-out, unlike the sklearn
# K-fold path. The trainable still emits exactly one ``tune.report`` at the
# end, with mean/std/SE aggregated across the folds, and a single refit
# booster persisted as the deployable checkpoint.


def _xgb_params_from_config(
    config: Dict[str, Any],
    cpus_per_trial: int,
    gpus_per_trial: int,
    task_type: str,
) -> Dict[str, Any]:
    """Build the xgb-native params dict from a ritme trainable config.

    Mirrors the single-split branch of ``train_xgb`` so the K-fold path
    consumes the same hyperparameters: it mutates the caller's ``config``
    in place (``nthread`` and optionally ``device``) and returns it.
    Currently only the regression objective is set implicitly by xgb's
    defaults; the classification path will overlay ``objective`` /
    ``num_class`` separately in Task 3+4.
    """
    config["nthread"] = cpus_per_trial
    if gpus_per_trial > 0:
        config["device"] = "cuda"
    return config


def _xgb_fold_metrics(
    booster: xgb.Booster,
    dtrain: xgb.DMatrix,
    dvalid: xgb.DMatrix,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    task_type: str,
) -> Dict[str, float]:
    """Per-fold metric dict for the regression K-fold path.

    Produces the same metric KEYS that the single-split path reports
    (``rmse_train``, ``rmse_val``, ``r2_train``, ``r2_val``) so the
    aggregated dict and downstream 1-SE selection see a consistent
    schema across the two paths.
    """
    y_pred_tr = booster.predict(dtrain)
    y_pred_va = booster.predict(dvalid)
    return {
        "rmse_train": float(root_mean_squared_error(y_tr, y_pred_tr)),
        "rmse_val": float(root_mean_squared_error(y_va, y_pred_va)),
        "r2_train": float(r2_score(y_tr, y_pred_tr)),
        "r2_val": float(r2_score(y_va, y_pred_va)),
    }


def _xgb_refit_rounds(
    per_fold_best_iter: List[Optional[int]], n_estimators_config: int
) -> int:
    """Refit num_boost_round from the K-fold signal.

    Median of per-fold best_iteration; falls back to ``n_estimators_config``
    if any fold's early-stop did not trigger (``best_iteration`` is None or
    unset on the Booster).
    """
    if any(b is None for b in per_fold_best_iter):
        return int(n_estimators_config)
    return int(np.median(per_fold_best_iter))


def _save_xgb_checkpoint(refit_booster: xgb.Booster) -> None:
    """Persist the refit booster as the trial's deployable checkpoint.

    ``load_xgb_model`` (in :mod:`ritme.evaluate_models`) loads the booster
    from ``result.checkpoint.to_directory() / "checkpoint"``. The K-fold
    path needs to surface its refit booster through the same Ray Tune
    checkpoint API. Plan Task 5 wires the actual ``tune.report(metrics,
    checkpoint=...)`` plumbing; for now we save the booster file under the
    trial dir at the expected filename so a follow-up task can pick it up.
    """
    ckpt_path = os.path.join(ray.tune.get_context().get_trial_dir(), "checkpoint")
    refit_booster.save_model(ckpt_path)


def _run_kfold_xgb(
    config: Dict[str, Any],
    train_val: pd.DataFrame,
    target: str,
    host_id: str,
    stratify_by: List[str] | None,
    seed_data: int,
    seed_model: int,
    tax: pd.DataFrame,
    n_splits: int,
    cpus_per_trial: int,
    gpus_per_trial: int,
    task_type: str,
) -> None:
    """Sequential K-fold + full-data refit for ``train_xgb``.

    One ``tune.report`` at the end with aggregated per-fold metrics, plus a
    refit on the full design matrix that produces the deployable
    checkpoint. The K-fold loop is intentionally sequential: in K-fold mode
    all ``cpus_per_trial`` go to each fold's ``xgb.train`` call via the
    ``nthread`` slot in ``xgb_params``, so per-fold parallelism inside xgb
    itself absorbs the trial's CPU budget without an outer Ray-remote
    fan-out.
    """
    xgb_params = _xgb_params_from_config(
        config, cpus_per_trial, gpus_per_trial, task_type
    )
    np.random.seed(seed_model)
    random.seed(seed_model)

    engineered = process_train_kfold(
        config,
        train_val,
        target,
        host_id,
        tax,
        seed_data,
        n_splits,
        stratify_by=stratify_by,
    )

    n_estimators = int(config["n_estimators"])
    early_stop = max(10, int(0.1 * n_estimators))

    per_fold_metrics: List[Dict[str, float]] = []
    per_fold_best_iter: List[Optional[int]] = []
    for tr_idx, va_idx in engineered.fold_indices:
        X_tr, y_tr = engineered.X_full[tr_idx], engineered.y_full[tr_idx]
        X_va, y_va = engineered.X_full[va_idx], engineered.y_full[va_idx]
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dvalid = xgb.DMatrix(X_va, label=y_va)
        booster = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dvalid, "val")],
            early_stopping_rounds=early_stop,
            verbose_eval=False,
        )
        per_fold_metrics.append(
            _xgb_fold_metrics(booster, dtrain, dvalid, y_tr, y_va, task_type)
        )
        per_fold_best_iter.append(getattr(booster, "best_iteration", None))

    aggregated = _aggregate_fold_metrics(per_fold_metrics)
    aggregated["nb_features"] = int(engineered.X_full.shape[1])

    refit_rounds = _xgb_refit_rounds(per_fold_best_iter, n_estimators)
    dfull = xgb.DMatrix(engineered.X_full, label=engineered.y_full)
    refit = xgb.train(
        xgb_params, dfull, num_boost_round=refit_rounds, verbose_eval=False
    )

    _save_taxonomy(tax)
    if task_type == "classification":
        _save_label_encoder(config)
    _save_xgb_checkpoint(refit)
    tune.report(metrics=aggregated)


def train_xgb(
    config: Dict[str, Any],
    train_val: pd.DataFrame,
    target: str,
    host_id: str,
    stratify_by: List[str] | None,
    seed_data: int,
    seed_model: int,
    tax: pd.DataFrame = pd.DataFrame(),
    tree_phylo: skbio.TreeNode = skbio.TreeNode(),
    cpus_per_trial: int = 1,
    gpus_per_trial: int = 0,
    task_type: str = "regression",
    k_folds: int = 1,
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
    cpus_per_trial (int): Number of CPUs allocated by Ray Tune for this trial.
    gpus_per_trial (int): Number of GPUs allocated by Ray Tune for this trial.
    k_folds (int): Number of K-fold splits; values >1 take the K-fold path
        (see :func:`_run_kfold_xgb`), 1 keeps the single-split callback path.

    Returns:
    None
    """
    n_splits = int(k_folds or 1)
    if n_splits > 1:
        return _run_kfold_xgb(
            config,
            train_val,
            target,
            host_id,
            stratify_by,
            seed_data,
            seed_model,
            tax,
            n_splits,
            cpus_per_trial,
            gpus_per_trial,
            task_type,
        )
    # Limit XGBoost threads to Ray-allocated CPUs to avoid oversubscription
    config["nthread"] = cpus_per_trial
    # Use GPU when allocated by Ray Tune
    if gpus_per_trial > 0:
        config["device"] = "cuda"

    # ! process dataset
    X_train, y_train, X_val, y_val = process_train(
        config, train_val, target, host_id, tax, seed_data, stratify_by=stratify_by
    )
    # Set seeds
    np.random.seed(seed_model)
    random.seed(seed_model)

    # Build input matrices for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    _save_taxonomy(tax)
    # ! model
    # Decoupled metric/checkpoint reporting: per-iteration metrics for ASHA,
    # checkpoint writes only on validation improvement.
    checkpoint_callback = _RitmeXGBCheckpointCallback(
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
        score_attr="rmse_val",
        score_mode="min",
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


def custom_xgb_class_metric(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> List[Tuple[str, float]]:
    """Eval metric for ``multi:softprob``: computes the ritme classification
    metric set from the booster's per-class probabilities."""
    y = dtrain.get_label().astype(int)
    classes = list(range(predt.shape[1]))
    y_pred = predt.argmax(axis=1)
    metrics = _classification_metrics_dict(y, y_pred, predt, classes)
    return [(name, value) for name, value in metrics.items()]


def train_xgb_class(
    config: Dict[str, Any],
    train_val: pd.DataFrame,
    target: str,
    host_id: str,
    stratify_by: List[str] | None,
    seed_data: int,
    seed_model: int,
    tax: pd.DataFrame = pd.DataFrame(),
    tree_phylo: skbio.TreeNode = skbio.TreeNode(),
    cpus_per_trial: int = 1,
    gpus_per_trial: int = 0,
    task_type: str = "classification",
    k_folds: int = 1,
) -> None:
    config["nthread"] = cpus_per_trial
    if gpus_per_trial > 0:
        config["device"] = "cuda"

    X_train, y_train, X_val, y_val = process_train(
        config, train_val, target, host_id, tax, seed_data, stratify_by=stratify_by
    )

    # Get label encoder for string targets (from process_train) or create
    # one for numeric targets to ensure 0-indexed integer labels
    le = config.pop("_label_encoder", None)
    if le is not None:
        # String targets: already 0-indexed by process_train
        y_train_enc = np.round(y_train).astype(int)
        y_val_enc = np.round(y_val).astype(int)
    else:
        le = LabelEncoder()
        y_all = np.concatenate([y_train, y_val])
        le.fit(np.round(y_all).astype(int))
        y_train_enc = le.transform(np.round(y_train).astype(int))
        y_val_enc = le.transform(np.round(y_val).astype(int))

    config["objective"] = "multi:softprob"
    config["num_class"] = len(le.classes_)

    np.random.seed(seed_model)
    random.seed(seed_model)

    dtrain = xgb.DMatrix(X_train, label=y_train_enc)
    dval = xgb.DMatrix(X_val, label=y_val_enc)

    _save_taxonomy(tax)

    # Save label encoder for prediction-time inverse transform
    le_path = os.path.join(ray.tune.get_context().get_trial_dir(), "label_encoder.pkl")
    joblib.dump(le, le_path)

    checkpoint_callback = _RitmeXGBCheckpointCallback(
        metrics={
            "roc_auc_macro_ovr_train": "train-roc_auc_macro_ovr",
            "roc_auc_macro_ovr_val": "val-roc_auc_macro_ovr",
            "log_loss_train": "train-log_loss",
            "log_loss_val": "val-log_loss",
            "f1_macro_train": "train-f1_macro",
            "f1_macro_val": "val-f1_macro",
            "balanced_accuracy_train": "train-balanced_accuracy",
            "balanced_accuracy_val": "val-balanced_accuracy",
            "mcc_train": "train-mcc",
            "mcc_val": "val-mcc",
        },
        filename="checkpoint",
        results_postprocessing_fn=lambda results: add_nb_features_to_results(
            results, X_train.shape[1]
        ),
        score_attr="roc_auc_macro_ovr_val",
        score_mode="max",
    )
    patience = max(10, int(0.1 * config["n_estimators"]))
    xgb.train(
        config,
        dtrain,
        num_boost_round=config["n_estimators"],
        evals=[(dtrain, "train"), (dval, "val")],
        callbacks=[checkpoint_callback],
        custom_metric=custom_xgb_class_metric,
        early_stopping_rounds=patience,
    )
