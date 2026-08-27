"""Module with tune trainables of all static models"""

import math
import os
import pickle
import random
import tempfile
import time
from contextlib import contextmanager
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

import joblib
import numpy as np

# classo uses np.infty which was removed in NumPy 2.0
if not hasattr(np, "infty"):
    np.infty = np.inf

import pandas as pd
import ray
import skbio
import xgboost as xgb
from classo import Classo
from ray import tune
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

from ritme.feature_space._process_trac_specific import (
    _preprocess_taxonomy_aggregation,
    create_matrix_from_tree,
)
from ritme.feature_space._process_train import process_train, process_train_kfold
from ritme.model_space._model_trac_calc import min_least_squares_solution

# Refuse nn_corn runs when the rounded target has more discrete levels than
# this: CORN cannot predict outside the train+val rounded unique set, so a
# continuous target degenerates into an ordinal classifier with no
# extrapolation.
DEFAULT_NN_CORN_MAX_LEVELS = 20


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


def _emit_running_fold_aggregate(
    per_fold_metrics: List[Dict[str, float]],
    nb_features: int,
    fold_idx: int,
    n_splits: int,
) -> None:
    """Emit a mid-trial ``tune.report`` carrying the running fold aggregate.

    Used by the sequential K-fold paths (``_run_kfold_xgb``,
    ``nn_trainables._run_kfold_nn``) and, via
    :func:`_emit_running_aggregate_for_completed`, by the parallel sklearn /
    trac paths, so the scheduler can prune obviously-bad trials before all K
    folds finish. Fires
    only for ``fold_idx`` in ``0..n_splits - 2`` -- the last fold's metrics
    are surfaced by the post-refit final report, which also carries the
    deployable checkpoint. No checkpoint is attached here.

    Bare metric keys (``<metric>``) are stripped from the running report; only
    the suffixed variants (``<metric>_mean`` / ``_std`` / ``_se``) plus
    ``n_folds`` / ``nb_features`` survive. This isolates the bare keys to the
    final post-refit report, so Ray Tune's ``get_best_result(metric=<metric>)``
    can only land on a row that carries the deployable checkpoint -- see
    ``issue_eval_class.md`` for the no-checkpoint crash this fix prevents.
    The K-fold scheduler is configured to monitor ``<metric>_mean`` instead
    of ``<metric>`` so ASHA pruning still fires at fold boundaries.
    """
    if fold_idx >= n_splits - 1:
        return
    running = _aggregate_fold_metrics(per_fold_metrics)
    running["nb_features"] = nb_features
    tune.report(metrics=_strip_bare_metric_keys(running))


# Bookkeeping keys that survive bare-metric stripping in checkpoint-less reports.
_STRIP_KEEP_BARE = frozenset({"n_folds", "nb_features", "time_capped"})
_STRIP_KEEP_SUFFIXES = ("_mean", "_std", "_se")


def _strip_bare_metric_keys(metrics: Dict[str, float]) -> Dict[str, float]:
    """Drop bare metric keys, keeping suffixed variants + bookkeeping keys.

    Reports without a deployable checkpoint / ``model_path`` (mid-trial
    running aggregates, time-capped final reports) must not carry bare
    ``<metric>`` keys, so ``get_best_result(metric=<metric>)`` can only land
    on rows with a deployable artifact (see ``issue_eval_class.md``).
    """
    return {
        k: v
        for k, v in metrics.items()
        if k in _STRIP_KEEP_BARE or k.endswith(_STRIP_KEEP_SUFFIXES)
    }


def _time_cap_reached(
    trial_start: float, max_trial_duration_s: Optional[float]
) -> bool:
    """True when the trial's elapsed wall clock reached the per-trial cap."""
    return (
        max_trial_duration_s is not None
        and time.monotonic() - trial_start >= max_trial_duration_s
    )


def _past_deadline(deadline: Optional[float]) -> bool:
    """True when the (monotonic) per-trial deadline has passed."""
    return deadline is not None and time.monotonic() >= deadline


def _report_time_capped_aggregate(
    per_fold_metrics: List[Dict[str, float]], nb_features: int
) -> None:
    """Final report for a K-fold trial stopped by ``max_trial_duration_s``.

    The trial stops cleanly at a fold boundary (never by injecting an
    exception into a running fit -- asynchronous aborts inside native code
    can corrupt the process). The report mirrors the shape of a mid-trial
    running aggregate -- suffixed metric keys only, no checkpoint or
    ``model_path`` -- plus ``time_capped: True`` for auditing, so a capped
    trial can never be selected as best while its partial fold aggregate
    still informs the search algorithm.
    """
    print(
        f"Trial reached max_trial_duration_s after {len(per_fold_metrics)} "
        f"fold(s); stopping at the fold boundary."
    )
    metrics = _aggregate_fold_metrics(per_fold_metrics)
    metrics["nb_features"] = nb_features
    metrics["time_capped"] = True
    tune.report(metrics=_strip_bare_metric_keys(metrics))


def _emit_running_aggregate_for_completed(
    fold_results: List[Optional[Dict[str, float]]],
    nb_features: int,
    n_splits: int,
) -> None:
    """Emit the running fold aggregate over the folds completed so far.

    Adapter for the parallel K-fold paths (sklearn / trac), where
    ``fold_results`` is indexed by fold and fills as tasks complete
    (``None`` for folds still in flight or never submitted). Aggregates the
    completed entries and reports via
    :func:`_emit_running_fold_aggregate`, which skips the report once all
    ``n_splits`` folds are in -- those metrics are carried by the final
    post-refit report instead.
    """
    completed = [r for r in fold_results if r is not None]
    _emit_running_fold_aggregate(completed, nb_features, len(completed) - 1, n_splits)


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
# unfitted estimator from explicit hyperparameter arguments. Module-level
# definitions let Ray dispatch estimator construction by function reference
# plus an explicit kwargs dict, rather than cloudpickling a closure together
# with whatever ``config`` happened to be in scope. Each builder accepts
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


def _trial_artifact_dir() -> str:
    """Durable per-trial artifact directory (identical to ``Result.path``).

    Not ``get_trial_dir()``: as of Ray 2.55 that returns the driver-staging
    directory for a fresh actor but an *unsynced* scratch working directory
    once the actor is reused (``reuse_actors=True``), so artifacts written
    there never reach the trial's storage path and downstream retrieval
    (``get_taxonomy`` / ``get_model`` read ``Result.path``) fails. The
    storage context's
    ``trial_fs_path`` is exactly the path ``Result.path`` resolves to; ritme
    always runs with a local ``storage_path``, so writing to it directly is a
    plain filesystem write.
    """
    path = ray.tune.get_context().get_storage().trial_fs_path
    os.makedirs(path, exist_ok=True)
    return path


def _save_label_encoder(config: dict) -> None:
    """Save label encoder from process_train config to the trial directory."""
    le = config.pop("_label_encoder", None)
    if le is not None:
        le_path = os.path.join(_trial_artifact_dir(), "label_encoder.pkl")
        joblib.dump(le, le_path)


def _save_sklearn_model(model: BaseEstimator) -> str:
    """
    Save a Scikit-learn model to a file.

    Parameters:
    model (BaseEstimator): The model to save.

    Returns:
    str: The path to the saved model file.
    """
    model_path = os.path.join(_trial_artifact_dir(), "model.pkl")
    joblib.dump(model, model_path)
    return model_path


def _save_taxonomy(tax: pd.DataFrame) -> None:
    taxonomy_path = os.path.join(_trial_artifact_dir(), "taxonomy.pkl")
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
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    estimator_builder: Callable[..., Any],
    builder_kwargs: Dict[str, Any],
    seed_model: int,
    cpus_per_fold: int,
) -> Dict[str, float]:
    """Fit one sklearn regression fold and return per-fold metrics.

    Runs inside a ``ray.remote`` task. The train and val design matrices are
    pre-engineered per fold (no train/val leakage) and arrive as materialized
    numpy arrays via Ray plasma object refs. The estimator is built inside the
    worker from a top-level builder function plus an explicit hyperparameter
    dict — Ray pickles the builder by reference and the kwargs dict is plain
    data, so the worker does not carry an implicit closure over the parent's
    ``config``. Seeds are reset at function entry to preserve deterministic
    per-fold initialization.
    """
    np.random.seed(seed_model)
    random.seed(seed_model)
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
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
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
    y_tr = np.round(y_tr).astype(int)
    y_va = np.round(y_va).astype(int)
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
    """Refit an sklearn estimator on the deployable-refit design matrix.

    Runs inside a ``ray.remote`` task so that the refit happens in parallel
    with the K fold fits rather than sequentially after them. ``X_full`` /
    ``y_full`` here are the full-``train_val`` matrices produced by
    ``process_train_kfold`` for the deployable refit -- distinct from the
    per-fold matrices that the fold tasks consume, and passed via their
    own ``ray.put`` ObjectRefs. The estimator is built from the same
    module-level builder + kwargs pair used by the fold tasks. Seeds are
    reset at function entry so the refit is deterministic. For
    ``classification=True`` the targets are rounded to integers (matching
    the per-fold classification path). Returns the fitted estimator, which
    the caller pickles to disk as the deployable checkpoint.
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
    report_running: Optional[Callable[[List[Any]], None]] = None,
    deadline: Optional[float] = None,
) -> Tuple[List[Any], Any]:
    """Dispatch K fold tasks (throttled) and then the refit task.

    ``submit_fold(i)`` and ``submit_refit()`` submit a Ray task and return
    its ObjectRef; this helper drives the scheduling: it keeps at most
    ``n_workers`` fold tasks in flight at any time, collects their results
    in submission order, and only after every fold has returned does it
    submit the refit task. This bounds the peak per-node thread count to
    roughly ``n_workers * cpus_per_fold`` during the fold phase and to
    ``cpus_per_trial`` during the refit, keeping actual usage within the
    trial's CPU reservation -- which would otherwise be ~2x oversubscribed
    if all K fold tasks plus the refit ran simultaneously with
    ``num_cpus=0``.

    ``report_running`` (when given) is invoked with the current
    ``fold_results`` list after each fold result is collected, so Ray Tune
    sees running aggregates mid-trial. With ``n_workers >= n_folds`` every
    fold may already be in flight when the first aggregate lands, so a
    scheduler stop mainly skips the refit and feeds the rung statistics
    that prune *other* trials; with fewer workers it also skips the folds
    not yet submitted.

    ``deadline`` (a ``time.monotonic`` timestamp, when given) implements the
    per-trial wall-clock cap: once past it, no further fold task is
    submitted (fold 0 is always submitted so the trial has at least one
    result), tasks already in flight are drained normally -- never
    cancelled, so no exception is injected into a running fit -- and the
    refit is skipped. The truncated return is ``(fold_results, None)`` with
    ``None`` entries for folds that were never submitted.
    """
    fold_results: List[Any] = [None] * n_folds
    in_flight: Dict[Any, int] = {}
    next_idx = 0
    while next_idx < n_folds or in_flight:
        while (
            next_idx < n_folds
            and len(in_flight) < n_workers
            and not (_past_deadline(deadline) and next_idx > 0)
        ):
            ref = submit_fold(next_idx)
            in_flight[ref] = next_idx
            next_idx += 1
        if not in_flight:
            break
        done, _ = ray.wait(list(in_flight.keys()), num_returns=1)
        ref = done[0]
        idx = in_flight.pop(ref)
        fold_results[idx] = ray.get(ref)
        if report_running is not None:
            report_running(fold_results)
    if next_idx < n_folds or _past_deadline(deadline):
        return fold_results, None
    refit_result = ray.get(submit_refit())
    return fold_results, refit_result


def _submit_sklearn_fold(
    i: int,
    *,
    fold_refs: List[Tuple[Any, Any, Any, Any]],
    remote_fold_fn: Any,
    strategy: NodeAffinitySchedulingStrategy,
    estimator_builder: Callable[..., Any],
    builder_kwargs: Dict[str, Any],
    seed_model: int,
    cpus_per_fold: int,
) -> Any:
    """Submit one sklearn-style K-fold task to Ray.

    ``fold_refs[i]`` is the ``(X_tr_ref, y_tr_ref, X_va_ref, y_va_ref)``
    tuple of Ray ObjectRefs to that fold's per-fold-engineered design and
    target arrays.
    """
    X_tr_ref, y_tr_ref, X_va_ref, y_va_ref = fold_refs[i]
    return remote_fold_fn.options(scheduling_strategy=strategy).remote(
        X_tr_ref,
        y_tr_ref,
        X_va_ref,
        y_va_ref,
        estimator_builder,
        builder_kwargs,
        seed_model,
        cpus_per_fold,
    )


def _submit_sklearn_refit(
    *,
    remote_refit_fn: Any,
    strategy: NodeAffinitySchedulingStrategy,
    X_refit_ref: Any,
    y_refit_ref: Any,
    estimator_builder: Callable[..., Any],
    builder_kwargs: Dict[str, Any],
    seed_model: int,
    cpus_per_trial: int,
    classification: bool,
) -> Any:
    """Submit the sklearn-style refit-on-full-data task to Ray."""
    return remote_refit_fn.options(scheduling_strategy=strategy).remote(
        X_refit_ref,
        y_refit_ref,
        estimator_builder,
        builder_kwargs,
        seed_model,
        cpus_per_trial,
        classification,
    )


def _dispatch_kfold_and_refit_sklearn(
    folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    X_refit: np.ndarray,
    y_refit: np.ndarray,
    estimator_builder: Callable[..., Any],
    builder_kwargs: Dict[str, Any],
    seed_model: int,
    cpus_per_fold: int,
    cpus_per_trial: int,
    classification: bool,
    n_workers: int,
    nb_features: int,
    deadline: Optional[float] = None,
) -> Tuple[List[Dict[str, float]], BaseEstimator]:
    """Dispatch K-fold fits and the full-data refit as Ray tasks.

    Places each fold's pre-engineered ``(X_tr, y_tr, X_va, y_va)`` tuple and
    the refit matrices in Ray's object store via ``ray.put`` and dispatches
    the K+1 tasks against those refs. Selects the regression/classification
    fold runner, then submits up to ``n_workers`` per-fold tasks at a time
    via :func:`_dispatch_folds_then_refit`. The refit task is submitted only
    after every fold has returned, so peak per-node thread count stays
    within the parent trial's CPU reservation. A running fold aggregate is
    reported after each completed fold (``nb_features`` = deployable design
    width) so the scheduler can prune at fold boundaries.

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
        order; ``full_model`` is the estimator refit on ``X_refit`` /
        ``y_refit``.
    """
    fold_refs: List[Tuple[Any, Any, Any, Any]] = [
        (ray.put(X_tr), ray.put(y_tr), ray.put(X_va), ray.put(y_va))
        for X_tr, y_tr, X_va, y_va in folds
    ]
    X_refit_ref = ray.put(X_refit)
    y_refit_ref = ray.put(y_refit)
    node_id = ray.get_runtime_context().get_node_id()
    strategy = NodeAffinitySchedulingStrategy(node_id, soft=False)

    fold_fn = (
        _fit_one_fold_sklearn_classification
        if classification
        else _fit_one_fold_sklearn_regression
    )
    remote_fold_fn = ray.remote(num_cpus=0)(fold_fn)
    remote_refit_fn = ray.remote(num_cpus=0)(_fit_full_data_sklearn)

    submit_fold = partial(
        _submit_sklearn_fold,
        fold_refs=fold_refs,
        remote_fold_fn=remote_fold_fn,
        strategy=strategy,
        estimator_builder=estimator_builder,
        builder_kwargs=builder_kwargs,
        seed_model=seed_model,
        cpus_per_fold=cpus_per_fold,
    )
    submit_refit = partial(
        _submit_sklearn_refit,
        remote_refit_fn=remote_refit_fn,
        strategy=strategy,
        X_refit_ref=X_refit_ref,
        y_refit_ref=y_refit_ref,
        estimator_builder=estimator_builder,
        builder_kwargs=builder_kwargs,
        seed_model=seed_model,
        cpus_per_trial=cpus_per_trial,
        classification=classification,
    )
    report_running = partial(
        _emit_running_aggregate_for_completed,
        nb_features=nb_features,
        n_splits=len(folds),
    )

    return _dispatch_folds_then_refit(
        submit_fold,
        len(folds),
        submit_refit,
        n_workers,
        report_running=report_running,
        deadline=deadline,
    )


def _finalize_and_report_sklearn(
    nb_features: int,
    full_model: BaseEstimator,
    fold_metrics: List[Dict[str, float]],
    classification: bool,
    tax: pd.DataFrame,
    config: Dict[str, Any],
) -> None:
    """Persist artifacts and report a K-fold trainable's metrics to Tune.

    Accepts the already-refit ``full_model`` produced by
    :func:`_dispatch_kfold_and_refit_sklearn`, so the refit is not on the
    critical path of this function. For
    classification trainables saves the label encoder first so the trial
    directory holds it alongside the model — this must happen after all
    parallel tasks have returned because the encoder is stashed in
    ``config`` by the feature-engineering step. Then persists the model
    pickle and taxonomy, aggregates the per-fold metrics into
    mean/std/standard-error fields, augments with ``model_path`` /
    ``nb_features``, and finally calls ``tune.report``.

    ``full_model is None`` signals a trial truncated by
    ``max_trial_duration_s`` (the dispatcher skipped the refit): the report
    then carries only the completed folds' aggregate with no deployable
    artifact -- see :func:`_report_time_capped_aggregate`.
    """
    if full_model is None:
        completed = [m for m in fold_metrics if m is not None]
        _report_time_capped_aggregate(completed, nb_features)
        return

    if classification:
        _save_label_encoder(config)

    metrics = _aggregate_fold_metrics(fold_metrics)
    metrics["model_path"] = _save_sklearn_model(full_model)
    metrics["nb_features"] = nb_features
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
    max_trial_duration_s: Optional[float] = None,
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
      1. Engineer features per fold via ``process_train_kfold`` (each fold
         fits engineering on its own train slice and applies the captured
         params on its val slice -- no cross-sample stat crosses any fold's
         train/val boundary; see ``issue_val_leakage.md``), plus a separate
         full-data engineering pass for the deployable refit.
      2. Allocate the trial's CPU budget between parallel fold workers and
         each fold's inner fit.
      3. Dispatch the K per-fold fits plus the full-data refit in parallel
         via ``_dispatch_kfold_and_refit_sklearn``.
      4. Persist artifacts and report aggregated metrics to Ray Tune via
         ``_finalize_and_report_sklearn``.

    When ``max_trial_duration_s`` is set, folds still unsubmitted at the
    deadline are skipped along with the refit, and the trial ends on a
    time-capped partial aggregate (no deployable artifact).
    """
    trial_start = time.monotonic()
    deadline = (
        trial_start + max_trial_duration_s if max_trial_duration_s is not None else None
    )
    engineered = process_train_kfold(
        config,
        train_val,
        target,
        host_id,
        tax,
        seed_data,
        n_splits,
        stratify_by=stratify_by,
        n_workers=max(1, min(n_splits + 1, int(cpus_per_trial))),
    )

    n_workers, cpus_per_fold = _allocate_fold_resources(n_splits, cpus_per_trial)

    nb_features = int(engineered.X_refit.shape[1])
    fold_metrics, full_model = _dispatch_kfold_and_refit_sklearn(
        engineered.folds,
        engineered.X_refit,
        engineered.y_refit,
        estimator_builder,
        builder_kwargs,
        seed_model,
        cpus_per_fold,
        cpus_per_trial,
        classification,
        n_workers,
        nb_features,
        deadline=deadline,
    )

    _finalize_and_report_sklearn(
        nb_features,
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
    nn_corn_max_levels: int = DEFAULT_NN_CORN_MAX_LEVELS,
    max_trial_duration_s: Optional[float] = None,
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
            max_trial_duration_s=max_trial_duration_s,
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
    path_to_save = _trial_artifact_dir()
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
    lg_tr: np.ndarray,
    y_tr: np.ndarray,
    lg_va: np.ndarray,
    y_va: np.ndarray,
    nleaves: np.ndarray,
    a_df: pd.DataFrame,
    lam: float,
    seed_model: int,
) -> Dict[str, float]:
    """Fit one TRAC fold and return per-fold RMSE / R2 on train and val.

    Runs inside a ``ray.remote`` task. The train and val log-geom design
    matrices are pre-computed per fold (no train/val leakage) and arrive as
    materialized numpy arrays via Ray plasma object refs. Seeds are reset at
    function entry to preserve deterministic per-fold behavior (mirrors the
    previous joblib closure).
    """
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


def _submit_trac_fold(
    i: int,
    *,
    fold_refs: List[Tuple[Any, Any, Any, Any]],
    remote_fold_fn: Any,
    strategy: NodeAffinitySchedulingStrategy,
    nleaves: np.ndarray,
    a_df_ref: Any,
    lam: float,
    seed_model: int,
) -> Any:
    """Submit one TRAC K-fold task to Ray.

    ``fold_refs[i]`` is the ``(lg_tr_ref, y_tr_ref, lg_va_ref, y_va_ref)``
    tuple of Ray ObjectRefs to that fold's per-fold log-geom design and
    target arrays.
    """
    lg_tr_ref, y_tr_ref, lg_va_ref, y_va_ref = fold_refs[i]
    return remote_fold_fn.options(scheduling_strategy=strategy).remote(
        lg_tr_ref,
        y_tr_ref,
        lg_va_ref,
        y_va_ref,
        nleaves,
        a_df_ref,
        lam,
        seed_model,
    )


def _submit_trac_refit(
    *,
    remote_refit_fn: Any,
    strategy: NodeAffinitySchedulingStrategy,
    lg_refit_ref: Any,
    nleaves: np.ndarray,
    y_refit_ref: Any,
    a_df_ref: Any,
    lam: float,
    seed_model: int,
) -> Any:
    """Submit the TRAC refit-on-full-data task to Ray."""
    return remote_refit_fn.options(scheduling_strategy=strategy).remote(
        lg_refit_ref,
        nleaves,
        y_refit_ref,
        a_df_ref,
        lam,
        seed_model,
    )


def _dispatch_kfold_and_refit_trac(
    fold_log_geoms: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    log_geom_refit: np.ndarray,
    y_refit: np.ndarray,
    nleaves: np.ndarray,
    a_df: pd.DataFrame,
    lam: float,
    seed_model: int,
    n_workers: int,
    deadline: Optional[float] = None,
) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    """TRAC counterpart of :func:`_dispatch_kfold_and_refit_sklearn`.

    Shares the same throttle-then-refit dispatch shape via
    :func:`_dispatch_folds_then_refit`; differs only in the per-task args
    (log-geom design matrix, nleaves, A) and the refit return type (TRAC
    model bundle dict). Each fold's ``(lg_tr, y_tr, lg_va, y_va)`` is
    pre-computed by the caller from per-fold leak-free engineered matrices.
    Running fold aggregates report ``nb_features`` as the refit log-geom
    design width; the final report keeps the nonzero-coefficient count.
    """
    fold_refs: List[Tuple[Any, Any, Any, Any]] = [
        (ray.put(lg_tr), ray.put(y_tr), ray.put(lg_va), ray.put(y_va))
        for lg_tr, y_tr, lg_va, y_va in fold_log_geoms
    ]
    lg_refit_ref = ray.put(log_geom_refit)
    y_refit_ref = ray.put(y_refit)
    a_df_ref = ray.put(a_df)
    node_id = ray.get_runtime_context().get_node_id()
    strategy = NodeAffinitySchedulingStrategy(node_id, soft=False)

    remote_fold_fn = ray.remote(num_cpus=0)(_fit_one_fold_trac)
    remote_refit_fn = ray.remote(num_cpus=0)(_fit_full_data_trac)

    submit_fold = partial(
        _submit_trac_fold,
        fold_refs=fold_refs,
        remote_fold_fn=remote_fold_fn,
        strategy=strategy,
        nleaves=nleaves,
        a_df_ref=a_df_ref,
        lam=lam,
        seed_model=seed_model,
    )
    submit_refit = partial(
        _submit_trac_refit,
        remote_refit_fn=remote_refit_fn,
        strategy=strategy,
        lg_refit_ref=lg_refit_ref,
        nleaves=nleaves,
        y_refit_ref=y_refit_ref,
        a_df_ref=a_df_ref,
        lam=lam,
        seed_model=seed_model,
    )
    report_running = partial(
        _emit_running_aggregate_for_completed,
        nb_features=int(log_geom_refit.shape[1]),
        n_splits=len(fold_log_geoms),
    )

    return _dispatch_folds_then_refit(
        submit_fold,
        len(fold_log_geoms),
        submit_refit,
        n_workers,
        report_running=report_running,
        deadline=deadline,
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
    nn_corn_max_levels: int = DEFAULT_NN_CORN_MAX_LEVELS,
    max_trial_duration_s: Optional[float] = None,
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
    trial_start = time.monotonic()
    deadline = (
        trial_start + max_trial_duration_s if max_trial_duration_s is not None else None
    )
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
            n_workers=max(1, min(n_splits + 1, int(cpus_per_trial))),
        )
        # Log-geom transform each fold's pre-engineered (X_tr, X_va) and the
        # full-data refit matrix. ``nleaves`` depends only on ``a_df``, so it
        # is the same across all calls and is captured once from the refit
        # transform below.
        fold_log_geoms: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for X_tr, y_tr, X_va, y_va in engineered.folds:
            lg_tr, _ = _preprocess_taxonomy_aggregation(X_tr, a_df)
            lg_va, _ = _preprocess_taxonomy_aggregation(X_va, a_df)
            fold_log_geoms.append((lg_tr, y_tr, lg_va, y_va))
        log_geom_refit, nleaves = _preprocess_taxonomy_aggregation(
            engineered.X_refit, a_df
        )

        n_workers, _ = _allocate_fold_resources(n_splits, cpus_per_trial)
        fold_metrics, full_model = _dispatch_kfold_and_refit_trac(
            fold_log_geoms,
            log_geom_refit,
            engineered.y_refit,
            nleaves,
            a_df,
            config["lambda"],
            seed_model,
            n_workers,
            deadline=deadline,
        )

        if full_model is None:
            # Truncated by max_trial_duration_s: report the completed folds'
            # aggregate; no deployable artifact exists.
            completed = [m for m in fold_metrics if m is not None]
            _report_time_capped_aggregate(completed, int(log_geom_refit.shape[1]))
            return

        df_alpha_with_labels = full_model["model"]
        path_to_save = _trial_artifact_dir()
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
    nn_corn_max_levels: int = DEFAULT_NN_CORN_MAX_LEVELS,
    max_trial_duration_s: Optional[float] = None,
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
            max_trial_duration_s=max_trial_duration_s,
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
    nn_corn_max_levels: int = DEFAULT_NN_CORN_MAX_LEVELS,
    max_trial_duration_s: Optional[float] = None,
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
            max_trial_duration_s=max_trial_duration_s,
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
    nn_corn_max_levels: int = DEFAULT_NN_CORN_MAX_LEVELS,
    max_trial_duration_s: Optional[float] = None,
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
            max_trial_duration_s=max_trial_duration_s,
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


def _check_nn_corn_levels(n_levels: int, nn_corn_max_levels: int) -> None:
    """Validate that the rounded ``nn_corn`` target has at most
    ``nn_corn_max_levels`` discrete levels.

    Raises ``ValueError`` naming ``nn_corn_max_levels`` so the user can lift
    the cap via the experiment config. Also rejects nonsensical cap values
    (anything but an ``int`` >= 2): CORN's output head is ``n_levels - 1``
    and a cap below 2 yields a degenerate model.
    """
    if not isinstance(nn_corn_max_levels, int) or isinstance(nn_corn_max_levels, bool):
        raise ValueError(
            f"nn_corn_max_levels must be a positive int (got "
            f"{nn_corn_max_levels!r} of type "
            f"{type(nn_corn_max_levels).__name__})."
        )
    if nn_corn_max_levels < 2:
        raise ValueError(
            f"nn_corn_max_levels must be >= 2 (got {nn_corn_max_levels}); "
            f"CORN's head is n_levels - 1 and a cap below 2 yields a "
            f"degenerate model."
        )
    if n_levels > nn_corn_max_levels:
        raise ValueError(
            f"nn_corn requires a target with few discrete rounded levels "
            f"(got {n_levels}, max nn_corn_max_levels={nn_corn_max_levels}). "
            f"Use nn_reg / xgb / linreg for continuous regression targets, "
            f"or raise the cap explicitly in the experiment config if you "
            f"know what you are doing."
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
# fold's ``xgb.train`` call (via ``nthread``), so the fold *fits* run
# serially in the trainable's own process, unlike the sklearn K-fold path
# (the per-fold feature engineering is fanned out separately via
# ``process_train_kfold``). The trainable emits ``n_splits`` ``tune.report``
# calls total:
# K-1 running-aggregate mid-trial reports (no checkpoint) plus one final
# report after the full-data refit carrying the full aggregate and the
# deployable checkpoint. Mid-trial reports let ASHA prune at fold
# boundaries.


def _prepare_xgb_classification_labels(
    config: Dict[str, Any],
    y_refit: np.ndarray,
    extra_ys: Sequence[np.ndarray] = (),
) -> Tuple[np.ndarray, int, List[np.ndarray]]:
    """Encode K-fold targets to contiguous 0..K-1 ints for ``multi:softprob``.

    Mirrors the label-encoding branch of single-split ``train_xgb_class``:
    if ``process_train_kfold`` already stashed a LabelEncoder on
    ``config["_label_encoder"]`` (non-numeric target), reuse it -- the y
    values are already 0-indexed floats, so we only round + cast. For
    numeric targets the encoder is absent; we fit one here on the rounded
    ints of ``y_refit`` (which sees every label) and stash it on ``config``
    so the caller can persist it via ``_save_label_encoder(config)`` before
    xgb param construction (the encoder MUST leave ``config`` before
    ``xgb.train`` to avoid an "unknown parameter" warning).

    ``extra_ys`` are per-fold ``y_tr`` / ``y_va`` arrays that need to share
    the same label-to-int mapping as ``y_refit`` -- they are encoded with
    the same logic (round + cast for the non-numeric path, ``le.transform``
    on rounded ints for the numeric path) and returned in input order.

    Returns ``(y_refit_int, num_classes, extras_int)``.
    """
    le = config.get("_label_encoder")
    if le is None:
        le = LabelEncoder()
        le.fit(np.round(y_refit).astype(int))
        config["_label_encoder"] = le
        y_refit_int = np.asarray(le.transform(np.round(y_refit).astype(int)))
        extras_int = [
            np.asarray(le.transform(np.round(y).astype(int))) for y in extra_ys
        ]
    else:
        y_refit_int = np.round(y_refit).astype(int)
        extras_int = [np.round(y).astype(int) for y in extra_ys]
    return y_refit_int, len(le.classes_), extras_int


def _xgb_params_from_config(
    config: Dict[str, Any],
    cpus_per_trial: int,
    gpus_per_trial: int,
    task_type: str,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the xgb-native params dict from a ritme trainable config.

    Mirrors the single-split branches of ``train_xgb`` and
    ``train_xgb_class`` so the K-fold path consumes the same
    hyperparameters: mutates the caller's ``config`` in place (``nthread``,
    ``device``, and for classification ``objective`` + ``num_class``) and
    returns it.

    For ``task_type == "classification"`` we set ``objective`` to
    ``multi:softprob`` and ``num_class`` from ``num_classes`` (the K-fold
    orchestrator counts classes from the encoded label set once
    ``process_train_kfold`` has populated / fitted the label encoder).
    Regression leaves the objective at xgb's default (squared error).
    """
    config["nthread"] = cpus_per_trial
    if gpus_per_trial > 0:
        config["device"] = "cuda"
    if task_type == "classification":
        if num_classes is None:
            raise ValueError(
                "num_classes must be provided when task_type='classification'"
            )
        config["objective"] = "multi:softprob"
        config["num_class"] = int(num_classes)
    return config


def _xgb_fold_metrics(
    booster: xgb.Booster,
    dtrain: xgb.DMatrix,
    dvalid: xgb.DMatrix,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    task_type: str,
) -> Dict[str, float]:
    """Per-fold metric dict for the regression and classification K-fold paths.

    Produces the same metric KEYS that the single-split path reports
    (regression: ``rmse_train`` / ``rmse_val`` / ``r2_train`` / ``r2_val``;
    classification: ``roc_auc_macro_ovr`` / ``f1_macro`` /
    ``balanced_accuracy`` / ``mcc`` / ``log_loss`` each with ``_train`` and
    ``_val`` suffix) so the aggregated dict and downstream 1-SE selection
    see a consistent schema across the two paths.

    When early stopping triggered, ``booster.best_iteration`` is set
    (0-indexed). xgb's default ``predict`` uses **all** trees in the
    booster -- including the ``early_stopping_rounds`` overshoot past the
    best iteration -- so the raw metrics would reflect the post-early-stop
    overfit state, not the best state the single-split path's checkpoint
    callback captures. Restricting to ``iteration_range=(0, best+1)``
    keeps the two paths' metric semantics aligned. xgb treats ``(0, 0)``
    as "use all trees" (the default), so the no-early-stop fallback path
    is unchanged. Both regression and classification branches apply the
    same ``iteration_range`` discipline.
    """
    best_iter = getattr(booster, "best_iteration", None)
    end = (best_iter + 1) if best_iter is not None else 0
    iteration_range = (0, end)
    y_pred_tr = booster.predict(dtrain, iteration_range=iteration_range)
    y_pred_va = booster.predict(dvalid, iteration_range=iteration_range)
    if task_type == "classification":
        # ``multi:softprob`` returns an (n_samples, n_classes) probability
        # matrix; argmax gives the class label and the matrix is the
        # threshold-free input for roc_auc / log_loss in
        # _classification_metrics_dict.
        classes = list(range(y_pred_tr.shape[1]))
        y_tr_int = y_tr.astype(int)
        y_va_int = y_va.astype(int)
        train_metrics = _classification_metrics_dict(
            y_tr_int, y_pred_tr.argmax(axis=1), y_pred_tr, classes
        )
        val_metrics = _classification_metrics_dict(
            y_va_int, y_pred_va.argmax(axis=1), y_pred_va, classes
        )
        return {
            **{f"{k}_train": v for k, v in train_metrics.items()},
            **{f"{k}_val": v for k, v in val_metrics.items()},
        }
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

    Median of per-fold ``best_iteration + 1``; falls back to
    ``n_estimators_config`` if any fold's early-stop did not trigger
    (``best_iteration`` is None or unset on the Booster).

    ``best_iteration`` is 0-indexed: to rebuild a booster containing all
    trees up to and including the best iteration, ``num_boost_round`` must
    be ``best_iteration + 1`` (xgb's own internal CV-result truncation uses
    ``[: best_iteration + 1]`` for this reason).
    """
    if any(b is None for b in per_fold_best_iter):
        return int(n_estimators_config)
    return int(np.median(per_fold_best_iter)) + 1


@contextmanager
def _save_xgb_checkpoint(
    refit_booster: xgb.Booster,
) -> Iterator["ray.train.Checkpoint"]:
    """Yield a Ray Tune :class:`Checkpoint` containing the refit booster.

    Mirrors the single-split path's checkpoint plumbing
    (:meth:`_RitmeXGBCheckpointCallback._save_and_report_checkpoint` ->
    parent class's ``_get_checkpoint``): write the booster to a temp dir
    under the filename ``"checkpoint"`` so it lands at the exact path
    :func:`ritme.evaluate_models._get_checkpoint_path` reads
    (``result.checkpoint.to_directory() / "checkpoint"``), wrap that
    directory in a :class:`ray.train.Checkpoint`, and yield it for the
    caller to hand to ``tune.report(metrics=..., checkpoint=...)``.

    The temp dir lives for the duration of the ``with`` block. Ray Tune
    persists the checkpoint contents to durable storage during the
    ``tune.report`` call, so it is safe to let the temp dir vanish at
    exit; the persisted copy is what ``result.checkpoint.to_directory()``
    materialises at load time.
    """
    with tempfile.TemporaryDirectory(prefix="ritme_xgb_refit_") as tmpdir:
        refit_booster.save_model(os.path.join(tmpdir, "checkpoint"))
        yield ray.train.Checkpoint.from_directory(tmpdir)


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
    max_trial_duration_s: Optional[float] = None,
) -> None:
    """Sequential K-fold + full-data refit for ``train_xgb`` /
    ``train_xgb_class``.

    One ``tune.report`` after each completed fold carrying the running
    aggregate over folds-so-far (so ASHA can prune obviously-bad trials
    without waiting for all K folds), plus a final ``tune.report`` after
    the full-data refit that carries the full aggregate and the deployable
    checkpoint. ``_RitmeXGBCheckpointCallback`` stays scoped to the
    single-split path so ASHA can prune at sub-iteration granularity
    there; K-fold trials emit ``n_splits`` reports total. The K-fold loop
    is intentionally sequential: in K-fold mode all ``cpus_per_trial`` go
    to each fold's ``xgb.train`` call via the ``nthread`` slot in
    ``xgb_params``, so per-fold parallelism inside xgb itself absorbs the
    trial's CPU budget without an outer Ray-remote fan-out.

    For ``task_type == "classification"`` the per-fold and refit targets
    returned by ``process_train_kfold`` are encoded by ``_encode_target``
    (already 0-indexed for string targets via the LabelEncoder it stashes
    in ``config["_label_encoder"]``; numeric targets pass through as
    floats). We mirror the single-split branch of ``train_xgb_class``: if
    no encoder was stashed (numeric target) we fit one here on the rounded
    ints of ``y_refit`` (which sees every label) via
    ``_prepare_xgb_classification_labels``, which also encodes the
    flattened per-fold ``(y_tr, y_va)`` arrays consistently so xgb sees
    contiguous 0..K-1 integer labels for ``multi:softprob`` everywhere.
    We then call ``_save_label_encoder(config)`` immediately -- before
    building xgb params -- so the encoder is persisted for
    prediction-time inverse transform AND popped off ``config``
    (otherwise it leaks into ``xgb.train`` and xgb logs a "Parameters: {
    _label_encoder } are not used" warning each fold + at refit).

    When ``max_trial_duration_s`` is set, the elapsed wall clock is checked
    at each fold boundary and before the refit; a capped trial stops cleanly
    with the completed folds' aggregate (no checkpoint) -- never by
    interrupting a running ``xgb.train`` call.
    """
    trial_start = time.monotonic()
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
        n_workers=max(1, min(n_splits + 1, int(cpus_per_trial))),
    )

    y_refit = engineered.y_refit
    folds_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = list(
        engineered.folds
    )
    num_classes: Optional[int] = None
    if task_type == "classification":
        # Encode the refit targets AND every per-fold (y_tr, y_va) using a
        # single shared label-to-int map. Order in the flattened list is
        # (fold0_tr, fold0_va, fold1_tr, fold1_va, ...) so the unpacker
        # below stays in sync.
        flat_extra_ys = [y for fold in engineered.folds for y in (fold[1], fold[3])]
        y_refit, num_classes, encoded_fold_ys = _prepare_xgb_classification_labels(
            config, y_refit, flat_extra_ys
        )
        folds_data = [
            (
                X_tr,
                encoded_fold_ys[2 * i],
                X_va,
                encoded_fold_ys[2 * i + 1],
            )
            for i, (X_tr, _, X_va, _) in enumerate(engineered.folds)
        ]
        # Persist + drop the label encoder before building xgb params so it
        # doesn't leak into ``xgb.train`` (which would log "Parameters: {
        # _label_encoder } are not used" each fold + at refit).
        _save_label_encoder(config)

    xgb_params = _xgb_params_from_config(
        config, cpus_per_trial, gpus_per_trial, task_type, num_classes=num_classes
    )

    n_estimators = int(config["n_estimators"])
    early_stop = max(10, int(0.1 * n_estimators))

    per_fold_metrics: List[Dict[str, float]] = []
    per_fold_best_iter: List[Optional[int]] = []
    nb_features = int(engineered.X_refit.shape[1])
    for fold_idx, (X_tr, y_tr, X_va, y_va) in enumerate(folds_data):
        if per_fold_metrics and _time_cap_reached(trial_start, max_trial_duration_s):
            _report_time_capped_aggregate(per_fold_metrics, nb_features)
            return
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

        _emit_running_fold_aggregate(per_fold_metrics, nb_features, fold_idx, n_splits)

    if _time_cap_reached(trial_start, max_trial_duration_s):
        # All folds finished but the refit no longer fits inside the cap:
        # end without a deployable checkpoint rather than overshooting.
        _report_time_capped_aggregate(per_fold_metrics, nb_features)
        return

    aggregated = _aggregate_fold_metrics(per_fold_metrics)
    aggregated["nb_features"] = nb_features

    refit_rounds = _xgb_refit_rounds(per_fold_best_iter, n_estimators)
    dfull = xgb.DMatrix(engineered.X_refit, label=y_refit)
    refit = xgb.train(
        xgb_params, dfull, num_boost_round=refit_rounds, verbose_eval=False
    )

    _save_taxonomy(tax)
    with _save_xgb_checkpoint(refit) as checkpoint:
        tune.report(metrics=aggregated, checkpoint=checkpoint)


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
    nn_corn_max_levels: int = DEFAULT_NN_CORN_MAX_LEVELS,
    max_trial_duration_s: Optional[float] = None,
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
            max_trial_duration_s=max_trial_duration_s,
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
    nn_corn_max_levels: int = DEFAULT_NN_CORN_MAX_LEVELS,
    max_trial_duration_s: Optional[float] = None,
) -> None:
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
            task_type="classification",
            max_trial_duration_s=max_trial_duration_s,
        )
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
    le_path = os.path.join(_trial_artifact_dir(), "label_encoder.pkl")
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
