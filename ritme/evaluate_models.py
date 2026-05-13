import json
import os
import pickle
import warnings
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
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
from ritme.feature_space.utils import _add_suffix, _extract_time_labels, _slice_snapshot
from ritme.model_space.static_trainables import NeuralNet
from ritme.tune_models import CLASSIFICATION_MODELS, TASK_METRICS

plt.rcParams.update({"font.family": "DejaVu Sans"})
plt.style.use("seaborn-v0_8-pastel")

# custom color map
color_map = {
    "train": "lightskyblue",
    "test": "peachpuff",
    "rmse_train": "lightskyblue",
    "rmse_val": "plum",
}


def _get_inference_device() -> torch.device:
    """Return preferred inference device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    device = _get_inference_device()
    model = NeuralNet.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        map_location=device,
        weights_only=False,
    )
    model.to(device)
    return model


def get_model(model_type: str, result: Result) -> Any:
    """
    Get the model of a specified type from the result object.
    """
    model_loaders = {
        "linreg": load_sklearn_model,
        "logreg": load_sklearn_model,
        "trac": load_trac_model,
        "rf": load_sklearn_model,
        "rf_class": load_sklearn_model,
        "xgb": load_xgb_model,
        "xgb_class": load_xgb_model,
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
        self,
        model: Any,
        data_config: Dict[str, str],
        tax: pd.DataFrame,
        path: str,
        model_type: str = None,
    ):
        self.model = model
        self.data_config = data_config
        self.tax = tax
        self.path = path
        self.model_type = model_type
        self.label_encoder = None
        self.ft_prefix = "F"
        # per-snapshot enrichment tracking
        self.snapshot_enriched_cols: Dict[str, List[str]] = {}
        self.snapshot_enriched_other_ft: Dict[str, List[str]] = {}
        # per-snapshot selected raw microbial feature names
        self.snapshot_selected_map: Dict[str, List[str]] = {}
        # ordered list of time labels encountered (extracted from training data)
        self.time_labels: List[str] = []
        # final suffixed feature columns used for modeling
        # (set on train, reused on test)
        self.final_feature_cols: List[str] = []

    @staticmethod
    def _slice_snapshot(ft_all: pd.DataFrame, time_label: str) -> pd.DataFrame:
        return _slice_snapshot(ft_all, time_label)

    @staticmethod
    def _add_suffix(df: pd.DataFrame, time_label: str) -> pd.DataFrame:
        return _add_suffix(df, time_label)

    def aggregate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate a single snapshot (unsuffixed) microbial feature table."""
        return aggregate_microbial_features(
            data, self.data_config.get("data_aggregation"), self.tax
        )

    def select(self, data: pd.DataFrame, time_label: str, split: str) -> pd.DataFrame:
        """Select microbial features for a single snapshot (unsuffixed).

        Train: compute and store selected (possibly grouped) feature names.
        Test: reuse stored selection; if grouping used, add grouped column.
        """
        method = self.data_config.get("data_selection")
        if split == "train" and time_label not in self.snapshot_selected_map:
            train_selected = select_microbial_features(
                data, self.data_config, self.ft_prefix
            )
            self.snapshot_selected_map[time_label] = train_selected.columns.tolist()
            return train_selected
        if time_label not in self.snapshot_selected_map:
            raise ValueError(
                "Model must be fitted on train split for this snapshot before "
                "test prediction."
            )
        sel_cols = self.snapshot_selected_map[time_label]

        # group remaining (unseen) features into the training group column
        ft_to_sum = [c for c in data.columns if c not in sel_cols]
        if len(ft_to_sum) > 0 and method is not None:
            group_name = "_low_abun" if method.startswith("abundance") else "_low_var"
            data[f"{self.ft_prefix}{group_name}"] = data[ft_to_sum].sum(axis=1)
            if f"{self.ft_prefix}{group_name}" not in sel_cols:
                sel_cols = sel_cols + [f"{self.ft_prefix}{group_name}"]
        # features kept during training but missing from test are filled with 0
        missing = [c for c in sel_cols if c not in data.columns]
        if missing:
            warnings.warn(
                f"Snapshot {time_label!r}: features present during training but "
                f"missing from the test set are filled with 0: {missing}."
            )
        return data.reindex(columns=sel_cols, fill_value=0)

    def transform(self, data: pd.DataFrame, time_label: str) -> pd.DataFrame:
        """Transform a single snapshot (unsuffixed)."""
        method = self.data_config.get("data_transform", None)
        if method == "alr":
            denom_map = self.data_config.get("data_alr_denom_idx_map", None)
            if not denom_map:
                raise ValueError(
                    "ALR transform requires 'data_alr_denom_idx_map' per snapshot."
                )
            denom_idx = denom_map.get(time_label)
            return transform_microbial_features(data, method, denom_idx)
        return transform_microbial_features(data, method, None)

    def enrich(
        self,
        raw: pd.DataFrame,
        microbial_ft: List[str],
        transformed: pd.DataFrame,
        config: dict,
        time_label: str,
        split: str = "train",
    ) -> tuple[List[str], pd.DataFrame]:
        """Enrich a single snapshot (unsuffixed).

        functionality dependent on split:
        - Train: compute enrichment for this snapshot and persist column ordering
          and the list of added features.
        - Test: reuse enrichment schema (dummy columns etc.) learned on the train
          snapshot.
        """
        other_ft_ls, enriched = enrich_features(raw, microbial_ft, transformed, config)
        if split == "train" and time_label not in self.snapshot_enriched_cols:
            self.snapshot_enriched_cols[time_label] = enriched.columns.tolist()
            self.snapshot_enriched_other_ft[time_label] = other_ft_ls
            return other_ft_ls, enriched
        if time_label not in self.snapshot_enriched_cols:
            raise ValueError(
                "Model must be fitted (enrich) on train split for this snapshot "
                "before test enrichment."
            )
        # reindex to train snapshot's enrichment columns
        train_cols = self.snapshot_enriched_cols[time_label]
        enriched = enriched.reindex(columns=train_cols, fill_value=0)
        other_ft_ls = self.snapshot_enriched_other_ft[time_label]
        return other_ft_ls, enriched

    def build_design_matrix(self, data: pd.DataFrame, split: str) -> pd.DataFrame:
        """Run per-snapshot feature engineering pipeline and return the design matrix.

        This is the same pipeline used internally by ``predict`` but stops before
        calling the underlying model, returning the processed feature DataFrame
        instead.
        """
        micro_ft_raw_all = [c for c in data.columns if c.startswith(self.ft_prefix)]
        self.time_labels = (
            _extract_time_labels(micro_ft_raw_all, self.ft_prefix)
            if not self.time_labels
            else self.time_labels
        )
        if not self.time_labels:
            raise ValueError("No time labels detected in microbial feature columns.")

        accum = pd.DataFrame(index=data.index)
        micro_ft_transf_all: list[str] = []
        other_ft_ls_all: list[str] = []

        for tlabel in self.time_labels:
            snap_data = self._slice_snapshot(data, tlabel)

            # 1. Raw microbial features (unsuffixed) + metadata
            snap_micro_ft_raw = [
                c for c in snap_data.columns if c.startswith(self.ft_prefix)
            ]
            snap_ft_unsuff = snap_data[snap_micro_ft_raw].copy()
            snap_md_unsuff = snap_data.drop(columns=snap_micro_ft_raw)

            # Separate NaN rows (missing past observations) from real data
            nan_mask = snap_ft_unsuff.isna().all(axis=1)
            has_nan_rows = nan_mask.any()
            if has_nan_rows:
                real_ft = snap_ft_unsuff[~nan_mask]
                real_data = snap_data[~nan_mask]
                real_md = snap_md_unsuff[~nan_mask]
            else:
                real_ft = snap_ft_unsuff
                real_data = snap_data
                real_md = snap_md_unsuff

            # 2. Aggregate & 3. Select (unsuffixed, real rows only)
            snap_agg_unsuff = self.aggregate(real_ft)
            snap_sel_unsuff = self.select(snap_agg_unsuff, tlabel, split)

            # 4. Transform (unsuffixed output)
            snap_transf_unsuff = self.transform(snap_sel_unsuff, tlabel)

            # 6. Enrich (unsuffixed) using raw + transformed + metadata
            other_ft_ls_unsuff, enriched = self.enrich(
                raw=real_data,
                microbial_ft=real_ft.columns.tolist(),
                transformed=snap_transf_unsuff.join(real_md),
                config=self.data_config,
                time_label=tlabel,
                split=split,
            )

            # Reintroduce NaN rows
            if has_nan_rows:
                nan_placeholder = pd.DataFrame(
                    np.nan,
                    index=snap_data.index[nan_mask],
                    columns=enriched.columns,
                )
                enriched = pd.concat([enriched, nan_placeholder]).loc[snap_data.index]

            # 7. Suffix columns once (t0 stays unsuffixed)
            enriched_suff = self._add_suffix(enriched, tlabel)

            # Track transformed microbial and enrichment features
            if tlabel == "t0":
                micro_ft_transf_all.extend(snap_transf_unsuff.columns.tolist())
                other_ft_ls_all.extend(other_ft_ls_unsuff)
            else:
                micro_ft_transf_all.extend(
                    [f"{c}__{tlabel}" for c in snap_transf_unsuff.columns]
                )
                other_ft_ls_all.extend([f"{c}__{tlabel}" for c in other_ft_ls_unsuff])

            # 8. Accumulate
            accum = accum.join(enriched_suff, how="left")

        final_cols = micro_ft_transf_all + other_ft_ls_all
        if split == "train" and not self.final_feature_cols:
            self.final_feature_cols = final_cols
        use_cols = self.final_feature_cols if split != "train" else final_cols
        X = accum.reindex(columns=use_cols, fill_value=0).fillna(np.nan).astype(float)
        return X

    def predict(self, data: pd.DataFrame, split: str) -> Any:
        """Run per-snapshot pipeline and predict hard labels / regression values."""
        X = self.build_design_matrix(data, split)

        if isinstance(self.model, NeuralNet):
            self.model.eval()
            with torch.no_grad():
                model_device = next(self.model.parameters()).device
                X_t = (
                    torch.tensor(X.values, dtype=torch.float32)
                    .clone()
                    .detach()
                    .to(model_device)
                )
                predicted = self.model(X_t)
                predicted = self.model._prepare_predictions(predicted)
                predicted = predicted.detach().cpu().numpy().flatten()
        elif isinstance(self.model, dict):
            # trac model
            log_geom, _ = _preprocess_taxonomy_aggregation(
                X.values, self.model["matrix_a"]
            )
            alpha = self.model["model"].values
            predicted = log_geom.dot(alpha[1:]) + alpha[0]
        elif isinstance(self.model, xgb.core.Booster):
            raw = self.model.predict(xgb.DMatrix(X))
            if self.model_type == "xgb_class" and raw.ndim == 2:
                predicted = raw.argmax(axis=1)
            else:
                predicted = raw.flatten()
        else:
            predicted = self.model.predict(X.values).flatten()

        if self.label_encoder is not None:
            predicted = self.label_encoder.inverse_transform(predicted.astype(int))
        return predicted

    def predict_proba(self, data: pd.DataFrame, split: str) -> tuple:
        """Return ``(proba, classes)`` for classification models.

        ``proba`` has shape ``(n_samples, n_classes)`` and its columns align
        with ``classes`` in the original target space.
        """
        X = self.build_design_matrix(data, split)

        if isinstance(self.model, NeuralNet):
            if self.model.nn_type not in ("classification", "ordinal_regression"):
                raise ValueError(
                    "predict_proba is only defined for classification "
                    f"models; nn_type={self.model.nn_type!r}."
                )
            self.model.eval()
            with torch.no_grad():
                model_device = next(self.model.parameters()).device
                X_t = (
                    torch.tensor(X.values, dtype=torch.float32)
                    .clone()
                    .detach()
                    .to(model_device)
                )
                logits = self.model(X_t)
                proba = self.model._predict_proba(logits)
            classes = list(self.model.classes)
            if self.label_encoder is not None:
                classes = list(
                    self.label_encoder.inverse_transform(
                        np.asarray(classes).astype(int)
                    )
                )
            return proba, classes

        if isinstance(self.model, xgb.core.Booster):
            if self.model_type != "xgb_class":
                raise ValueError(
                    "predict_proba is only defined for classification XGBoost "
                    f"models; got model_type={self.model_type!r}."
                )
            proba = self.model.predict(xgb.DMatrix(X))
            if proba.ndim == 1:
                proba = np.column_stack([1.0 - proba, proba])
            if self.label_encoder is None:
                raise ValueError(
                    "xgb_class TunedModel is missing its label_encoder; "
                    "cannot recover original class labels."
                )
            classes = list(self.label_encoder.classes_)
            return proba, classes

        if isinstance(self.model, dict):
            raise ValueError("predict_proba is not defined for TRAC models.")

        if not hasattr(self.model, "predict_proba"):
            raise ValueError(
                f"Underlying model of type {type(self.model).__name__} does "
                "not support predict_proba."
            )
        proba = self.model.predict_proba(X.values)
        classes = list(self.model.classes_)
        if self.label_encoder is not None:
            classes = list(
                self.label_encoder.inverse_transform(np.asarray(classes).astype(int))
            )
        return proba, classes


# Hyperparameters that, within a single model type, indicate "more
# regularised / simpler" configurations. The sign tells whether smaller
# (sign=+1) or larger (sign=-1) values are simpler. Used as the secondary
# tiebreaker by the one-standard-error rule below.
_MODEL_SIMPLICITY_KNOBS: Dict[str, List[tuple[str, int]]] = {
    # ElasticNet: higher alpha -> stronger regularisation -> simpler.
    "linreg": [("alpha", -1), ("l1_ratio", -1)],
    # Logistic regression: lower C -> stronger regularisation -> simpler.
    "logreg": [("C", +1)],
    # TRAC: higher lambda -> sparser solution -> simpler.
    "trac": [("lambda", -1)],
    # Random forest: fewer / shallower trees, larger leaves -> simpler.
    "rf": [
        ("n_estimators", +1),
        ("max_depth", +1),
        ("min_samples_leaf", -1),
    ],
    "rf_class": [
        ("n_estimators", +1),
        ("max_depth", +1),
        ("min_samples_leaf", -1),
    ],
    # XGBoost: fewer / shallower rounds, more regularisation -> simpler.
    "xgb": [
        ("n_estimators", +1),
        ("max_depth", +1),
        ("gamma", -1),
        ("reg_alpha", -1),
        ("reg_lambda", -1),
    ],
    "xgb_class": [
        ("n_estimators", +1),
        ("max_depth", +1),
        ("gamma", -1),
        ("reg_alpha", -1),
        ("reg_lambda", -1),
    ],
    # NN: fewer / smaller layers, more dropout / weight decay -> simpler.
    "nn_reg": [("n_hidden_layers", +1), ("dropout_rate", -1), ("weight_decay", -1)],
    "nn_class": [("n_hidden_layers", +1), ("dropout_rate", -1), ("weight_decay", -1)],
    "nn_corn": [("n_hidden_layers", +1), ("dropout_rate", -1), ("weight_decay", -1)],
}


def _trial_simplicity_key(
    model_type: str, trial_metrics: Dict[str, Any], trial_config: Dict[str, Any]
) -> tuple:
    """Build a sort key where smaller is simpler.

    Primary axis: ``nb_features`` (data-engineering side, shared across model
    types). Secondary axes: model-internal regularisation knobs from
    ``_MODEL_SIMPLICITY_KNOBS``. NaNs / missing values sort last.
    """
    nb_features = trial_metrics.get("nb_features")
    primary = float("inf") if nb_features is None else float(nb_features)
    secondary: list[float] = []
    for knob, sign in _MODEL_SIMPLICITY_KNOBS.get(model_type, []):
        v = trial_config.get(knob)
        if v is None:
            secondary.append(float("inf"))
            continue
        try:
            secondary.append(sign * float(v))
        except (TypeError, ValueError):
            secondary.append(float("inf"))
    return (primary, *secondary)


def _select_best_with_one_se(
    result_grid,
    metric: str,
    mode: str,
    model_type: str,
):
    """Apply the one-standard-error rule to a Ray Tune ``ResultGrid``.

    Picks the simplest configuration whose mean cross-validation score is
    within one standard error of the best mean. (Same idea as glmnet's
    ``lambda.1se``: trade a small loss in cross-validation score for a
    simpler model that is likelier to generalise.)

    Trials without a finite ``<metric>_se`` (single-split runs, or K-fold
    runs that had K-1 NaN folds) are excluded from selection. If no trial
    has a finite SE, defers to ``ResultGrid.get_best_result(scope='all')``.
    """
    sign = 1 if mode == "min" else -1
    candidates = []
    for result in result_grid:
        m = result.metrics or {}
        mean = m.get(f"{metric}_mean", m.get(metric))
        se = m.get(f"{metric}_se")
        if mean is None or (isinstance(mean, float) and np.isnan(mean)):
            continue
        # Only trials with a finite SE participate. Single-split trials (no
        # ``_se`` key) and K-fold trials whose K-1 folds returned NaN (``_se``
        # is NaN) have unreliable means and cannot anchor the 1-SE band.
        if se is None or (isinstance(se, float) and np.isnan(se)):
            continue
        candidates.append((result, float(mean), float(se)))

    if not candidates:
        # No reliable K-fold trials -> defer to Ray Tune's single-best lookup,
        # preserving the pre-K-fold behavior. ``metric`` / ``mode`` are passed
        # explicitly because ``TuneConfig`` no longer carries them (the
        # scheduler owns them; Ray Tune disallows both -- see tune_models.py).
        return result_grid.get_best_result(metric=metric, mode=mode, scope="all")

    # Best by sign-adjusted mean (lower-is-better -> sign=+1, etc.).
    best_result, best_mean, best_se = min(candidates, key=lambda x: sign * x[1])
    # Tolerance band: configs whose mean is within one SE of the best mean
    # remain candidates (sign-adjusted to handle both min and max metrics).
    band = [(r, m) for (r, m, _) in candidates if sign * (m - best_mean) <= best_se]
    if len(band) <= 1:
        return best_result

    # Within the band, pick the simplest configuration.
    band_sorted = sorted(
        band,
        key=lambda rm: _trial_simplicity_key(
            model_type, rm[0].metrics or {}, rm[0].config or {}
        ),
    )
    return band_sorted[0][0]


def retrieve_n_init_best_models(
    result_dic: Dict[str, Result], train_val: pd.DataFrame
) -> Dict[str, TunedModel]:
    """
    Retrieve and initialize the best models from the result dictionary.

    When trials report K-fold metrics (``<metric>_mean`` and ``<metric>_se``),
    the one-standard-error rule selects the simplest configuration whose mean
    is within one standard error of the best. Trials without K-fold metrics
    fall back to Ray Tune's standard best-result lookup.

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
        # The same metric is used for ranking as during tuning. We honor
        # task_type via the regression vs classification model registry.
        if model_type in CLASSIFICATION_MODELS:
            metric, mode = TASK_METRICS["classification"]
        else:
            metric, mode = TASK_METRICS["regression"]
        best_result = _select_best_with_one_se(
            result_grid, metric=metric, mode=mode, model_type=model_type
        )

        best_model = get_model(model_type, best_result)
        best_data_proc = get_data_processing(best_result)
        best_tax = get_taxonomy(best_result)
        best_path = best_result.path

        tmodel = TunedModel(
            best_model, best_data_proc, best_tax, best_path, model_type=model_type
        )

        # Load label encoder for classification models (string targets)
        le_path = os.path.join(best_result.path, "label_encoder.pkl")
        if os.path.exists(le_path):
            tmodel.label_encoder = load(le_path)

        best_model_dic[model_type] = tmodel
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
