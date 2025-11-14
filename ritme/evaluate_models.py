import json
import os
import pickle
from typing import Any, Dict, List

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
from ritme.feature_space.utils import _extract_time_labels
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

    def _slice_snapshot(self, ft_all: pd.DataFrame, time_label: str) -> pd.DataFrame:
        suffix = f"__{time_label}"
        cols = [c for c in ft_all.columns if c.endswith(suffix)]
        snap = ft_all[cols].copy()
        snap.columns = [c[: -len(suffix)] for c in cols]
        return snap

    def _add_suffix(self, df: pd.DataFrame, time_label: str) -> pd.DataFrame:
        df_s = df.copy()
        df_s.columns = [f"{c}__{time_label}" for c in df_s.columns]
        return df_s

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

        # group remaining if necessary
        ft_to_sum = [c for c in data.columns if c not in sel_cols]
        if len(ft_to_sum) > 0:
            group_name = "_low_abun" if method.startswith("abundance") else "_low_var"
            data[f"{self.ft_prefix}{group_name}"] = data[ft_to_sum].sum(axis=1)
            # ensure grouped column present if part of original selection output
            if f"{self.ft_prefix}{group_name}" not in sel_cols:
                sel_cols = sel_cols + [f"{self.ft_prefix}{group_name}"]
        return data[sel_cols]

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

    def predict(self, data: pd.DataFrame, split: str) -> Any:
        """Run per-snapshot pipeline and predict."""
        micro_ft_raw_all = [c for c in data.columns if c.startswith(self.ft_prefix)]
        self.time_labels = (
            _extract_time_labels(micro_ft_raw_all, self.ft_prefix)
            if not self.time_labels
            else self.time_labels
        )
        if not self.time_labels:
            raise ValueError(
                "No time labels detected in microbial feature columns. "
                "Ensure columns are suffixed."
            )

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

            # 2. Aggregate & 3. Select (unsuffixed)
            snap_agg_unsuff = self.aggregate(snap_ft_unsuff)
            snap_sel_unsuff = self.select(snap_agg_unsuff, tlabel, split)

            # 4. Transform (unsuffixed output)
            snap_transf_unsuff = self.transform(snap_sel_unsuff, tlabel)

            # 6. Enrich (unsuffixed) using raw + transformed + metadata
            other_ft_ls_unsuff, enriched = self.enrich(
                raw=snap_data,
                microbial_ft=snap_ft_unsuff.columns.tolist(),
                transformed=snap_transf_unsuff.join(snap_md_unsuff),
                config=self.data_config,
                time_label=tlabel,
                split=split,
            )

            # 7. Suffix columns once: transformed + enrichment (+ metadata) all at once
            enriched_suff = self._add_suffix(enriched, tlabel)

            # Track transformed microbial (suffixed) and enrichment features
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
        X = accum.reindex(columns=use_cols, fill_value=0).astype(float)

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
