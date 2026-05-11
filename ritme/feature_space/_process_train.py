from typing import List, NamedTuple, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ritme.feature_space.aggregate_features import aggregate_microbial_features
from ritme.feature_space.enrich_features import enrich_features
from ritme.feature_space.select_features import select_microbial_features
from ritme.feature_space.transform_features import (
    _find_most_nonzero_feature_idx,
    transform_microbial_features,
)
from ritme.feature_space.utils import _add_suffix, _extract_time_labels, _slice_snapshot
from ritme.split_train_test import _make_kfold_splitter, _split_data_grouped


# Bundle of artifacts returned by ``process_train_kfold``. ``fold_indices``
# holds (train_idx, val_idx) integer-array pairs; downstream Ray-remote fold
# dispatch slices ``X_full`` / ``y_full`` inside each task to avoid pickling
# pre-sliced numpy arrays across actors.
class KFoldEngineered(NamedTuple):
    X_full: np.ndarray
    y_full: np.ndarray
    ft_ls_used: List[str]
    fold_indices: List[Tuple[np.ndarray, np.ndarray]]


def _engineer_features(
    config,
    train_val: pd.DataFrame,
    tax: pd.DataFrame,
) -> tuple[pd.DataFrame, List[str]]:
    """Run the per-snapshot aggregate/select/transform/enrich pipeline.

    Returns the engineered design matrix (indexed like ``train_val``) and the
    list of feature columns used. Mutates ``config`` to record the ALR denominator
    map when applicable.
    """
    feat_prefix = "F"
    microbial_ft_ls = [x for x in train_val if x.startswith(feat_prefix)]

    # Determine time snapshots
    time_labels = _extract_time_labels(microbial_ft_ls, feat_prefix)

    # Accumulator for per-snapshot enriched features (starts empty)
    train_val_accum = pd.DataFrame(index=train_val.index).copy()
    microbial_ft_ls_transf_all: List[str] = []
    other_ft_ls_all: List[str] = []

    # Per-snapshot: aggregate -> select -> transform -> enrich (unsuffixed)
    denom_idx_map = {}
    for tlabel in time_labels:
        # Slice full snapshot (microbial + metadata), unsuffixed
        snap_all = _slice_snapshot(train_val, tlabel)
        snap_micro_ls = [c for c in snap_all.columns if c.startswith(feat_prefix)]
        snap_ft_unsuff = snap_all[snap_micro_ls]
        snap_md_unsuff = snap_all.drop(columns=snap_micro_ls)

        # Identify rows where all microbial features are NaN (missing past obs)
        nan_mask = snap_ft_unsuff.isna().all(axis=1)
        has_nan_rows = nan_mask.any()
        if has_nan_rows:
            real_ft = snap_ft_unsuff[~nan_mask]
            real_all = snap_all[~nan_mask]
            real_md = snap_md_unsuff[~nan_mask]
        else:
            real_ft = snap_ft_unsuff
            real_all = snap_all
            real_md = snap_md_unsuff

        # AGGREGATE (unsuffixed, real rows only)
        snap_agg = aggregate_microbial_features(
            real_ft, config["data_aggregation"], tax
        )

        # SELECT (unsuffixed)
        snap_sel = select_microbial_features(snap_agg, config, feat_prefix)

        # TRANSFORM (unsuffixed). If only one feature, skip transform.
        if len(snap_sel.columns) == 1:
            snap_transf = snap_sel.copy()
            denom_idx = None
        else:
            denom_idx = (
                _find_most_nonzero_feature_idx(snap_sel)
                if config["data_transform"] == "alr"
                else None
            )
            if config["data_transform"] == "alr":
                denom_idx_map[tlabel] = denom_idx
            snap_transf = transform_microbial_features(
                snap_sel, config["data_transform"], denom_idx
            )

        # ENRICH on unsuffixed snapshot (real rows only)
        other_ft_ls_snap_unsuff, enriched_unsuff = enrich_features(
            real_all,
            snap_micro_ls,
            snap_transf.join(real_md),
            config,
        )

        # Reintroduce NaN rows with the same columns
        if has_nan_rows:
            nan_placeholder = pd.DataFrame(
                np.nan, index=snap_all.index[nan_mask], columns=enriched_unsuff.columns
            )
            enriched_unsuff = pd.concat([enriched_unsuff, nan_placeholder]).loc[
                snap_all.index
            ]

        # Suffix once for the whole enriched snapshot (t0 stays unsuffixed)
        enriched_suff = _add_suffix(enriched_unsuff, tlabel)

        # Track transformed microbial and other features
        if tlabel == "t0":
            microbial_ft_ls_transf_all.extend(snap_transf.columns.tolist())
            other_ft_ls_all.extend(other_ft_ls_snap_unsuff)
        else:
            microbial_ft_ls_transf_all.extend(
                [f"{c}__{tlabel}" for c in snap_transf.columns]
            )
            other_ft_ls_all.extend([f"{c}__{tlabel}" for c in other_ft_ls_snap_unsuff])

        # Append enriched suffixed snapshot to accumulator
        train_val_accum = train_val_accum.join(enriched_suff, how="left")

    if config["data_transform"] == "alr":
        config["data_alr_denom_idx_map"] = denom_idx_map

    # Final features used
    ft_ls_used = microbial_ft_ls_transf_all + other_ft_ls_all

    return train_val_accum, ft_ls_used


def _encode_target(
    config, train_val: pd.DataFrame, target: str, target_subset: pd.Series
) -> np.ndarray:
    """Encode target column to a float numpy array, fitting a LabelEncoder for
    non-numeric targets (and stashing it in ``config['_label_encoder']``)."""
    if pd.api.types.is_numeric_dtype(train_val[target]):
        return target_subset.fillna(np.nan).astype(float).values
    le = config.get("_label_encoder")
    if le is None:
        le = LabelEncoder()
        le.fit(train_val[target].dropna())
        config["_label_encoder"] = le
    return le.transform(target_subset).astype(float)


def process_train(
    config,
    train_val: pd.DataFrame,
    target: str,
    host_id: str,
    tax: pd.DataFrame,
    seed_data: int,
    stratify_by: list[str] | None = None,
) -> tuple:
    """Process training data per snapshot (aggregate/select/transform/enrich) and
    perform a grouped (and optionally stratified) split.
    """
    train_val_accum, ft_ls_used = _engineer_features(config, train_val, tax)

    # SPLIT
    train, val = _split_data_grouped(
        train_val_accum,
        host_id,
        0.8,
        seed_data,
        stratify_by=stratify_by,
    )

    X_train = train[ft_ls_used].fillna(np.nan).astype(float)
    X_val = val[ft_ls_used].fillna(np.nan).astype(float)

    y_train = _encode_target(config, train_val, target, train[target])
    y_val = _encode_target(config, train_val, target, val[target])

    return (X_train.values, y_train, X_val.values, y_val)


def process_train_kfold(
    config,
    train_val: pd.DataFrame,
    target: str,
    host_id: str,
    tax: pd.DataFrame,
    seed_data: int,
    n_splits: int,
    stratify_by: list[str] | None = None,
) -> KFoldEngineered:
    """K-fold variant of :func:`process_train`.

    Engineers features once on the full ``train_val`` (same pipeline as
    ``process_train``) and produces the integer-index fold pairs for the
    appropriate scikit-learn K-fold splitter (group / stratified / both,
    honoring all the constraints of :func:`_split_data_grouped`). Callers
    slice ``X_full`` / ``y_full`` themselves (typically inside Ray-remote
    fold tasks) to avoid pickling K pre-sliced numpy arrays.

    Returns
    -------
    KFoldEngineered
        A NamedTuple with fields:

        - ``X_full`` (np.ndarray): Engineered design matrix for the full
          ``train_val`` (also used to refit the chosen configuration on all
          data for the saved checkpoint).
        - ``y_full`` (np.ndarray): Encoded target for the full ``train_val``.
        - ``ft_ls_used`` (list[str]): Names of the feature columns used
          (for downstream serialization).
        - ``fold_indices`` (list[tuple[np.ndarray, np.ndarray]]): Each entry
          is the ``(train_idx, val_idx)`` pair of integer index arrays used
          to slice ``X_full`` / ``y_full`` for the corresponding fold.
    """
    train_val_accum, ft_ls_used = _engineer_features(config, train_val, tax)

    X_full = train_val_accum[ft_ls_used].fillna(np.nan).astype(float).values
    y_full = _encode_target(config, train_val, target, train_val[target])

    fold_indices = list(
        _make_kfold_splitter(
            train_val,
            group_by_column=host_id,
            stratify_by=stratify_by,
            n_splits=n_splits,
            seed=seed_data,
        )
    )

    return KFoldEngineered(
        X_full=X_full,
        y_full=y_full,
        ft_ls_used=ft_ls_used,
        fold_indices=fold_indices,
    )
