from typing import List

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
from ritme.split_train_test import _split_data_grouped


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

    if pd.api.types.is_numeric_dtype(train_val[target]):
        y_train = train[target].fillna(np.nan).astype(float).values
        y_val = val[target].fillna(np.nan).astype(float).values
    else:
        le = LabelEncoder()
        le.fit(train_val[target].dropna())
        config["_label_encoder"] = le
        y_train = le.transform(train[target]).astype(float)
        y_val = le.transform(val[target]).astype(float)

    return (X_train.values, y_train, X_val.values, y_val)
