from typing import List

import pandas as pd

from ritme.feature_space.aggregate_features import aggregate_microbial_features
from ritme.feature_space.enrich_features import enrich_features
from ritme.feature_space.select_features import select_microbial_features
from ritme.feature_space.transform_features import (
    _find_most_nonzero_feature_idx,
    transform_microbial_features,
)
from ritme.feature_space.utils import _extract_time_labels
from ritme.split_train_test import _split_data_grouped


def _slice_snapshot(ft_all: pd.DataFrame, time_label: str) -> pd.DataFrame:
    """
    Return snapshot slice of microbial features and strip time suffix if present.
    """
    suffix = f"__{time_label}"
    cols = [c for c in ft_all.columns if c.endswith(suffix)]
    snap = ft_all[cols].copy()
    # strip suffix
    snap.columns = [c[: -len(suffix)] for c in cols]
    return snap


def _add_suffix(df: pd.DataFrame, time_label: str) -> pd.DataFrame:
    df_s = df.copy()
    df_s.columns = [f"{c}__{time_label}" for c in df_s.columns]
    return df_s


def process_train(config, train_val, target, host_id, tax, seed_data):
    feat_prefix = "F"
    microbial_ft_ls = [x for x in train_val if x.startswith(feat_prefix)]

    # Determine time snapshots
    time_labels = _extract_time_labels(microbial_ft_ls, feat_prefix)

    # Accumulator for per-snapshot enriched features (starts empty)
    train_val_accum = pd.DataFrame(index=train_val.index).copy()
    microbial_ft_ls_transf_all: List[str] = []
    other_ft_ls_all: List[str] = []

    # Per-snapshot: aggregate -> select -> transform -> enrich
    denom_idx_map = {}
    for tlabel in time_labels:
        # Slice raw microbial features for this snapshot
        snap_raw = _slice_snapshot(train_val[microbial_ft_ls], tlabel)

        # AGGREGATE
        snap_agg = aggregate_microbial_features(
            snap_raw, config["data_aggregation"], tax
        )

        # SELECT
        snap_sel = select_microbial_features(snap_agg, config, feat_prefix)

        # TRANSFORM. If only one feature, skip transform.
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

        # Add time suffix to transformed columns
        snap_transf_s = _add_suffix(snap_transf, tlabel)
        microbial_ft_ls_transf_all.extend(snap_transf_s.columns.tolist())
        # Join transformed snapshot with metadata of the same snapshot for enrichment
        md_cols_t = [
            c
            for c in train_val.columns
            if (not c.startswith(feat_prefix)) and c.endswith(f"__{tlabel}")
        ]
        train_val_t_snap = train_val[md_cols_t].join(snap_transf_s)

        # Build list of raw microbial features (with suffix) for this snapshot
        microbial_ft_ls_t = [
            c
            for c in train_val.columns
            if c.startswith(feat_prefix) and c.endswith(f"__{tlabel}")
        ]

        # ENRICH per snapshot (Shannon + metadata as configured)
        other_ft_ls_snap, enriched_snap = enrich_features(
            train_val, microbial_ft_ls_t, train_val_t_snap, config
        )

        # Suffix newly created generic columns (e.g., shannon_entropy) with time
        # label add add same columns to newly created dataframe enriched_snap
        rename_map = {}
        for col in other_ft_ls_snap:
            if "__t" not in col:
                rename_map[col] = f"{col}__{tlabel}"
        if rename_map:
            enriched_snap = enriched_snap.rename(columns=rename_map)
            other_ft_ls_snap = [rename_map.get(c, c) for c in other_ft_ls_snap]

        # Append all enriched snapshot columns - we rely on prior suffixing to
        # keep names unique per snapshot.
        train_val_accum = train_val_accum.join(enriched_snap, how="left")

        # Extend other feature list
        other_ft_ls_all.extend(other_ft_ls_snap)

    if config["data_transform"] == "alr":
        config["data_alr_denom_idx_map"] = denom_idx_map

    # Final features used
    ft_ls_used = microbial_ft_ls_transf_all + other_ft_ls_all

    # SPLIT
    train, val = _split_data_grouped(train_val_accum, host_id, 0.8, seed_data)
    X_train, y_train = train[ft_ls_used].astype(float), train[target].astype(float)
    X_val, y_val = val[ft_ls_used].astype(float), val[target].astype(float)
    return (X_train.values, y_train.values, X_val.values, y_val.values)
