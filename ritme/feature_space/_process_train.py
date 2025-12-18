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
    """Return snapshot slice of all columns and strip time suffix if present."""
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

        # AGGREGATE (unsuffixed)
        snap_agg = aggregate_microbial_features(
            snap_ft_unsuff, config["data_aggregation"], tax
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

        # ENRICH on unsuffixed snapshot
        other_ft_ls_snap_unsuff, enriched_unsuff = enrich_features(
            snap_all,
            snap_micro_ls,
            snap_transf.join(snap_md_unsuff),
            config,
        )

        # Suffix once for the whole enriched snapshot
        enriched_suff = _add_suffix(enriched_unsuff, tlabel)

        # Track transformed microbial and other features (suffixed)
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

    X_train = train[ft_ls_used].astype(float)
    y_train = train[target].astype(float)
    X_val = val[ft_ls_used].astype(float)
    y_val = val[target].astype(float)
    return (X_train.values, y_train.values, X_val.values, y_val.values)
