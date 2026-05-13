from typing import Any, Dict, List, NamedTuple, Optional, Tuple

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


# Bundle of artifacts returned by ``process_train_kfold``.
#
# ``folds`` holds K independently engineered ``(X_tr, y_tr, X_va, y_va)``
# tuples -- each fold's design matrices are produced by fitting feature
# engineering on the fold's train slice and applying the captured params on
# its val slice, so no cross-sample statistic crosses the train/val boundary
# (see ``issue_val_leakage.md`` for the leakage that this design eliminates).
# ``X_refit`` / ``y_refit`` are engineered on the full ``train_val`` and are
# used only for the post-loop deployable refit (where leakage is fine -- it
# is the model that gets shipped, not a validation-score model).
class KFoldEngineered(NamedTuple):
    folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    X_refit: np.ndarray
    y_refit: np.ndarray
    ft_ls_used: List[str]


# Frozen state captured during a fit pass of ``_engineer_features``. Keyed
# by time-snapshot label. Each entry contains:
#   - ``selected_cols`` (list[str]): column order surviving
#     ``select_microbial_features`` on the fit slice (including the
#     synthetic ``F_low_*`` group column when grouping happened);
#   - ``denom_idx`` (int): ALR denominator column index in that post-select
#     matrix (present only when ``data_transform == "alr"``).
# Both are reused by the apply pass on the val slice; any other engineering
# step (aggregate, transform CLR/ILR/PA/rank, enrich) is row-local and
# needs no fit state.
SnapshotFrozenParams = Dict[str, Any]
FrozenEngineeringParams = Dict[str, SnapshotFrozenParams]


def _apply_frozen_selection(
    snap_agg: pd.DataFrame,
    method: Optional[str],
    selected_cols: List[str],
    ft_prefix: str,
) -> pd.DataFrame:
    """Reapply a captured ``select_microbial_features`` result on a fresh slice.

    Mirrors the apply branch of :meth:`TunedModel.select`: features that were
    grouped into ``F_low_abun`` / ``F_low_var`` on the fit slice are summed
    again on the new slice; columns kept during fit but absent from the new
    slice are filled with 0; column order matches the captured list.
    """
    sel_cols = list(selected_cols)
    if method is None:
        return snap_agg.reindex(columns=sel_cols, fill_value=0)

    ft_to_sum = [c for c in snap_agg.columns if c not in sel_cols]
    if ft_to_sum:
        group_name = "_low_abun" if method.startswith("abundance") else "_low_var"
        group_col = f"{ft_prefix}{group_name}"
        snap_agg = snap_agg.copy()
        snap_agg[group_col] = snap_agg[ft_to_sum].sum(axis=1)
    return snap_agg.reindex(columns=sel_cols, fill_value=0)


def _engineer_features(
    config,
    train_val: pd.DataFrame,
    tax: pd.DataFrame,
    frozen_params: Optional[FrozenEngineeringParams] = None,
) -> Tuple[pd.DataFrame, List[str], FrozenEngineeringParams]:
    """Run the per-snapshot aggregate/select/transform/enrich pipeline.

    Two modes, controlled by ``frozen_params``:

    - **Fit** (``frozen_params is None``): computes cross-sample statistics
      (feature selection survivors, ALR denominator index) on this slice and
      returns them as the third tuple element. Callers retain the returned
      dict to reapply the same engineering on a held-out slice without
      cross-sample leakage.
    - **Apply** (``frozen_params`` is a fit-produced dict): skips the
      cross-sample fit steps and reuses the captured selection / ALR
      denominator. Row-local transforms (CLR / ILR / PA / rank) and the
      tax-driven aggregate step run as in fit mode.

    Returns the engineered design matrix (indexed like ``train_val``), the
    list of feature columns used, and the frozen-params dict.

    Side effect (compat shim): the ALR denominator map is also stashed on
    ``config['data_alr_denom_idx_map']`` so downstream code that reads it
    from the config (TunedModel test-time path) keeps working.
    """
    feat_prefix = "F"
    fit_mode = frozen_params is None
    captured_params: FrozenEngineeringParams = {} if fit_mode else dict(frozen_params)

    microbial_ft_ls = [x for x in train_val if x.startswith(feat_prefix)]

    # Determine time snapshots
    time_labels = _extract_time_labels(microbial_ft_ls, feat_prefix)

    # Accumulator for per-snapshot enriched features (starts empty)
    train_val_accum = pd.DataFrame(index=train_val.index).copy()
    microbial_ft_ls_transf_all: List[str] = []
    other_ft_ls_all: List[str] = []

    # Per-snapshot: aggregate -> select -> transform -> enrich (unsuffixed)
    denom_idx_map: Dict[str, int] = {}
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

        # AGGREGATE (unsuffixed, real rows only) -- tax-driven, no fit state.
        snap_agg = aggregate_microbial_features(
            real_ft, config["data_aggregation"], tax
        )

        # SELECT (unsuffixed) -- the first cross-sample fit step. In fit
        # mode we run selection and capture the surviving column order;
        # in apply mode we reuse the captured columns (and reproduce the
        # ``F_low_*`` grouping on this slice).
        if fit_mode:
            snap_sel = select_microbial_features(snap_agg, config, feat_prefix)
            captured_params[tlabel] = {
                "selected_cols": snap_sel.columns.tolist(),
            }
        else:
            snap_params = captured_params.get(tlabel, {})
            snap_sel = _apply_frozen_selection(
                snap_agg,
                config.get("data_selection"),
                snap_params.get("selected_cols", snap_agg.columns.tolist()),
                feat_prefix,
            )

        # TRANSFORM (unsuffixed). If only one feature, skip transform.
        if len(snap_sel.columns) == 1:
            snap_transf = snap_sel.copy()
            denom_idx = None
        else:
            if config["data_transform"] == "alr":
                if fit_mode:
                    denom_idx = _find_most_nonzero_feature_idx(snap_sel)
                    captured_params[tlabel]["denom_idx"] = denom_idx
                else:
                    denom_idx = captured_params.get(tlabel, {}).get("denom_idx")
                    if denom_idx is None:
                        denom_idx = _find_most_nonzero_feature_idx(snap_sel)
                denom_idx_map[tlabel] = denom_idx
            else:
                denom_idx = None
            snap_transf = transform_microbial_features(
                snap_sel, config["data_transform"], denom_idx
            )

        # ENRICH on unsuffixed snapshot (real rows only) -- metadata-driven,
        # no microbial cross-sample fit state.
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

    return train_val_accum, ft_ls_used, captured_params


def _stash_enrich_categories(config: dict, train_val: pd.DataFrame) -> None:
    """Record the full-data categorical universe for each ``data_enrich_with`` feat.

    Stashes ``config["_enrich_categories"] = {feat: sorted_unique_values}`` so
    every downstream call to :func:`enrich_features` (per-fold train, per-fold
    val, full-data refit, predict-time test) sees the same set of categories
    and produces aligned dummy columns with a stable reference category.
    Without this, ``pd.get_dummies(drop_first=True)`` derives the column set
    per slice, and small or imbalanced data can produce a train-fold dummy
    column that the val fold does not have -- which raises ``KeyError`` at
    column alignment time (see ``issue_kfold.md``).

    Numeric features in ``data_enrich_with`` are skipped (they don't produce
    dummies). The stash is a no-op when ``data_enrich_with`` is None / empty.
    """
    feats = config.get("data_enrich_with")
    if not feats:
        return
    universes: Dict[str, List[Any]] = {}
    for feat in feats:
        if feat not in train_val.columns:
            continue
        col = train_val[feat]
        if isinstance(col.dtype, pd.CategoricalDtype) or col.dtype == object:
            universes[feat] = sorted(col.dropna().unique().tolist())
    if universes:
        config["_enrich_categories"] = universes


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
    """Split first, then engineer features per slice to avoid train/val leakage.

    The split is computed on the raw ``train_val`` (engineering does not
    change which rows belong to train vs val). Feature engineering then runs
    in fit mode on the train slice (capturing selection survivors + ALR
    denominator) and in apply mode on the val slice with the same captured
    params. Cross-sample statistics therefore never cross the train/val
    boundary.

    Side effect: when the target column is non-numeric, fits a ``LabelEncoder``
    and stashes it on ``config['_label_encoder']`` so the val slice (and
    downstream model persistence) share the same label-to-int map. The
    trainable's ``_save_label_encoder`` consumes and removes this entry.
    """
    _stash_enrich_categories(config, train_val)

    train_raw, val_raw = _split_data_grouped(
        train_val,
        host_id,
        0.8,
        seed_data,
        stratify_by=stratify_by,
    )

    train_engineered, ft_ls_used, frozen_params = _engineer_features(
        config, train_raw, tax
    )
    val_engineered, _, _ = _engineer_features(
        config, val_raw, tax, frozen_params=frozen_params
    )

    X_train = train_engineered[ft_ls_used].fillna(np.nan).astype(float)
    X_val = val_engineered[ft_ls_used].fillna(np.nan).astype(float)

    y_train = _encode_target(config, train_val, target, train_raw[target])
    y_val = _encode_target(config, train_val, target, val_raw[target])

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
    """K-fold variant of :func:`process_train` with per-fold engineering.

    For each fold: fits feature engineering on the train slice (capturing
    selection survivors + ALR denominator) and applies the captured params
    on the val slice. The K folds therefore use K independent
    engineering-parameter sets and cross-sample statistics never cross any
    fold's train/val boundary.

    A separate full-data engineering pass produces ``X_refit`` / ``y_refit``
    for the post-loop deployable refit; leakage there is fine since that
    matrix is used only to refit on all of ``train_val`` (the model that
    gets shipped), not to score validation.

    Side effect: for non-numeric targets, fits a ``LabelEncoder`` and stashes
    it on ``config['_label_encoder']`` (consumed by the trainable's
    ``_save_label_encoder``).

    Returns
    -------
    KFoldEngineered
        A NamedTuple with fields:

        - ``folds`` (list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
          One ``(X_tr, y_tr, X_va, y_va)`` tuple per fold, each fold
          engineered independently with no train/val leakage.
        - ``X_refit`` (np.ndarray): Design matrix engineered on the full
          ``train_val`` (used for the post-loop deployable refit).
        - ``y_refit`` (np.ndarray): Encoded target for the full ``train_val``.
        - ``ft_ls_used`` (list[str]): Column names of ``X_refit`` (the
          deployable refit's column space). Per-fold matrices may carry
          different column sets when ``data_selection`` is data-dependent
          (each fold's surviving features differ); within-fold X_tr/X_va
          alignment is preserved by the engineering itself and does not
          depend on ``ft_ls_used``.
    """
    _stash_enrich_categories(config, train_val)

    fold_indices = list(
        _make_kfold_splitter(
            train_val,
            group_by_column=host_id,
            stratify_by=stratify_by,
            n_splits=n_splits,
            seed=seed_data,
        )
    )

    # Per-fold engineering: fit on the train slice, apply on the val slice
    # with the captured params. Per-fold column lists may differ from each
    # other and from the refit list whenever ``data_selection`` produces
    # data-dependent survivors (e.g. abundance / variance thresholds), so
    # each fold tracks its own ``ft_ls`` for the within-fold X_tr/X_va
    # alignment.
    folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for tr_idx, va_idx in fold_indices:
        train_raw = train_val.iloc[tr_idx]
        val_raw = train_val.iloc[va_idx]

        train_engineered, fold_ft_ls, frozen_params = _engineer_features(
            config, train_raw, tax
        )
        val_engineered, _, _ = _engineer_features(
            config, val_raw, tax, frozen_params=frozen_params
        )

        X_tr = train_engineered[fold_ft_ls].fillna(np.nan).astype(float).values
        X_va = val_engineered[fold_ft_ls].fillna(np.nan).astype(float).values
        y_tr = _encode_target(config, train_val, target, train_raw[target])
        y_va = _encode_target(config, train_val, target, val_raw[target])
        folds.append((X_tr, y_tr, X_va, y_va))

    # Full-data refit engineering -- last so the final ``config`` ALR
    # denom map matches the deployable artifact (TunedModel test-time
    # path reads ``config['data_alr_denom_idx_map']``). ``ft_ls_used`` in
    # the returned bundle describes the refit matrix (the deployable
    # checkpoint's column space) and may differ from any individual
    # fold's column list.
    refit_engineered, refit_ft_ls, _ = _engineer_features(config, train_val, tax)
    X_refit = refit_engineered[refit_ft_ls].fillna(np.nan).astype(float).values
    y_refit = _encode_target(config, train_val, target, train_val[target])

    return KFoldEngineered(
        folds=folds,
        X_refit=X_refit,
        y_refit=y_refit,
        ft_ls_used=refit_ft_ls,
    )
