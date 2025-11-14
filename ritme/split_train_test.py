import os
import warnings
from typing import List, Sequence, Tuple

import pandas as pd
import qiime2 as q2
import typer
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedShuffleSplit,
    train_test_split,
)

from ritme._decorators import helper_function, main_function
from ritme.feature_space.utils import _biom_to_df, _df_to_biom


# ----------------------------------------------------------------------------
@helper_function
def _ft_rename_microbial_features(
    ft: pd.DataFrame, feature_prefix: str
) -> pd.DataFrame:
    """Append a prefix to all microbial feature names.

    Parameters
    ----------
    ft : pd.DataFrame
        Feature table with columns denoting microbial features.
    feature_prefix : str
        Prefix to prepend to each feature id (e.g., "F").

    Returns
    -------
    pd.DataFrame
        Copy of ``ft`` with renamed columns.
    """
    ft_renamed = ft.copy()
    ft_renamed.columns = [f"{feature_prefix}{i}" for i in ft.columns.tolist()]
    return ft_renamed


@helper_function
def _ft_remove_zero_features(ft: pd.DataFrame) -> pd.DataFrame:
    """Remove features that are all zero.

    Emits a warning listing the dropped features when any are removed.
    """
    ft_removed = ft.copy()
    drop_fts = ft_removed.loc[:, ft_removed.sum(axis=0) == 0.0].columns.tolist()
    if len(drop_fts) > 0:
        warnings.warn(f"Dropping these features with all zero values: {drop_fts}")
        ft_removed.drop(columns=drop_fts, inplace=True)
    else:
        ft_removed = ft.copy()
    return ft_removed


@helper_function
def _ft_get_relative_abundance(ft: pd.DataFrame) -> pd.DataFrame:
    """Transform a feature table from absolute to relative abundances.

    Notes
    -----
    Uses BIOM's ``Table.norm(axis="sample")`` for normalization.
    Asserts that each row sums to 1.0 (rounded to 3 decimals).
    """

    # biom.norm faster than skbio.stats.composition.closure
    ft_biom = _df_to_biom(ft)

    # Normalize the feature table using biom.Table.norm()
    ft_rel_biom = ft_biom.norm(axis="sample", inplace=False)

    # Convert the normalized biom.Table back to a pandas DataFrame
    ft_rel = _biom_to_df(ft_rel_biom)

    # round needed as certain 1.0 are represented in different digits 2e-16
    assert ft_rel.sum(axis=1).round(3).eq(1.0).all()

    return ft_rel


# ----------------------------------------------------------------------------
# Multi-snapshot helpers


@helper_function
def _verify_identical_indices(df_list: Sequence[pd.DataFrame], kind: str) -> None:
    """
    Verify that all dataframes in df_list have identical indices.
    Raises a ValueError if a mismatch is found.
    """
    if not df_list:
        raise ValueError(f"No {kind} dataframes provided.")
    base_index = df_list[0].index
    for i, df in enumerate(df_list[1:], start=1):
        if not base_index.equals(df.index):
            raise ValueError(
                f"Indices of provided {kind} dataframe at position {i} do not "
                "match the first one."
            )


@helper_function
def _generate_time_labels(n: int) -> List[str]:
    """Generate time labels ['t0', 't-1', 't-2', ...] for n snapshots."""
    return ["t0"] + [f"t-{i}" for i in range(1, n)]


@helper_function
def _append_time_suffix_to_features(ft: pd.DataFrame, time_label: str) -> pd.DataFrame:
    """
    Append a time suffix to microbial feature columns keeping 'F' prefix first,
    e.g., 'F123' -> 'F123__t0'.
    """
    ft_suffixed = ft.copy()
    ft_suffixed.columns = [f"{col}__{time_label}" for col in ft_suffixed.columns]
    return ft_suffixed


@helper_function
def _prepare_single_feature_table(ft: pd.DataFrame) -> pd.DataFrame:
    """Apply standard preprocessing to a feature table."""
    ft_prep = _ft_rename_microbial_features(ft, "F")
    ft_prep = _ft_remove_zero_features(ft_prep)
    relative_abundances = ft_prep.sum(axis=1).round(3).eq(1.0).all()
    if not relative_abundances:
        warnings.warn(
            "Provided feature table contains absolute instead of relative abundances. "
            "Hence, converting it to relative abundances..."
        )
        ft_prep = _ft_get_relative_abundance(ft_prep)
    return ft_prep


@helper_function
def _merge_time_snapshots(
    md_list: Sequence[pd.DataFrame], ft_list: Sequence[pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merge per-snapshot metadata and feature tables with time suffixes.

    Assumptions
    -----------
    - ``md_list`` and ``ft_list`` have identical indices across snapshots.
    - Input order corresponds to time labels: ``t0, t-1, t-2, ...``.
    - Feature column sets are identical across snapshots.

    Behavior
    --------
    - Each feature table is preprocessed (rename with prefix "F", removal of
      all-zero features, relative abundance conversion if needed) then suffixed
      with its time label (e.g. ``F123__t0``).
    - Metadata columns are suffixed with their time label (e.g. ``age__t-1``).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (merged_metadata, merged_features) with time-suffixed columns.
    """
    if len(md_list) != len(ft_list):
        raise ValueError(
            "Number of metadata files must match number of feature tables."
        )
    _verify_identical_indices(md_list, kind="metadata")
    _verify_identical_indices(ft_list, kind="feature table")

    base_cols = set(ft_list[0].columns)
    for i, ft in enumerate(ft_list[1:], start=1):
        if set(ft.columns) != base_cols:
            raise ValueError(
                "Feature column sets must match across all snapshots; mismatch "
                f"found at position {i}."
            )

    time_labels = _generate_time_labels(len(ft_list))
    ft_prepared: List[pd.DataFrame] = []
    md_prepared: List[pd.DataFrame] = []
    for ft, tlabel in zip(ft_list, time_labels):
        ft_s = _append_time_suffix_to_features(ft, tlabel)
        ft_prepared.append(ft_s)
    for md, tlabel in zip(md_list, time_labels):
        md_s = md.copy()
        md_s.columns = [f"{col}__{tlabel}" for col in md_s.columns]
        md_prepared.append(md_s)

    merged_ft = pd.concat(ft_prepared, axis=1)
    merged_md = pd.concat(md_prepared, axis=1)
    return merged_md, merged_ft


@helper_function
def _load_data(path2md: str, path2ft: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load metadata and feature table from disk.

    Parameters
    ----------
    path2md : str
        Path to a tab-delimited metadata file (``.tsv``) with sample ids in the
        index column.
    path2ft : str
        Path to a feature table as ``.tsv`` (tab-delimited, samples as rows)
        or a QIIME2 artifact ``.qza`` that can be viewed as a DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (metadata, feature_table).
    """
    # read metadata
    md = pd.read_csv(path2md, sep="\t", index_col=0)

    # read feature table
    if path2ft.endswith(".tsv"):
        ft = pd.read_csv(path2ft, sep="\t", index_col=0)
    elif path2ft.endswith(".qza"):
        ft = q2.Artifact.load(path2ft).view(pd.DataFrame)
    return md, ft


@helper_function
def _generate_host_time_snapshots_from_df(
    md: pd.DataFrame,
    ft: pd.DataFrame,
    time_col: str,
    host_col: str,
    s: int,
    feature_prefix: str = "F",
    missing_mode: str = "nan",
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """Generate per-host sliding-window snapshots: [t0, t-1, ..., t-s].

    Each returned snapshot pair (metadata and features) shares the same index:
    the sample ids corresponding to each t0 instance (one per host-time).
    This aligns rows across snapshots for downstream modeling and allows every
    observed time to act as its own t0.

    Important
    ---------
    - Snapshot times are computed from the numeric values in ``time_col``.
    - When a target past time is missing:
        * ``missing_mode='nan'`` retains the row, filling features and metadata
          with ``NA`` (host id and the computed target time are still set).
        * ``missing_mode='exclude'`` omits that t0 instance unless all required
          past times are present.
    - Feature snapshots are preprocessed via :func:`_prepare_single_feature_table`
      (prefix "F", remove all-zero features, convert to relative if needed) and
      include only microbial feature columns (prefixed with ``feature_prefix``).
    - Metadata snapshots keep the original metadata columns.

    Parameters
    ----------
    md, ft : pd.DataFrame
        Metadata and feature table with identical indices (sample ids).
    time_col : str
        Column in ``md`` with numeric time ordering (integers recommended).
    host_col : str
        Column in ``md`` identifying host/trajectory.
    s : int
        Number of previous snapshots to include (total snapshots = ``s+1``).
    feature_prefix : str, default "F"
        Prefix identifying microbial features after preprocessing.
    missing_mode : {"exclude", "nan"}
        Handling of missing earlier snapshots.

    Returns
    -------
    tuple[list[pd.DataFrame], list[pd.DataFrame]]
        Lists of metadata and feature DataFrames for offsets ``0..s``.
    """
    if missing_mode not in {"exclude", "nan"}:
        raise ValueError("missing_mode must be one of {'exclude','nan'}")
    if time_col not in md.columns or host_col not in md.columns:
        raise ValueError(
            f"Metadata must contain columns '{time_col}' and '{host_col}'."
        )
    if not md.index.equals(ft.index):
        warnings.warn(
            "Metadata and feature table indices must match -"
            "Merging is performed to achieve this."
        )
        common_idx = md.index.intersection(ft.index)
        md = md.loc[common_idx]
        ft = ft.loc[common_idx]

    ft_prepared = _prepare_single_feature_table(ft)
    microbial_cols = [c for c in ft_prepared.columns if c.startswith(feature_prefix)]

    # ! Build per-host mapping time -> sample index (numeric times required)
    # ! -> host_time_to_idx
    # ! -> host_all_times
    host_groups = md.groupby(host_col, sort=False)
    host_time_to_idx: dict[str, dict[int, str]] = {}
    host_all_times: dict[str, List[int]] = {}
    for host, sub_md in host_groups:
        times_num = pd.to_numeric(sub_md[time_col], errors="coerce")
        if times_num.isna().any():
            raise ValueError(
                "Non-numeric times detected for host "
                f"'{host}'. Provide integer/float times for sliding window generation."
            )
        # Preserve order ascending
        ordered = sorted(set(int(t) for t in times_num.tolist()))
        mapping: dict[int, str] = {}
        # If duplicate times exist, keep last occurrence (overwrite semantics)
        for idx, tval in zip(sub_md.index, times_num):
            mapping[int(tval)] = idx
        host_time_to_idx[host] = mapping
        host_all_times[host] = ordered

    # ! Construct list of (host, t0_time) instances (each time value becomes a t0)
    # ! -> instance_specs
    instance_specs: List[tuple[str, int]] = []
    for host, times_list in host_all_times.items():
        for t0_time in times_list:  # every time point acts as its own t0
            # Determine required contiguous window times
            target_times = [t0_time - i for i in range(0, s + 1)]
            if missing_mode == "exclude" and not all(
                tt in host_time_to_idx[host] for tt in target_times
            ):
                # Skip this t0 instance if any required past time missing
                continue
            instance_specs.append((host, t0_time))

    if missing_mode == "exclude" and len(instance_specs) == 0:
        raise ValueError(
            "No (host, t0) instances have complete contiguous windows for requested s."
        )

    # Canonical index: use sample id corresponding to each t0 instance
    canonical_index: List[str] = [
        host_time_to_idx[host][t0_time] for host, t0_time in instance_specs
    ]

    meta_snapshots: List[pd.DataFrame] = []
    ft_snapshots: List[pd.DataFrame] = []
    # Build snapshots for offsets 0..s (0 = t0)
    for offset in range(s + 1):
        md_rows: List[pd.Series] = []
        ft_rows: List[pd.Series] = []
        for host, t0_time in instance_specs:
            target_time = t0_time - offset
            sample_idx = host_time_to_idx[host].get(target_time)
            if sample_idx is not None:
                md_rows.append(md.loc[sample_idx].copy())
                ft_rows.append(ft_prepared.loc[sample_idx, microbial_cols].copy())
            else:
                md_row = pd.Series({c: pd.NA for c in md.columns})
                md_row[host_col] = host
                md_row[time_col] = target_time
                md_rows.append(md_row)
                ft_rows.append(pd.Series({c: pd.NA for c in microbial_cols}))
        md_snap = pd.DataFrame(md_rows, index=canonical_index)
        ft_snap = pd.DataFrame(ft_rows, index=canonical_index, columns=microbial_cols)
        meta_snapshots.append(md_snap)
        ft_snapshots.append(ft_snap)

    return meta_snapshots, ft_snapshots


@helper_function
def _resolve_column_to_t0(df: pd.DataFrame, col: str | None) -> str | None:
    """Resolve an input column name to its '__t0' form if needed.

    Behavior
    --------
    - If ``col`` is None, return None.
    - If ``col`` exists in ``df.columns``, return it unchanged.
    - Else, try ``f"{col}__t0"``; if it exists, return the suffixed name.
    - Otherwise, raise ValueError indicating the column is not found.
    """
    if col is None:
        return None
    if col in df.columns:
        return col
    alt = f"{col}__t0"
    if alt in df.columns:
        return alt
    raise ValueError(f"Column '{col}' not found in data (nor as '{alt}').")


@helper_function
def _resolve_columns_to_t0(
    df: pd.DataFrame, cols: Sequence[str] | None
) -> list[str] | None:
    """Resolve a list of column names to their '__t0' forms when needed.

    Returns None if ``cols`` is None. Preserves order.
    """
    if cols is None:
        return None
    resolved: list[str] = []
    for c in cols:
        resolved.append(_resolve_column_to_t0(df, c))
    return resolved


@helper_function
def _split_data_grouped(
    data: pd.DataFrame,
    group_by_column: str,
    train_size: float,
    seed: int,
    stratify_by: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly split data into train and test sets, with optional grouping.

    Behavior
    --------
    - If ``train_size == 0.0``, no split is performed: train is empty (0 rows,
      same columns) and test contains all rows.
    - If ``group_by_column`` is provided, rows sharing a group value are kept
      together to prevent leakage.
    - Otherwise a standard random split is performed.
    - If ``stratify_by`` is provided (list of column names), stratified
        splitting is applied to preserve the joint distribution of the specified
        columns. When grouping is used, stratification happens at the group level
        (each group must have consistent values for the stratify columns).
    """
    # Special-case: return everything as test
    if train_size == 0.0:
        train = data.iloc[0:0].copy()
        test = data.copy()
        print(f"Train: {train.shape}, Test: {test.shape}")
        return train, test
    if stratify_by is not None and len(stratify_by) == 0:
        stratify_by = None

    if group_by_column is None:
        stratify_labels = None
        if stratify_by is not None:
            missing = [c for c in stratify_by if c not in data.columns]
            if missing:
                raise ValueError(
                    "Stratification columns not found in data: " + ", ".join(missing)
                )
            # Build composite label; use a unique separator unlikely to appear.
            stratify_labels = data[stratify_by].astype(str).agg("__§__".join, axis=1)
        train, test = train_test_split(
            data,
            train_size=train_size,
            random_state=seed,
            stratify=stratify_labels,
        )
    else:
        if len(data[group_by_column].unique()) == 1:
            raise ValueError(
                f"Only one unique value of '{group_by_column}' available in dataset."
            )
        if stratify_by is None:
            gss = GroupShuffleSplit(
                n_splits=1, train_size=train_size, random_state=seed
            )
            split = gss.split(data, groups=data[group_by_column])
            train_idx, test_idx = next(split)
            train, test = data.iloc[train_idx], data.iloc[test_idx]
        else:
            # Validate columns
            missing = [c for c in stratify_by if c not in data.columns]
            if missing:
                raise ValueError(
                    "Stratification columns not found in data: " + ", ".join(missing)
                )
            # Build group-level labels and ensure consistency within groups
            grp = data.groupby(group_by_column, sort=False)
            group_ids: List[str] = []
            group_labels: List[str] = []
            group_to_rows: dict[str, pd.Index] = {}
            for gid, sub in grp:
                label_series = sub[stratify_by].astype(str).agg("__§__".join, axis=1)
                unique_labels = label_series.unique().tolist()
                if len(unique_labels) != 1:
                    raise ValueError(
                        "Stratification columns must be constant within each group; "
                        f"found multiple values for group '{gid}'."
                    )
                group_ids.append(gid)
                group_labels.append(unique_labels[0])
                group_to_rows[gid] = sub.index

            # Perform stratified split on groups
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=train_size, random_state=seed
            )
            # Create a dummy X with same length as groups
            X = pd.Series(group_labels)
            y = pd.Series(group_labels)
            grp_train_idx, grp_test_idx = next(sss.split(X, y))

            train_groups = [group_ids[i] for i in grp_train_idx]
            test_groups = [group_ids[i] for i in grp_test_idx]

            train = data.loc[data[group_by_column].isin(train_groups)]
            test = data.loc[data[group_by_column].isin(test_groups)]

    print(f"Train: {train.shape}, Test: {test.shape}")
    return train, test


# ----------------------------------------------------------------------------
@main_function
def split_train_test(
    md: pd.DataFrame,
    ft: pd.DataFrame,
    group_by_column: str = None,
    train_size: float = 0.8,
    seed: int = 42,
    stratify_by: Sequence[str] | None = None,
    # Optional snapshot generation from single md/ft
    time_col: str | None = None,
    host_col: str | None = None,
    n_prev: int | None = None,
    missing_mode: str = "exclude",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare a modeling table and split into train/test sets.

    Behavior
    --------
    - When ``time_col``, ``host_col`` and ``n_prev`` are provided, the function
      first generates sliding-window snapshots per host/time using
      :func:`_generate_host_time_snapshots_from_df`, then merges them with
      time-suffixed columns (``__t0``, ``__t-1``, ...). Otherwise, it treats the
      input as a single ``t0`` snapshot and simply suffixes columns with
      ``__t0``.
    - Feature columns are always prefixed with ``F`` and preprocessed (all-zero
      removal, relative abundance conversion if needed).
    - If ``group_by_column`` does not exist in the suffixed table, its ``__t0``
      counterpart (e.g., ``host_id__t0``) is used for grouping.
        - If ``train_size == 0.0``, all processed rows are returned in the test
            set and the train set is empty.

    Parameters
    ----------
    md : pd.DataFrame
        Metadata table (samples as rows). When temporal arguments are provided,
        must contain ``time_col`` and ``host_col``.
    ft : pd.DataFrame
        Feature table (samples as rows), will be preprocessed.
    group_by_column : str, optional
        Column by which to group the split (e.g. ``host_id``). Prevents leakage
        across groups. Default ``None``.
    train_size : float, optional
        Proportion for the train split. Default ``0.8``.
    seed : int, optional
        Random seed. Default ``42``.
    stratify_by : Sequence[str], optional
        Columns to use for stratification. When grouping is used, the
        stratification is performed at the group level and the specified
        columns must be constant within each group. Column names can be
        provided without time suffix; they will be resolved to ``__t0`` if
        needed.
    time_col : str, optional
        Metadata column for time. Enables temporal snapshotting when set.
    host_col : str, optional
        Metadata column identifying host/trajectory. Required when
        ``time_col`` is set.
    n_prev : int, optional
        Number of previous snapshots to include (total snapshots = ``n_prev+1``)
        when temporal snapshotting is enabled.
    missing_mode : {"exclude", "nan"}
        Handling of missing earlier snapshots in temporal mode. Default
        ``"exclude"``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Train/validation and test DataFrames with time-suffixed columns.
    """
    # Expand into multiple snapshots if temporal parameters are provided;
    # otherwise treat as a single t0 snapshot.
    if time_col and host_col and n_prev is not None and n_prev >= 0:
        md_list, ft_list = _generate_host_time_snapshots_from_df(
            md,
            ft,
            time_col=time_col,
            host_col=host_col,
            s=n_prev,
            missing_mode=missing_mode,
        )
        md_base, ft_merged = _merge_time_snapshots(md_list, ft_list)
    else:
        time_label = "t0"
        md_base = md.copy()
        md_base.columns = [f"{c}__{time_label}" for c in md_base.columns]
        ft_single = _prepare_single_feature_table(ft)
        ft_merged = _append_time_suffix_to_features(ft_single, time_label)

    # merge md and feature table (inner join on sample ids)
    data = md_base.join(ft_merged, how="inner")

    if missing_mode == "exclude" and train_size != 0.0:
        # merging could add all-zero features - remove
        data = _ft_remove_zero_features(data)

    # Resolve group_by_column and stratify_by to suffixed names if needed
    # (e.g., host_id -> host_id__t0)
    group_col = _resolve_column_to_t0(data, group_by_column)
    strat_cols = _resolve_columns_to_t0(data, stratify_by)
    # split
    train_val, test = _split_data_grouped(
        data,
        group_by_column=group_col,
        train_size=train_size,
        seed=seed,
        stratify_by=strat_cols,
    )

    return train_val, test


@main_function
def cli_split_train_test(
    output_path: str,
    path_to_md: str,
    path_to_ft: str,
    group_by_column: str = None,
    train_size: float = 0.8,
    seed: int = 42,
    # Optional snapshot generation from single md/ft
    time_col: str = None,
    host_col: str = None,
    n_prev: int = None,
    missing_mode: str = None,
    stratify_by: str = None,
):
    """CLI wrapper for :func:`split_train_test` supporting temporal mode.

    Notes
    -----
    - ``path_to_md``: tab-delimited ``.tsv`` with sample ids as index.
    - ``path_to_ft``: ``.tsv`` or QIIME2 ``.qza`` artifact convertible to DataFrame.
    - Temporal arguments trigger sliding-window snapshot generation and time
      suffixing (``__t0``, ``__t-1``, ...).

    Side Effects
    ------------
    Writes ``train_val.pkl`` and ``test.pkl`` into ``output_path``.
    """
    md, ft = _load_data(path_to_md, path_to_ft)
    # Parse stratify_by string into list if provided (comma-separated)
    strat_cols = None
    if stratify_by:
        strat_cols = [c.strip() for c in stratify_by.split(",") if c.strip()]

    train_val, test = split_train_test(
        md,
        ft,
        group_by_column,
        train_size,
        seed,
        stratify_by=strat_cols,
        time_col=time_col,
        host_col=host_col,
        n_prev=n_prev,
        missing_mode=missing_mode,
    )

    # write to file
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    train_val.to_pickle(os.path.join(output_path, "train_val.pkl"))
    test.to_pickle(os.path.join(output_path, "test.pkl"))

    print(f"Train and test splits were saved in {output_path}.")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    typer.run(cli_split_train_test)
