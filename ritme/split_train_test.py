import os
import warnings
from typing import List, Sequence, Tuple, Union

import pandas as pd
import qiime2 as q2
import typer
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from ritme._decorators import helper_function, main_function
from ritme.feature_space.utils import _biom_to_df, _df_to_biom


# ----------------------------------------------------------------------------
@helper_function
def _ft_rename_microbial_features(
    ft: pd.DataFrame, feature_prefix: str
) -> pd.DataFrame:
    """Append feature_prefix to all microbial feature names"""
    ft_renamed = ft.copy()
    ft_renamed.columns = [f"{feature_prefix}{i}" for i in ft.columns.tolist()]
    return ft_renamed


@helper_function
def _ft_remove_zero_features(ft: pd.DataFrame) -> pd.DataFrame:
    """Remove features that are all zero."""
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
    """
    Transform feature table from absolute to relative abundance.
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
    """Apply standard preprocessing to a single snapshot feature table."""
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
    """
    Merge multiple metadata and feature tables denoting time snapshots.

    Assumptions:
    - Order corresponds to time labels t0, t-1, t-2, ...

    Returns
    -------
    md_prepared: pd.DataFrame
        Metadata suffixed by their time labels, e.g., 'age__t0', 'age__t-1'.
    merged_ft: pd.DataFrame
        Feature table with microbial columns suffixed by their time labels.
    """
    if len(md_list) != len(ft_list):
        raise ValueError(
            "Number of metadata files must match number of feature tables."
        )
    _verify_identical_indices(md_list, kind="metadata")
    _verify_identical_indices(ft_list, kind="feature table")

    # feature column sets must match across snapshots
    base_cols = set(ft_list[0].columns)
    for i, ft in enumerate(ft_list[1:], start=1):
        if set(ft.columns) != base_cols:
            raise ValueError(
                "Feature column sets must match across all snapshots; mismatch "
                f"found at position {i}."
            )

    time_labels = _generate_time_labels(len(ft_list))

    # Prepare and suffix features for each snapshot
    ft_prepared: List[pd.DataFrame] = []
    # Prepare and suffix metadata for each snapshot
    md_prepared: List[pd.DataFrame] = []
    for ft, tlabel in zip(ft_list, time_labels):
        ft_p = _prepare_single_feature_table(ft)
        ft_s = _append_time_suffix_to_features(ft_p, tlabel)
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
    """
    Load data from provided paths
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
def _load_data_multi(
    paths_md: Sequence[str], paths_ft: Sequence[str]
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """Load multiple metadata and feature table files."""
    if len(paths_md) != len(paths_ft):
        raise ValueError("Number of metadata and feature table paths must match.")
    md_list: List[pd.DataFrame] = []
    ft_list: List[pd.DataFrame] = []
    for pmd, pft in zip(paths_md, paths_ft):
        md, ft = _load_data(pmd, pft)
        md_list.append(md)
        ft_list.append(ft)
    return md_list, ft_list


@helper_function
def _split_data_grouped(
    data: pd.DataFrame,
    group_by_column: str,
    train_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Randomly splits the provided data into train and test sets, grouping rows by
    the specified `group_by_column`. Grouping ensures that rows with the same
    group value are not spread across multiple subsets, preventing data leakage.
    If no grouping is provided, a standard random split is performed.
    """
    if group_by_column is None:
        train, test = train_test_split(data, train_size=train_size, random_state=seed)
    else:
        if len(data[group_by_column].unique()) == 1:
            raise ValueError(
                f"Only one unique value of '{group_by_column}' available in dataset."
            )

        gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
        split = gss.split(data, groups=data[group_by_column])
        train_idx, test_idx = next(split)

        train, test = data.iloc[train_idx], data.iloc[test_idx]

    print(f"Train: {train.shape}, Test: {test.shape}")
    return train, test


# ----------------------------------------------------------------------------
@main_function
def split_train_test(
    md: Union[pd.DataFrame, Sequence[pd.DataFrame]],
    ft: Union[pd.DataFrame, Sequence[pd.DataFrame]],
    group_by_column: str = None,
    train_size: float = 0.8,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge metadata and feature table and split into train-test sets. Split can
    be performed with grouping by column "group_by_column" (e.g. host_id), this
    ensures that rows with the same group value are not spread across multiple
    subsets, preventing data leakage. All feature columns in ft are prefixed
    with an "F".

    Args:
    md (pd.DataFrame): Metadata dataframe.
    ft (pd.DataFrame): Feature table dataframe.
    group_by_column (str, optional): Column in metadata by which the split
    should be grouped. Defaults to None.
    train_size (float, optional): The proportion of the dataset to include
    in the train split. Defaults to 0.8.
    seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: A tuple containing train and test dataframes.
    """
    # Support single-snapshot and multi-snapshot inputs
    if isinstance(md, pd.DataFrame) and isinstance(ft, pd.DataFrame):
        # For consistency, suffix both metadata and microbial feature columns with __t0
        time_label = "t0"
        md_base = md.copy()
        md_base.columns = [f"{c}__{time_label}" for c in md_base.columns]
        ft_single = _prepare_single_feature_table(ft)
        ft_merged = _append_time_suffix_to_features(ft_single, time_label)
    else:
        # Expect sequences for both
        if not isinstance(md, Sequence) or not isinstance(ft, Sequence):
            raise ValueError(
                "md and ft must both be DataFrames or both be sequences of DataFrames."
            )
        md_base, ft_merged = _merge_time_snapshots(md, ft)

    # merge md and feature table (inner join on sample ids)
    data = md_base.join(ft_merged, how="inner")

    # Resolve group_by_column to suffixed name if needed (host_id -> host_id__t0)
    group_col = group_by_column
    if group_by_column is not None and group_by_column not in data.columns:
        alt = f"{group_by_column}__t0"
        if alt in data.columns:
            group_col = alt
        else:
            raise ValueError(f"Group by column '{group_by_column}' not found in data.")
    # split
    train_val, test = _split_data_grouped(
        data, group_by_column=group_col, train_size=train_size, seed=seed
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
):
    """
    Merge metadata and feature table and split into train-test sets. Split can
    be performed with grouping by column "group_by_column" (e.g. host_id), this
    ensures that rows with the same group value are not spread across multiple
    subsets, preventing data leakage. All feature columns in path_to_ft are
    prefixed with an "F".

    Args:
        output_path (str): Path to save output to.
        path_to_md (str): Path to metadata file.
        path_to_ft (str): Path to feature table file.
        group_by_column (str, optional): Column in metadata by which the split
        should be grouped. Defaults to None.
        train_size (float, optional): The proportion of the dataset to include
        in the train split. Defaults to 0.8.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    Side Effects:
        Writes the train and test splits to "train_val.pkl" and "test.pkl" files
        in the specified output path.
    """
    # Support comma-separated lists of files for multi-snapshot input
    if "," in path_to_md or "," in path_to_ft:
        paths_md = [p.strip() for p in path_to_md.split(",")]
        paths_ft = [p.strip() for p in path_to_ft.split(",")]
        md_list, ft_list = _load_data_multi(paths_md, paths_ft)
        train_val, test = split_train_test(
            md_list, ft_list, group_by_column, train_size, seed
        )
    else:
        md, ft = _load_data(path_to_md, path_to_ft)
        train_val, test = split_train_test(md, ft, group_by_column, train_size, seed)

    # write to file
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    train_val.to_pickle(os.path.join(output_path, "train_val.pkl"))
    test.to_pickle(os.path.join(output_path, "test.pkl"))

    print(f"Train and test splits were saved in {output_path}.")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    typer.run(cli_split_train_test)
