import os
import warnings

import pandas as pd
import qiime2 as q2
import typer
from sklearn.model_selection import GroupShuffleSplit

from q2_ritme._decorators import helper_function, main_function
from q2_ritme.feature_space.utils import _biom_to_df, _df_to_biom


# ----------------------------------------------------------------------------
@helper_function
def _ft_rename_microbial_features(ft: pd.DataFrame) -> pd.DataFrame:
    """Append "F" to microbial feature names if not present."""
    first_letter = set([i[0] for i in ft.columns.tolist()])
    ft_renamed = ft.copy()
    if first_letter != {"F"}:
        ft_renamed.columns = [f"F{i}" for i in ft.columns.tolist()]
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
    assert ft_rel.sum(axis=1).round(5).eq(1.0).all()

    return ft_rel


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
def _split_data_stratified(
    data: pd.DataFrame,
    stratify_by_column: str,
    train_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Randomly split data into train & test split stratified by column
    "stratify_by_column" (e.g. host_id).
    """
    if len(data[stratify_by_column].unique()) == 1:
        raise ValueError(
            f"Only one unique value of '{stratify_by_column}' available in dataset."
        )

    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    split = gss.split(data, groups=data[stratify_by_column])
    train_idx, test_idx = next(split)

    train, test = data.iloc[train_idx], data.iloc[test_idx]
    print(f"Train: {train.shape}, Test: {test.shape}")

    return train, test


# ----------------------------------------------------------------------------
@main_function
def split_train_test(
    md: pd.DataFrame,
    ft: pd.DataFrame,
    stratify_by_column: str,
    train_size: float = 0.8,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge metadata and feature table and split into train-test sets
    stratified by column "stratify_by_column" (e.g. host_id).

    Args:
    md (pd.DataFrame): Metadata dataframe.
    ft (pd.DataFrame): Feature table dataframe.
    stratify_by_column (str): Column in metadata by which the split should be
    stratified.
    train_size (float, optional): The proportion of the dataset to include in
    the train split. Defaults to 0.8.
    seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: A tuple containing train and test dataframes.
    """
    # preprocess feature table
    ft = _ft_rename_microbial_features(ft)
    ft = _ft_remove_zero_features(ft)

    relative_abundances = ft[ft.columns].sum(axis=1).round(5).eq(1.0).all()
    if not relative_abundances:
        warnings.warn(
            "Provided feature table contains absolute instead of relative abundances. "
            "Hence, converting it to relative abundances..."
        )
        ft = _ft_get_relative_abundance(ft)

    # merge md and ft
    data = md.join(ft, how="inner")

    # split
    train_val, test = _split_data_stratified(
        data, stratify_by_column=stratify_by_column, train_size=train_size, seed=seed
    )

    return train_val, test


@main_function
def cli_split_train_test(
    output_path: str,
    path2md: str,
    path2ft: str,
    stratify_by_column: str,
    train_size: float = 0.8,
    seed: int = 42,
):
    """
    Merge metadata and feature table and split into train-test sets
    stratified by column "stratify_by_column" (e.g. host_id).

    Args:
        output_path (str): Path to save output to.
        path2md (str): Path to metadata file.
        path2ft (str): Path to feature table file.
        stratify_by_column (str): Column in metadata by which the split should be
        stratified.
        train_size (float, optional): The proportion of the dataset to include in
        the train split. Defaults to 0.8.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    Side Effects:
        Writes the train and test splits to "train_val.tsv" and "test.tsv" files
        in the specified output path.
    """
    md, ft = _load_data(path2md, path2ft)

    train_val, test = split_train_test(md, ft, stratify_by_column, train_size, seed)

    # write to file
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    train_val.to_csv(os.path.join(output_path, "train_val.tsv"), sep="\t")
    test.to_csv(os.path.join(output_path + "test.tsv"), sep="\t")
    print(f"Train and test splits were saved in {output_path}.")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    typer.run(cli_split_train_test)
