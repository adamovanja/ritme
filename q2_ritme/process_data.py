import biom
import pandas as pd
import qiime2 as q2
from sklearn.model_selection import GroupShuffleSplit

# todo: adjust to json file to be read in from user
from q2_ritme.simulate_data import simulate_data


def get_relative_abundance(
    ft: pd.DataFrame, feature_prefix: str = "", no_features: list = []
) -> pd.DataFrame:
    """
    Transform feature table from absolute to relative abundance. Only columns in
    feature_prefix are transformed. If feature_prefix is not set, then all
    features except no_features are transformed.
    """
    if feature_prefix != "":
        ft_cols = [x for x in ft.columns if x.startswith(feature_prefix)]
    elif len(no_features) > 0:
        ft_cols = [x for x in ft.columns if x not in no_features]
    else:
        raise ValueError("Either feature_prefix or no_features must be set")
    ft_sel = ft[ft_cols]
    # Convert the pandas DataFrame to a biom.Table object
    ft_biom = biom.Table(
        ft_sel.T.values, observation_ids=ft_sel.columns, sample_ids=ft_sel.index
    )

    # Normalize the feature table using biom.Table.norm()
    ft_rel_biom = ft_biom.norm(axis="sample", inplace=False)

    # Convert the normalized biom.Table back to a pandas DataFrame
    ft_rel = pd.DataFrame(
        ft_rel_biom.matrix_data.toarray().T,
        index=ft_rel_biom.ids(axis="sample"),
        columns=ft_rel_biom.ids(axis="observation"),
    )

    # round needed as certain 1.0 are represented in different digits 2e-16
    assert ft_rel[ft_cols].sum(axis=1).round(5).eq(1.0).all()

    return ft_rel


def load_data(
    path2md: str = None, path2ft: str = None, target: str = None
) -> (pd.DataFrame, pd.DataFrame):
    """
    Load data from the provided paths or generate simulated data.

    Args:
        path2md (str): Path to metadata file. If None, simulated data is used.
        path2ft (str): Path to features file - must be relative abundances. If
        None, simulated data is used.

    Returns:
        tuple: A tuple containing two pandas DataFrames, first for features,
        second for metadata.
    """
    if path2md and path2ft:
        md = pd.read_csv(path2md, sep="\t", index_col=0)

        if path2ft.endswith(".tsv"):
            ft = pd.read_csv(path2ft, sep="\t", index_col=0)
        elif path2ft.endswith(".qza"):
            ft = q2.Artifact.load(path2ft).view(pd.DataFrame)

        # flag microbial features with prefix "F"
        first_letter = set([i[0] for i in ft.columns.tolist()])
        if first_letter != {"F"}:
            ft.columns = [f"F{i}" for i in ft.columns.tolist()]
    else:
        ft, md = simulate_data(1000, target)

    # assert that loaded ft has relative abundances
    relative_abundances = ft[ft.columns].sum(axis=1).round(5).eq(1.0).all()
    if not relative_abundances:
        print(
            "Feature columns do not sum to 1.0 for all samples - so they are being "
            "transformed."
        )
        ft = get_relative_abundance(ft, "F")

    return ft, md


def filter_merge_n_sort(
    md: pd.DataFrame,
    ft: pd.DataFrame,
    host_id: str,
    target: str,
    filter_md_cols: list = None,
) -> pd.DataFrame:
    """
    Merge filtered metadata and features and sort by host_id and target.

    Args:
        md (pd.DataFrame): Dataframe containing metadata.
        ft (pd.DataFrame): Dataframe containing features.
        host_id (str): ID name of the host.
        target (str): Name of target variable.
        filter_md_cols (list): List of metadata fields to include.

    Returns:
        pd.DataFrame: Merged and sorted data.
    """
    # filter on metadata fields to include
    if filter_md_cols:
        md = md[filter_md_cols].copy()
    data = md.join(ft, how="inner")
    # print(f"Data shape: {data.shape}")
    data.sort_values([host_id, target], inplace=True)
    return data


def split_data_by_host(
    data: pd.DataFrame,
    host_id: str,
    train_size: float,
    seed: int,
) -> (pd.DataFrame, pd.DataFrame):
    """
    Randomly split dataset into train & test split based on host_id.

    Args:
        data (pd.DataFrame): Merged dataset to be split.
        host_id (str): ID of the host.
        train_size (float): The proportion of the dataset to include in the train split.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing train and test dataframes.

    Raises:
        ValueError: If only one unique host is available in the dataset.
    """
    if len(data[host_id].unique()) == 1:
        raise ValueError("Only one unique host available in dataset.")

    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    split = gss.split(data, groups=data[host_id])
    train_idx, test_idx = next(split)

    train, test = data.iloc[train_idx], data.iloc[test_idx]
    print(f"Train: {train.shape}, Test: {test.shape}")

    return train, test


def load_n_split_data(
    path2md: str,
    path2ft: str,
    host_id: str,
    target: str,
    train_size: float,
    seed: int,
    filter_md_cols: list = None,
) -> (pd.DataFrame, pd.DataFrame):
    """
    Load, merge and sort data, then split into train-test sets by host_id.

    Args:
        path2md (str, optional): Path to metadata file. If None, simulated data
        is used.
        path2ft (str, optional): Path to features file. If None, simulated data
        is used.
        host_id (str, optional): ID of the host. Default is HOST_ID from config.
        target (str, optional): Name of target variable. Default is TARGET from
        config.
        filter_md_cols (list, optional): List of metadata fields to include. If None,
        all fields are included.
        train_size (float, optional): The proportion of the dataset to include
        in the train split. Default is TRAIN_SIZE from config.
        seed (int, optional): Random seed for reproducibility. Default is
        SEED_DATA from config.

    Returns:
        tuple: A tuple containing train and test dataframes.
    """
    ft, md = load_data(path2md, path2ft, target)

    data = filter_merge_n_sort(md, ft, host_id, target, filter_md_cols)

    # todo: add split also by study_id
    train_val, test = split_data_by_host(data, host_id, train_size, seed)

    return train_val, test
