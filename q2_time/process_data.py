import pandas as pd
import qiime2 as q2
from sklearn.model_selection import GroupShuffleSplit

# todo: adjust to json file to be read in from user
from q2_time.config import HOST_ID, SEED_DATA, TARGET, TRAIN_SIZE
from q2_time.simulate_data import simulate_data


def load_data(path2md: str, path2ft: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Load data from the provided paths or generate simulated data.

    Args:
        path2md (str): Path to metadata file. If None, simulated data is used.
        path2ft (str): Path to features file. If None, simulated data is used.

    Returns:
        tuple: A tuple containing two pandas DataFrames, first for features,
        second for metadata.
    """
    if path2md and path2ft:
        md = pd.read_csv(path2md, sep="\t", index_col=0)

        # todo: add test for distinction between qza and tsv
        if path2ft.endswith(".tsv"):
            ft = pd.read_csv(path2ft, sep="\t", index_col=0)
        elif path2ft.endswith(".qza"):
            ft = q2.Artifact.load(path2ft).view(pd.DataFrame)

        # flag microbial features with prefix "F"
        first_letter = set([i[0] for i in ft.columns.tolist()])
        if first_letter != {"F"}:
            ft.columns = [f"F{i}" for i in ft.columns.tolist()]
    else:
        ft, md = simulate_data(100)

    return ft, md


def filter_merge_n_sort(
    md: pd.DataFrame,
    ft: pd.DataFrame,
    host_id: str = HOST_ID,
    target: str = TARGET,
    filter_md: list = None,
) -> pd.DataFrame:
    """
    Merge filtered metadata and features and sort by host_id and target.

    Args:
        md (pd.DataFrame): Dataframe containing metadata.
        ft (pd.DataFrame): Dataframe containing features.
        host_id (str): ID of the host machine, default is HOST_ID from config.
        target (str): Target variable, default is TARGET from config.
        filter_md (list): List of metadata fields to include.

    Returns:
        pd.DataFrame: Merged and sorted data.
    """
    # filter on metadata fields to include
    if filter_md:
        md = md[filter_md].copy()
    data = md.join(ft, how="left")
    data.sort_values([host_id, target], inplace=True)
    return data


def split_data_by_host(
    data: pd.DataFrame,
    host_id: str = HOST_ID,
    train_size: float = TRAIN_SIZE,
    seed: int = SEED_DATA,
) -> (pd.DataFrame, pd.DataFrame):
    """
    Randomly split dataset into train & test split based on host_id.

    Args:
        data (pd.DataFrame): Merged dataset to be split.
        host_id (str): ID of the host, default is HOST_ID from config.
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
    path2md: str = None, path2ft: str = None
) -> (pd.DataFrame, pd.DataFrame):
    """
    Load, merge and sort data, then split intro train-test sets by host_id.

    Args:
        path2md (str, optional): Path to metadata file. If None, simulated data is used.
        path2ft (str, optional): Path to features file. If None, simulated data is used.

    Returns:
        tuple: A tuple containing train and test dataframes.
    """
    ft, md = load_data(path2md, path2ft)

    data = filter_merge_n_sort(md, ft)

    # todo: add split also by study_id
    train_val, test = split_data_by_host(data)

    return train_val, test
