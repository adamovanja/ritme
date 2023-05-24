import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# todo: adjust to json file to be read in from user
from q2_time.config import HOST_ID, SEED_DATA, TARGET, TRAIN_SIZE
from q2_time.simulate_data import simulate_data


def split_data_by_host(data, host_id=HOST_ID, train_size=TRAIN_SIZE, seed=SEED_DATA):
    """Randomly split dataset into train & test split based on host_id"""
    if len(data[host_id].unique()) == 1:
        raise ValueError("Only one unique host available in dataset.")

    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    split = gss.split(data, groups=data[host_id])
    train_idx, test_idx = next(split)

    train, test = data.iloc[train_idx], data.iloc[test_idx]
    print(f"Train: {train.shape}, Test: {test.shape}")

    return train, test


def merge_n_sort(md, ft, host_id=HOST_ID, target=TARGET):
    data = md.join(ft, how="left")
    data.sort_values([host_id, target], inplace=True)
    return data


def load_data(path2md, path2ft):
    if path2md and path2ft:
        md = pd.read_csv(path2md, sep="\t", index_col=0)
        ft = pd.read_csv(path2ft, sep="\t", index_col=0)
    else:
        ft, md = simulate_data(100)

    # extract all microbiome features
    micro_fts = ft.columns.tolist()

    return ft, md, micro_fts


def load_n_split_data(path2md=None, path2ft=None):
    ft, md, micro_fts = load_data(path2md, path2ft)
    data = merge_n_sort(md, ft)
    train_val, test = split_data_by_host(data)

    return train_val, test, micro_fts
