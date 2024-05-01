import string

import numpy as np
import pandas as pd
import scipy

DENSITY = 0.3
SEED_SIM = 12


def simulate_feature_table(
    n_samples: int, n_feat: int, density: float = DENSITY, seed: int = SEED_SIM
) -> pd.DataFrame:
    """
    Creates a random sparse matrix of relative abundances, ensuring at least one
    column has a non-zero feature in all samples.

    Parameters:
    n_samples (int): Number of samples.
    n_feat (int): Number of features.
    density (float): Density of the sparse matrix. Default value is DENSITY.
    seed (int): Seed for the random number generator. Default value is SEED_SIM.

    Returns:
    pd.DataFrame: Relative abundance matrix as a DataFrame.
    """
    # sparse matrix
    rvs = scipy.stats.binom(200, 0.1).rvs
    # rvs = scipy.stats.norm(40, 15).rvs
    # rvs = scipy.stats.expon(3).rvs
    np.random.seed(seed)
    matrix = scipy.sparse.random(
        n_samples, n_feat, density=density, random_state=seed, data_rvs=rvs
    )

    # convert sparse matrix to dense matrix
    dense_matrix = matrix.A

    # ensure at least one column has a non-zero feature in all samples
    non_zero_col = np.random.randint(0, n_feat)
    dense_matrix[:, non_zero_col] = scipy.stats.binom(200, 0.1).rvs(size=n_samples)

    # normalize the dense matrix so that features sum to 1.0 per sample
    row_sums = dense_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = dense_matrix / row_sums

    # transform to df
    feat_ls = [f"F{i}" for i in range(n_feat)]
    sample_ls = [f"SRR{i}" for i in range(n_samples)]
    feat_df = pd.DataFrame(normalized_matrix, columns=feat_ls, index=sample_ls)
    feat_df.index.name = "id"
    return feat_df


def simulate_metadata(
    feat_df: pd.DataFrame, n_hosts: int, target: str, seed: int = SEED_SIM
) -> pd.DataFrame:
    """
    Create simulated metadata table matching provided feature table with given
    number of unique hosts.

    Args:
        feat_df (pd.DataFrame): The feature DataFrame to match.
        n_hosts (int): The number of unique hosts.
        seed (int, optional): The seed for the random number generator. Defaults
        to SEED_SIM.

    Raises:
        ValueError: If the number of hosts is greater than the number of samples.

    Returns:
        pd.DataFrame: The simulated metadata.
    """
    n_samples = feat_df.shape[0]
    if n_hosts > n_samples:
        raise ValueError("More hosts than samples. Reset n_hosts to match the samples")
    # define temporal dimension: age
    np.random.seed(seed)
    # range of target depending on target in config
    # todo: make this more target agnostic
    if target == "age_days":
        age = np.random.uniform(low=0.0, high=2 * 365, size=n_samples).astype(int)
    elif target == "age_months":
        age = np.random.uniform(low=0.0, high=2 * 12, size=n_samples).astype(int)
    else:
        age = np.random.uniform(low=0.0, high=100, size=n_samples).astype(int)

    # set hosts
    host_id_options = list(string.ascii_uppercase)[:n_hosts]
    np.random.seed(seed)
    host_id = np.random.choice(host_id_options, n_samples)

    # combine
    md_df = pd.DataFrame(
        {"host_id": host_id, target: age}, index=feat_df.index.tolist()
    )
    md_df.index.name = "id"

    return md_df


def simulate_data(
    n_samples: int = 100,
    target: str = "age_days",
    n_feat: int = 20,
    n_hosts: int = 4,
    density: float = DENSITY,
    seed: int = SEED_SIM,
) -> (pd.DataFrame, pd.DataFrame):
    """
    Simulates data by calling the functions simulate_feature_table and
    simulate_metadata with the provided parameters.

    Parameters:
    n_samples (int): Number of samples. Default value is 10.
    n_feat (int): Number of features. Default value is 20.
    n_hosts (int): Number of hosts. Default value is 4.
    density (float): Density of the sparse matrix. Default value is DENSITY.
    seed (int): Seed for the random number generator. Default value is SEED_SIM.

    Returns:
    tuple: A tuple containing the feature table and metadata as two DataFrames.
    """
    ft = simulate_feature_table(n_samples, n_feat, density, seed)
    md = simulate_metadata(ft, n_hosts, target, seed)

    return ft, md
