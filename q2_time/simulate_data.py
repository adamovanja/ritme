import string

import numpy as np
import pandas as pd
import scipy

DENSITY = 0.3
SEED_SIM = 12


def simulate_feature_table(n_samples, n_feat, density=DENSITY, seed=SEED_SIM):
    """Creates random sparse matrix of integers"""
    # sparse matrix
    # todo: improve simulation here - semesterproject
    rvs = scipy.stats.binom(200, 0.1).rvs
    # rvs = scipy.stats.norm(40, 15).rvs
    # rvs = scipy.stats.expon(3).rvs
    np.random.seed(seed)
    matrix = scipy.sparse.random(
        n_samples, n_feat, density=density, random_state=seed, data_rvs=rvs
    )

    # transform to df
    feat_ls = [f"F{i}" for i in range(n_feat)]
    sample_ls = [f"SRR{i}" for i in range(n_samples)]
    feat_df = pd.DataFrame(matrix.A, columns=feat_ls, index=sample_ls)
    feat_df.index.name = "id"
    return feat_df


def simulate_metadata(feat_df, n_hosts, seed=SEED_SIM):
    """Create simulated metadata table matching provided feature table
    with n_hosts number of unique hosts."""
    n_samples = feat_df.shape[0]
    if n_hosts > n_samples:
        raise ValueError("More hosts than samples. Reset n_hosts to match the samples")
    # define temporal dimension: age
    np.random.seed(seed)
    age = np.random.uniform(low=0.0, high=2 * 365, size=n_samples).astype(int)

    # set hosts
    host_id_options = list(string.ascii_uppercase)[:n_hosts]
    np.random.seed(seed)
    host_id = np.random.choice(host_id_options, n_samples)

    # combine
    md_df = pd.DataFrame(
        {"host_id": host_id, "age_days": age}, index=feat_df.index.tolist()
    )
    md_df.index.name = "id"

    return md_df


def simulate_data(
    n_samples: int = 10,
    n_feat: int = 20,
    n_hosts: int = 4,
    density: float = DENSITY,
    seed: int = SEED_SIM,
):
    ft = simulate_feature_table(n_samples, n_feat, density, seed)
    md = simulate_metadata(ft, n_hosts, seed)

    # # saving created ft and md files
    # str_dens = str(density).replace(".", "")
    # config_suffix = (
    #     f"{n_samples}x{n_feat}_H{n_hosts}_D{str_dens}_S{seed}"
    # )
    # path_md = os.path.join(
    #     outdir,
    #     f"md_sim_{config_suffix}.tsv",
    # )
    # md.to_csv(path_md, sep="\t")

    # # todo: adjust to save as Q2 FeatureTable[Frequency] artifact
    # path_ft = os.path.join(
    #     outdir,
    #     f"ft_sim_{config_suffix}.tsv",
    # )
    # ft.to_csv(path_ft, sep="\t")

    # print(f"Saved simulated datasets in: {outdir}")

    return ft, md
