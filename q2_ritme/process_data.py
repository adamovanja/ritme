import pandas as pd
import qiime2 as q2
import skbio
from qiime2.plugins import phylogeny
from sklearn.model_selection import GroupShuffleSplit

from q2_ritme.feature_space.utils import _biom_to_df, _df_to_biom
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

    # biom.norm faster than skbio.stats.composition.closure
    ft_biom = _df_to_biom(ft_sel)

    # Normalize the feature table using biom.Table.norm()
    ft_rel_biom = ft_biom.norm(axis="sample", inplace=False)

    # Convert the normalized biom.Table back to a pandas DataFrame
    ft_rel = _biom_to_df(ft_rel_biom)

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


def load_tax_phylo(
    path2tax: str, path2phylo: str, ft: pd.DataFrame
) -> (pd.DataFrame, skbio.TreeNode):
    """
    Load taxonomy and phylogeny data.
    """
    # todo: add option for simulated data
    if path2tax and path2phylo:
        # taxonomy
        art_taxonomy = q2.Artifact.load(path2tax)
        df_tax = art_taxonomy.view(pd.DataFrame)
        # rename taxonomy to match new "F" feature names
        df_tax.index = df_tax.index.map(lambda x: "F" + str(x))

        # Filter the taxonomy based on the feature table
        df_tax_f = df_tax[df_tax.index.isin(ft.columns.tolist())]

        if df_tax_f.shape[0] == 0:
            raise ValueError("Taxonomy data does not match with feature table.")

        # phylogeny
        art_phylo = q2.Artifact.load(path2phylo)
        # filter tree by feature table: this prunes a phylogenetic tree to match
        # the input ids
        # Remove the first letter of each column name: "F" to match phylotree
        ft_i = ft.copy()
        ft_i.columns = [col[1:] for col in ft_i.columns]
        art_ft_i = q2.Artifact.import_data("FeatureTable[RelativeFrequency]", ft_i)

        (art_phylo_f,) = phylogeny.actions.filter_tree(tree=art_phylo, table=art_ft_i)
        tree_phylo_f = art_phylo_f.view(skbio.TreeNode)

        # add prefix "F" to leaf names in tree to remain consistent with ft
        for node in tree_phylo_f.tips():
            node.name = "F" + node.name

        # ensure that # leaves in tree == feature table dimension
        num_leaves = tree_phylo_f.count(tips=True)
        assert num_leaves == ft.shape[1]
    else:
        # load empty variables
        df_tax_f = pd.DataFrame()
        tree_phylo_f = skbio.TreeNode()

    return df_tax_f, tree_phylo_f


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
    path2tax: str,
    path2phylo: str,
    host_id: str,
    target: str,
    train_size: float,
    seed: int,
    filter_md_cols: list = None,
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, skbio.TreeNode):
    """
    Load, merge and sort data, then split into train-test sets by host_id.

    Args:
        path2md (str, optional): Path to metadata file. If None, simulated data
        is used.
        path2ft (str, optional): Path to features file. If None, simulated data
        is used.
        path2tax (str, optional): Path to taxonomy file. If None, model options
        requiring taxonomy can't be run.
        path2phylo (str, optional): Path to phylogeny file. If None, model
        options requiring taxonomy can't be run.
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
        : Train and test dataframes as well as matching taxonomy and phylogeny.
    """
    ft, md = load_data(path2md, path2ft, target)

    # tax: n_features x ("Taxon", "Confidence")
    # tree_phylo: n_features leaf nodes
    tax, tree_phylo = load_tax_phylo(path2tax, path2phylo, ft)
    data = filter_merge_n_sort(md, ft, host_id, target, filter_md_cols)

    # todo: add split also by study_id
    train_val, test = split_data_by_host(data, host_id, train_size, seed)

    return train_val, test, tax, tree_phylo
