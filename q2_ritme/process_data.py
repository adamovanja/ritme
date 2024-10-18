import pandas as pd
import qiime2 as q2
import skbio
from qiime2.plugins import phylogeny


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
