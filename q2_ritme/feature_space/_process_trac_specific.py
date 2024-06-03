import os

import numpy as np
import pandas as pd


def _verify_matrix_a(A, feature_columns, tree_phylo):
    # no all 1 in one column
    assert not np.any(np.all(A == 1.0, axis=0))

    # shape should be = feature_count + node_count
    nb_features = len(feature_columns)
    nb_non_leaf_nodes = len(list(tree_phylo.non_tips()))

    assert nb_features + nb_non_leaf_nodes == A.shape[1]


def _get_leaves_and_index_map(tree):
    leaves = list(tree.tips())
    leaf_names = [leaf.name for leaf in leaves]
    # map each leaf name to unique index
    leaf_index_map = {name: idx for idx, name in enumerate(leaf_names)}
    return leaves, leaf_index_map


def _get_internal_nodes(tree):
    # root is not included
    return list(tree.non_tips())


def _create_identity_matrix_for_leaves(num_leaves, tax, leaves):
    A1 = np.eye(num_leaves)
    # taxonomic name should include OTU name
    tax_e = tax.copy()
    tax_e["tax_ft"] = tax_e["Taxon"] + "; otu__" + tax_e.index
    # tax matched by leave name - order does not matter
    a1_node_names = tax_e.loc[[leaf.name for leaf in leaves], "tax_ft"].tolist()
    return A1, a1_node_names


def _populate_A2_for_node(A2, node, leaf_index_map, j):
    node_leaf_names = []
    # flag leaves that match to a node
    descendant_leaves = {leaf.name for leaf in node.tips()}
    for leaf_name in leaf_index_map:
        if leaf_name in descendant_leaves:
            node_leaf_names.append(leaf_name)
            A2[leaf_index_map[leaf_name], j] = 1
    return A2, node_leaf_names


def _create_consensus_taxonomy(node_leaf_names, tax, a2_node_names, j):
    tax_e = tax.copy()
    tax_e["tax_ft"] = tax_e["Taxon"] + "; otu__" + tax_e.index
    node_mapped_taxon = tax_e.loc[node_leaf_names, "tax_ft"].tolist()
    str_consensus_taxon = os.path.commonprefix(node_mapped_taxon)
    # get name before last ";"
    node_consensus_taxon = str_consensus_taxon.rpartition(";")[0]
    # if consensus name already exists, add index to make it unique
    if node_consensus_taxon in a2_node_names:
        node_consensus_taxon = node_consensus_taxon + "; n__" + str(j)
    return node_consensus_taxon


def _create_matrix_for_internal_nodes(num_leaves, internal_nodes, leaf_index_map, tax):
    # initialise it with zeros
    A2 = np.zeros((num_leaves, len(internal_nodes)))
    a2_node_names = []
    # Populate A2 with 1s for the leaves linked by each internal node # iterate
    # over all internal nodes to find descendents of this node and mark them
    # accordingly
    for j, node in enumerate(internal_nodes):
        A2, node_leaf_names = _populate_A2_for_node(A2, node, leaf_index_map, j)
        # create consensus taxonomy from all leaf_names- since node.name is just float
        node_consensus_taxon = _create_consensus_taxonomy(
            node_leaf_names, tax, a2_node_names, j
        )
        a2_node_names.append(node_consensus_taxon)
    return A2, a2_node_names


def create_matrix_from_tree(tree, tax) -> pd.DataFrame:
    # Get all leaves and create a mapping from leaf names to indices
    leaves, leaf_index_map = _get_leaves_and_index_map(tree)
    num_leaves = len(leaves)

    # Get all internal nodes
    internal_nodes = _get_internal_nodes(tree)

    # Create the identity matrix for the leaves: A1 (num_leaves x num_leaves)
    A1, a1_node_names = _create_identity_matrix_for_leaves(num_leaves, tax, leaves)

    # Create the matrix for the internal nodes: A2 (num_leaves x num_internal_nodes)
    A2, a2_node_names = _create_matrix_for_internal_nodes(
        num_leaves, internal_nodes, leaf_index_map, tax
    )

    # Concatenate A1 and A2 to create the final matrix A
    A = np.hstack((A1, A2))
    df_a = pd.DataFrame(
        A, columns=a1_node_names + a2_node_names, index=[leaf.name for leaf in leaves]
    )
    _verify_matrix_a(df_a.values, tax.index.tolist(), tree)

    return df_a


def _preprocess_taxonomy_aggregation(x, A):
    pseudo_count = 0.000001

    X = np.log(pseudo_count + x)
    nleaves = np.sum(A, axis=0)
    # safekeeping: dot-product would not work with wrong dimensions
    # X: n_samples, n_features,  A: n_features, (n_features+n_nodes)
    log_geom = X.dot(A) / nleaves

    return log_geom, nleaves
