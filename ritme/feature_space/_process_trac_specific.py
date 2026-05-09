import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ritme.feature_space.transform_features import PSEUDOCOUNT


def _verify_matrix_a(A, feature_columns, tree_phylo):
    # no all 1 in one column
    if sp.issparse(A):
        col_sums = np.asarray(A.sum(axis=0)).ravel()
        # A is binary at construction time, so an all-ones column has
        # sum equal to the number of rows.
        assert not np.any(col_sums == A.shape[0])
    else:
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
    A1 = sp.identity(num_leaves, dtype=float, format="csc")
    # taxonomic name should include OTU name
    tax_e = tax.copy()
    tax_e["tax_ft"] = tax_e["Taxon"] + "; otu__" + tax_e.index
    # tax matched by leave name - order does not matter
    a1_node_names = tax_e.loc[[leaf.name for leaf in leaves], "tax_ft"].tolist()
    return A1, a1_node_names


def _descendant_leaf_indices(node, leaf_index_map):
    descendant_leaves = {leaf.name for leaf in node.tips()}
    leaf_indices = []
    leaf_names = []
    for leaf_name, idx in leaf_index_map.items():
        if leaf_name in descendant_leaves:
            leaf_indices.append(idx)
            leaf_names.append(leaf_name)
    return leaf_indices, leaf_names


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
    # Build A2 directly from the leaf-membership pattern of each internal node.
    # Collinear columns (identical leaf sets) are dropped on the fly via a
    # signature lookup - no dense num_leaves x num_internal_nodes intermediate
    # and no transpose-based duplicate scan, both of which scale as O(F^2)
    # for high-dimensional feature spaces.
    rows = []
    cols = []
    a2_node_names = []
    seen_signatures = {}
    next_col = 0

    for j, node in enumerate(internal_nodes):
        leaf_indices, node_leaf_names = _descendant_leaf_indices(node, leaf_index_map)
        signature = tuple(sorted(leaf_indices))

        if signature in seen_signatures:
            # collinear with an earlier kept column - drop (keep="first")
            continue

        node_consensus_taxon = _create_consensus_taxonomy(
            node_leaf_names, tax, a2_node_names, j
        )
        seen_signatures[signature] = next_col
        rows.extend(leaf_indices)
        cols.extend([next_col] * len(leaf_indices))
        a2_node_names.append(node_consensus_taxon)
        next_col += 1

    data = np.ones(len(rows), dtype=float)
    A2 = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(num_leaves, next_col),
    ).tocsc()
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

    # Concatenate A1 and A2 in sparse form to avoid a dense O(F^2) hstack copy.
    A_sparse = sp.hstack([A1, A2], format="csc")
    _verify_matrix_a(A_sparse, tax.index.tolist(), tree)

    df_a_sparse = pd.DataFrame.sparse.from_spmatrix(
        A_sparse,
        columns=a1_node_names + a2_node_names,
        index=[leaf.name for leaf in leaves],
    )
    return df_a_sparse


def _as_aggregation_matrix(A):
    """Return a numpy or scipy.sparse 2D matrix usable for taxonomy aggregation.

    Pandas DataFrames whose columns all have ``SparseDtype`` are converted to a
    scipy CSC matrix without densifying - the path that keeps RAM low for
    high-dimensional feature spaces.
    """
    if sp.issparse(A):
        return A
    if isinstance(A, pd.DataFrame):
        all_sparse = len(A.columns) > 0 and all(
            isinstance(dt, pd.SparseDtype) for dt in A.dtypes
        )
        if all_sparse:
            return A.sparse.to_coo().tocsc()
        return A.values
    return A


def _preprocess_taxonomy_aggregation(x, A):
    A_eff = _as_aggregation_matrix(A)
    X = np.log(PSEUDOCOUNT + x)
    # safekeeping: dot-product would not work with wrong dimensions
    # X: n_samples, n_features,  A: n_features, (n_features+n_nodes)
    if sp.issparse(A_eff):
        nleaves = np.asarray(A_eff.sum(axis=0)).ravel()
        log_geom = X @ A_eff
        # Classo expects a dense design matrix
        if sp.issparse(log_geom):
            log_geom = log_geom.toarray()
        log_geom = log_geom / nleaves
    else:
        nleaves = np.sum(A_eff, axis=0)
        log_geom = X.dot(A_eff) / nleaves

    return log_geom, nleaves
