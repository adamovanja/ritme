import numpy as np
import pandas as pd

from q2_ritme.feature_space.transform_features import transform_features
from q2_ritme.process_data import split_data_by_host


def _create_matrix_from_tree(tree):
    # Get all leaves and create a mapping from leaf names to indices
    leaves = list(tree.tips())
    leaf_names = [leaf.name for leaf in leaves]
    # map each leaf name to unique index
    leaf_index_map = {name: idx for idx, name in enumerate(leaf_names)}

    # Get the number of leaves and internal nodes
    num_leaves = len(leaf_names)
    # root is not included
    internal_nodes = list(tree.non_tips())

    # Create the identity matrix for the leaves: A1 (num_leaves x num_leaves)
    A1 = np.eye(num_leaves)

    # Create the matrix for the internal nodes: A2 (num_leaves x
    # num_internal_nodes)
    # initialise it with zeros
    A2 = np.zeros((num_leaves, len(internal_nodes)))

    # Populate A2 with 1s for the leaves linked by each internal node
    # iterate over all internal nodes to find descendents of this node and mark
    # them accordingly
    a2_node_names = []
    for j, node in enumerate(internal_nodes):
        # todo: adjust names to consensus taxonomy from descentents
        # for now node names are just increasing integers - since node.name is float
        a2_node_names.append("n" + str(j))
        descendant_leaves = {leaf.name for leaf in node.tips()}
        for leaf_name in leaf_names:
            if leaf_name in descendant_leaves:
                A2[leaf_index_map[leaf_name], j] = 1

    # Concatenate A1 and A2 to create the final matrix A
    A = np.hstack((A1, A2))

    return A, a2_node_names


def _verify_matrix_a(A, feature_columns, tree_phylo):
    # no all 1 in one column
    assert not np.any(np.all(A == 1.0, axis=0))

    # shape should be = feature_count + node_count
    nb_features = len(feature_columns)
    nb_non_leaf_nodes = len(list(tree_phylo.non_tips()))

    assert nb_features + nb_non_leaf_nodes == A.shape[1]


def _preprocess_taxonomy_aggregation(x, A):
    pseudo_count = 0.000001
    # ? what happens if x is relative abundances
    X = np.log(pseudo_count + x)
    nleaves = np.sum(A, axis=0)
    log_geom = X.dot(A) / nleaves

    return log_geom, nleaves


def _transform_features_in_complete_data(config, train_val, target):
    features = [x for x in train_val if x.startswith("F")]
    non_features = [x for x in train_val if x not in features]

    ft_transformed = transform_features(
        train_val[features],
        config["data_transform"],
        config["data_alr_denom_idx"],
    )
    train_val_t = train_val[non_features].join(ft_transformed)

    return train_val_t, ft_transformed.columns


def process_train(config, train_val, target, host_id, seed_data):
    train_val_t, feature_columns = _transform_features_in_complete_data(
        config, train_val, target
    )

    train, val = split_data_by_host(train_val_t, host_id, 0.8, seed_data)
    X_train, y_train = train[feature_columns], train[target]
    X_val, y_val = val[feature_columns], val[target]
    return X_train.values, y_train.values, X_val.values, y_val.values


def process_train_trac(config, train_val, target, host_id, seed_data, tax, tree_phylo):
    train_val_t, feature_columns = _transform_features_in_complete_data(
        config, train_val, target
    )
    X_train_val, y_train_val = train_val_t[feature_columns], train_val_t[target]

    # no need to split train-val for trac since CV is performed within the model

    # derive matrix A
    A, a2_names = _create_matrix_from_tree(tree_phylo)
    _verify_matrix_a(A, feature_columns, tree_phylo)

    # get labels for all dimensions of A
    label = tax["Taxon"].values
    nb_features = len(feature_columns)
    assert len(label) == len(feature_columns)
    label = np.append(label, a2_names)
    assert len(label) == A.shape[1]
    A_df = pd.DataFrame(A, columns=label, index=label[:nb_features])

    # get log_geom
    log_geom_trainval, nleaves = _preprocess_taxonomy_aggregation(
        X_train_val.values, A_df.values
    )

    return log_geom_trainval, y_train_val, nleaves, A_df
