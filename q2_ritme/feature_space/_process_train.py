from q2_ritme.feature_space.aggregate_features import aggregate_microbial_features
from q2_ritme.feature_space.select_features import select_microbial_features
from q2_ritme.feature_space.transform_features import (
    _find_most_nonzero_feature_idx,
    transform_microbial_features,
)
from q2_ritme.split_train_test import _split_data_stratified


def process_train(config, train_val, target, host_id, tax, seed_data):
    # todo: make feat_prefix an adjustable variable
    feat_prefix = "F"
    microbial_ft_ls = [x for x in train_val if x.startswith(feat_prefix)]
    nonm_ft_ls = [x for x in train_val if x not in microbial_ft_ls]

    # AGGREGATE
    ft_agg = aggregate_microbial_features(
        train_val[microbial_ft_ls],
        config["data_aggregation"],
        tax,
    )
    # SELECT
    ft_selected = select_microbial_features(
        ft_agg,
        config,
        feat_prefix,
    )
    # TRANSFORM
    # during training alr_denom_idx is inferred, during eval the inferred value
    # is used
    config["data_alr_denom_idx"] = (
        _find_most_nonzero_feature_idx(ft_selected)
        if config["data_transform"] == "alr"
        else None
    )

    ft_transformed = transform_microbial_features(
        ft_selected, config["data_transform"], config["data_alr_denom_idx"]
    )
    microbial_ft_ls_transf = ft_transformed.columns

    # rejoin metadata to feature table
    train_val_t = train_val[nonm_ft_ls].join(ft_transformed)

    # SPLIT
    # todo: refine assignment of features to be used for modelling
    train, val = _split_data_stratified(train_val_t, host_id, 0.8, seed_data)
    X_train, y_train = train[microbial_ft_ls_transf], train[target]
    X_val, y_val = val[microbial_ft_ls_transf], val[target]

    return (
        X_train.values,
        y_train.values,
        X_val.values,
        y_val.values,
        microbial_ft_ls_transf,
    )
