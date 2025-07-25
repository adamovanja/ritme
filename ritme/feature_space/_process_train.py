from ritme.feature_space.aggregate_features import aggregate_microbial_features
from ritme.feature_space.enrich_features import enrich_features
from ritme.feature_space.select_features import select_microbial_features
from ritme.feature_space.transform_features import (
    _find_most_nonzero_feature_idx,
    transform_microbial_features,
)
from ritme.split_train_test import _split_data_grouped


def process_train(config, train_val, target, host_id, tax, seed_data):
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
    microbial_ft_ls_transf = ft_transformed.columns.tolist()

    # rejoin metadata to feature table
    train_val_t = train_val[nonm_ft_ls].join(ft_transformed)

    # ENRICH feature table
    other_ft_ls, train_val_te = enrich_features(
        train_val, microbial_ft_ls, train_val_t, config
    )
    ft_ls_used = microbial_ft_ls_transf + other_ft_ls

    # SPLIT
    train, val = _split_data_grouped(train_val_te, host_id, 0.8, seed_data)
    X_train, y_train = train[ft_ls_used], train[target].astype(float)
    X_val, y_val = val[ft_ls_used], val[target].astype(float)

    return (
        X_train.values,
        y_train.values,
        X_val.values,
        y_val.values,
        ft_ls_used,
    )
