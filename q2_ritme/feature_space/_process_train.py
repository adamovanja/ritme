from q2_ritme.feature_space.transform_features import transform_microbial_features
from q2_ritme.process_data import split_data_by_host


def process_train(config, train_val, target, host_id, seed_data):
    microbial_ft_ls = [x for x in train_val if x.startswith("F")]
    nonm_ft_ls = [x for x in train_val if x not in microbial_ft_ls]

    # AGGREGATE
    # todo: if tax empty - no tax_aggregation

    # TRANSFORM
    ft_transformed = transform_microbial_features(
        train_val[microbial_ft_ls],
        config["data_transform"],
        config["data_alr_denom_idx"],
    )
    microbial_ft_ls_transf = ft_transformed.columns
    train_val_t = train_val[nonm_ft_ls].join(ft_transformed)

    # SPLIT
    # todo: refine assignment of features to be used for modelling
    train, val = split_data_by_host(train_val_t, host_id, 0.8, seed_data)
    X_train, y_train = train[microbial_ft_ls_transf], train[target]
    X_val, y_val = val[microbial_ft_ls_transf], val[target]

    return (
        X_train.values,
        y_train.values,
        X_val.values,
        y_val.values,
        microbial_ft_ls_transf,
    )
