from q2_ritme.feature_space.transform_features import transform_features
from q2_ritme.process_data import split_data_by_host


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
    return X_train.values, y_train.values, X_val.values, y_val.values, feature_columns
