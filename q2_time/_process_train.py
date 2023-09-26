from q2_time.engineer_features import transform_features
from q2_time.process_data import split_data_by_host


def process_train(config, train_val, target, host_id, seed_data):
    non_features = [target, host_id]
    features = [x for x in train_val if x not in non_features]

    # feature engineering method -> config
    ft_transformed = transform_features(
        train_val[features],
        config["data_transform"],
        config["data_alr_denom_idx"],
    )
    train_val_t = train_val[non_features].join(ft_transformed)

    # train & val split - for training purposes
    train, val = split_data_by_host(train_val_t, host_id, 0.8, seed_data)
    X_train, y_train = train[ft_transformed.columns], train[target]
    X_val, y_val = val[ft_transformed.columns], val[target]
    return X_train.values, y_train.values, X_val.values, y_val.values
