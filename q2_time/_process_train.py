from q2_time._o_model import split_data_by_host
from q2_time.engineer_features import transform_features


def process_train(config, train_val, target, features, host_id, seed_data):
    non_features = [target, host_id]

    # feature engineering method -> config
    if config["data_transform"] is None:
        ft_transformed = train_val[features].copy()
    else:
        ft_transformed = transform_features(
            train_val[features],
            config["data_transform"],
            config["alr_denom_idx"],
        )
    train_val_t = train_val[non_features].join(ft_transformed)

    # train & val split - for training purposes
    train, val = split_data_by_host(train_val_t, host_id, 0.8, seed_data)
    X_train, y_train = train[ft_transformed.columns], train[target]
    X_val, y_val = val[ft_transformed.columns], val[target]
    return X_train.values, y_train.values, X_val.values, y_val.values
