import time

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit

TARGET = "age_days"
HOST_ID = "host_id"
SEED = 12
TRAIN_SIZE = 0.8


def split_data_by_host(data, host_id, train_size=TRAIN_SIZE, seed=SEED):
    """Randomly split dataset into train & test split based on host_id"""
    if len(data[host_id].unique()) == 1:
        raise ValueError("Only one unique host available in dataset.")

    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    split = gss.split(data, groups=data[host_id])
    train_idx, test_idx = next(split)

    train, test = data.iloc[train_idx], data.iloc[test_idx]
    print(f"Train: {train.shape}, Test: {test.shape}")

    return train, test


def fit_model(train, target, ls_features, model_type):
    """Fit model to train set"""
    # choose estimator
    if model_type == "LinReg":
        estimator = LinearRegression()
    # elif model_type == "LSTM":
    #     estimator = KerasClassifier(
    #         build_fn=create_LSTM_model, epochs=10, batch_size=2, verbose=0
    #     )

    print("Training model...")
    start = time.time()
    # assumption: features to use for modelling all available in given feat
    model = estimator.fit(train[ls_features], train[target])
    end = time.time()
    dur_min = (end - start) / 60.0
    print(f"... lasted {dur_min} min.")

    return model


def save_predictions(model, target, ls_features, subset):
    # id, true
    saved_pred = subset[[target]].copy()
    saved_pred.rename(columns={target: "true"}, inplace=True)
    # pred
    saved_pred["pred"] = model.predict(subset[ls_features])
    return saved_pred


def fit_n_predict_model(
    md: pd.DataFrame,
    feat: pd.DataFrame,
    target: str = TARGET,
    host_id: str = HOST_ID,
    model_type: str = "LinReg",
    train_size: float = TRAIN_SIZE,
    seed: int = SEED,
):
    """Fit and predict model on data provided"""
    # assumption: features are all columns provided in feat
    ls_features = [x for x in feat.columns]
    # merge md_df and feat_df to flat table (maybe 3D needed for dynamic)
    data = md.join(feat, how="left")
    data.sort_values([host_id, target], inplace=True)

    train, test = split_data_by_host(data, host_id, train_size, seed)

    model = fit_model(train, target, ls_features, model_type)

    # create model predictions: train & test
    pred_train = save_predictions(model, target, ls_features, train)
    pred_test = save_predictions(model, target, ls_features, test)

    # TODO: save config of experiment in directory
    # todo: store model: see https://scikit-learn.org/stable/model_persistence.html
    return model, pred_train, pred_test
