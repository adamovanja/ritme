import random
import time

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV

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


def fit_model(train, target, ls_features, model_type, seed):
    """Fit model to train set"""
    # choose estimator
    if model_type == "LinRegressor":
        estimator = LinearRegression()
        params = {}
    elif model_type == "RFRegressor":
        # todo: add CV fitting
        estimator = RandomForestRegressor(
            max_depth=2, random_state=seed, criterion="squared_error"
        )
        params = {
            "n_estimators": [10, 50, 100, 500],
            "max_depth": [2, 8, 16, None],
            "min_samples_split": [0.0001, 0.001, 0.01, 0.1],
            "min_samples_leaf": [0.00001, 0.0001],
            "max_features": [None, "sqrt", "log2", 0.1, 0.2, 0.5, 0.8],
            "min_impurity_decrease": [0.0001, 0.001, 0.01],
            "bootstrap": [True, False],
        }
    # elif model_type == "LSTM":
    #     estimator = KerasClassifier(
    #         build_fn=create_LSTM_model, epochs=10, batch_size=2, verbose=0
    #     )

    # CV for best parameters
    if len(params) > 0:
        # todo: adjust number of n_iter to try
        # todo: adjust scoring depending on classification vs. regression
        split_cv = min(5, train.shape[0])
        estimator = RandomizedSearchCV(
            estimator,
            params,
            n_iter=10,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            cv=split_cv,
            random_state=seed,
        )
    print("Training model...")
    start = time.time()
    # assumption: features to use for modelling all available in given feat
    model = estimator.fit(train[ls_features], train[target])
    end = time.time()
    dur_min = (end - start) / 60.0
    print(f"... lasted {dur_min} min.")

    # in case of CV
    if len(params) > 0:
        model = model.best_estimator_

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
    seed_data: int = SEED,
    seed_model: int = SEED,
):
    """Fit and predict model on data provided"""
    # assumption: features are all columns provided in feat
    ls_features = [x for x in feat.columns]
    # merge md_df and feat_df to flat table (maybe 3D needed for dynamic)
    data = md.join(feat, how="left")
    data.sort_values([host_id, target], inplace=True)

    train, test = split_data_by_host(data, host_id, train_size, seed_data)

    model = fit_model(train, target, ls_features, model_type, seed_model)

    # create model predictions: train & test
    pred_train = save_predictions(model, target, ls_features, train)
    pred_test = save_predictions(model, target, ls_features, test)

    return model, pred_train, pred_test


def run_models(dic_models, random_runs=10, seed_simulations=SEED):
    """
    Run `random_runs` model simulations - for demo purposes for now.
    `dic_model` must contain label of simulation as key and within item in this
    order: metadata, feature table and model to use.
    """
    random.seed(seed_simulations)
    seed_model_ls = random.sample(range(0, 10000), random_runs)

    out_dic = {}
    for tag, inpts in dic_models.items():
        md = inpts[0]
        ft = inpts[1]
        model = inpts[2]

        ls_train = []
        ls_test = []
        for seed_model in seed_model_ls:
            _, train, test = fit_n_predict_model(
                md,
                ft,
                "age_days",
                "host_id",
                model,
                # todo: decide do we want to shuffle data
                # todo: randomly or stick with fixed split?
                # seed_data=seed_model,
                seed_model=seed_model,
            )
            ls_train.append(train)
            ls_test.append(test)
        out_dic[tag] = [ls_train, ls_test]
    return out_dic
