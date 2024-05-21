from ray import tune


def find_most_nonzero_feature_idx(data):
    """
    Find the index of the first feature with the most non-zero values.

    Args:
        data (pd.DataFrame): DataFrame containing features.

    Returns:
        int: Index of the feature with the most non-zero values.
    """
    nonzero_counts = (data != 0).sum()
    if nonzero_counts.max() > 0:
        feature_name = nonzero_counts.idxmax()
        return data.columns.get_loc(feature_name)
    else:
        raise ValueError("All features are zero in all samples.")


def get_alr_denom_idx_space(train_val):
    features = [x for x in train_val if x.startswith("F")]
    nonzero_feature_idx = find_most_nonzero_feature_idx(train_val[features])
    return nonzero_feature_idx


def get_data_eng_space(train_val):
    return {
        # tune.grid_search specified here checks all options: so new nb_trials=
        # num_trials * nb of options in data_transform * nb of model types
        "data_transform": tune.grid_search([None, "clr", "ilr", "alr"]),
        "data_alr_denom_idx": get_alr_denom_idx_space(train_val),
    }


def get_linreg_space(train_val):
    data_eng_space = get_data_eng_space(train_val)
    return dict(
        model="linreg",
        **data_eng_space,
        **{
            "fit_intercept": True,
            # alpha controls overall regularization strength, alpha = 0 is equivalent to
            # an ordinary least square. large alpha -> more regularization
            "alpha": tune.loguniform(1e-5, 1.0),
            # balance between L1 and L2 reg., when =0 -> L2, when =1 -> L1
            "l1_ratio": tune.uniform(0, 1),
        },
    )


def get_rf_space(train_val):
    data_eng_space = get_data_eng_space(train_val)
    return dict(
        model="rf",
        **data_eng_space,
        **{
            "n_estimators": tune.randint(50, 300),
            "max_depth": tune.randint(2, 32),
            "min_samples_split": tune.choice([0.001, 0.01, 0.1]),
            "min_samples_leaf": tune.choice([0.0001, 0.001]),
            "max_features": tune.choice([None, "sqrt", "log2", 0.1, 0.2, 0.5, 0.8]),
            "min_impurity_decrease": tune.choice([0.0001, 0.001, 0.01]),
            "bootstrap": tune.choice([True, False]),
        },
    )


def get_nn_space(train_val, model_name):
    data_eng_space = get_data_eng_space(train_val)
    max_layers = 12
    nn_space = {
        # Sample random uniformly between [1,9] rounding to multiples of 3
        "n_hidden_layers": tune.qrandint(1, max_layers, 3),
        "learning_rate": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
        "epochs": tune.choice([10, 50, 100, 200]),
    }
    # first and last layer are fixed by shape of features and target
    for i in range(0, max_layers):
        # todo: increase!
        nn_space[f"n_units_hl{i}"] = tune.randint(3, 64)
    return dict(model=model_name, **data_eng_space, **nn_space)


def get_xgb_space(train_val):
    data_eng_space = get_data_eng_space(train_val)
    return dict(
        model="xgb",
        **data_eng_space,
        **{
            "objective": "reg:squarederror",
            # value between 2 and 6 is often a good starting point
            "max_depth": tune.randint(3, 10),
            # depends on sample size: 0 - 2% # todo make depends on sample number
            "min_child_weight": tune.randint(0, 4),
            "subsample": tune.choice([0.7, 0.8, 0.9, 1.0]),
            "eta": tune.choice([0.01, 0.05, 0.1, 0.2, 0.3]),
            # "n_estimators": tune.choice([5, 10, 20, 50])
            # todo add: nb gradient boosted trees
        },
    )


def get_trac_space(train_val):
    # no feature_transformation to be used for trac
    data_eng_space = {"data_transform": None, "data_alr_denom_idx": None}
    return dict(
        model="trac",
        **data_eng_space,
        **{
            # with loguniform: sampled values are more densely concentrated
            # towards the lower end of the range
            "lambda": tune.loguniform(1e-3, 1.0)
        },
    )


def get_search_space(train_val):
    return {
        "xgb": get_xgb_space(train_val),
        "nn_reg": get_nn_space(train_val, "nn_reg"),
        "nn_class": get_nn_space(train_val, "nn_class"),
        "nn_corn": get_nn_space(train_val, "nn_corn"),
        "linreg": get_linreg_space(train_val),
        "rf": get_rf_space(train_val),
        "trac": get_trac_space(train_val),
    }
