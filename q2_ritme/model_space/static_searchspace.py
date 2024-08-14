from ray import tune


def get_data_eng_space(tax, test_mode=False):
    if test_mode:
        # note: test mode can be adjusted to whatever one wants to test
        return {
            "data_aggregation": None,
            "data_selection": tune.choice([None, "abundance_ith"]),
            "dsi_option": tune.randint(1, 20),
            "dsq_option": tune.quniform(0.5, 0.9, 0.1),
            "dst_option": tune.choice(
                [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
            ),
            "data_transform": None,
        }
    return {
        # grid search specified here checks all options: so new nb_trials=
        # num_trials * nb of options w gridsearch * nb of model types
        "data_aggregation": tune.choice(
            [None, "tax_class", "tax_order", "tax_family", "tax_genus"]
        )
        if not tax.empty
        else None,
        "data_selection": tune.choice(
            [
                None,
                "abundance_ith",
                "variance_ith",
                "abundance_topi",
                "variance_topi",
                "abundance_quantile",
                "variance_quantile",
                "abundance_threshold",
                "variance_threshold",
            ]
        ),
        # top i and ith values
        "dsi_option": tune.randint(1, 20),
        # quantile:
        "dsq_option": tune.quniform(0.5, 0.9, 0.1),
        # threshold:
        "dst_option": tune.choice(
            [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
        ),
        "data_transform": tune.choice([None, "clr", "ilr", "alr", "pa"]),
    }


def get_linreg_space(tax, test_mode=False):
    data_eng_space = get_data_eng_space(tax, test_mode)
    return dict(
        model="linreg",
        **data_eng_space,
        **{
            "fit_intercept": True,
            # alpha controls overall regularization strength, alpha = 0 is equivalent to
            # an ordinary least square. large alpha -> more regularization
            "alpha": tune.uniform(0, 1),
            # balance between L1 and L2 reg., when =0 -> L2, when =1 -> L1
            "l1_ratio": tune.uniform(0, 1),
        },
    )


def get_rf_space(tax, test_mode=False):
    data_eng_space = get_data_eng_space(tax, test_mode)
    return dict(
        model="rf",
        **data_eng_space,
        **{
            # number of trees in orest: the more the higher computational costs
            "n_estimators": tune.randint(50, 300),
            # max depths of the tree: the higher the higher probab of overfitting
            "max_depth": tune.randint(5, 50),
            # min number of samples requires to split internal node: small
            # values higher probab of overfitting
            "min_samples_split": tune.quniform(0.01, 0.1, 0.01),
            # min # samples requires at leaf node: small values higher probab
            # of overfitting
            "min_samples_leaf": tune.choice([0.005, 0.01, 0.05, 0.1]),
            # max # features to consider when looking for best split: small can
            # reduce overfitting
            "max_features": tune.choice([None, "sqrt", "log2", 0.1, 0.2, 0.5]),
            # node split occurs if impurity is >= to this value: large values
            # prevent overfitting
            "min_impurity_decrease": tune.quniform(0.01, 0.1, 0.01),
            "bootstrap": tune.choice([True, False]),
        },
    )


def get_nn_space(tax, model_name, test_mode=False):
    data_eng_space = get_data_eng_space(tax, test_mode)
    max_layers = 30
    nn_space = {
        # Sample random uniformly between [1,max_layers] rounding to multiples of 5
        "n_hidden_layers": tune.qrandint(1, max_layers, 5),
        "learning_rate": tune.choice(
            [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
        ),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "epochs": tune.choice([10, 50, 100, 200]),
    }
    # first and last layer are fixed by shape of features and target
    for i in range(0, max_layers):
        nn_space[f"n_units_hl{i}"] = tune.choice([32, 64, 128, 256, 512])
    return dict(model=model_name, **data_eng_space, **nn_space)


def get_xgb_space(tax, test_mode=False):
    data_eng_space = get_data_eng_space(tax, test_mode)
    return dict(
        model="xgb",
        **data_eng_space,
        **{
            "objective": "reg:squarederror",
            # value between 2 and 6 is often a good starting point
            "max_depth": tune.randint(2, 10),
            # depends on sample size: 0 - 2% # todo make depends on sample number
            "min_child_weight": tune.randint(0, 4),
            "subsample": tune.choice([0.7, 0.8, 0.9, 1.0]),
            "eta": tune.choice([0.01, 0.05, 0.1, 0.2, 0.3]),
            "n_estimators": tune.qrandint(10, 100, 10),
        },
    )


def get_trac_space(tax, test_mode=False):
    # no feature_transformation to be used for trac
    # data_aggregate=taxonomy not an option because tax tree does not match with
    # regards to feature IDs here
    data_eng_space_trac = {
        "data_aggregation": None,
        "data_selection": None,
        "data_selection_i": None,
        "data_selection_q": None,
        "data_selection_t": None,
        "data_transform": None,
    }
    return dict(
        model="trac",
        **data_eng_space_trac,
        **{
            # with loguniform: sampled values are more densely concentrated
            # towards the lower end of the range
            "lambda": tune.loguniform(1e-3, 1.0)
        },
    )


def get_search_space(tax, test_mode=False):
    return {
        "xgb": get_xgb_space(tax, test_mode),
        "nn_reg": get_nn_space(tax, "nn_reg", test_mode),
        "nn_class": get_nn_space(tax, "nn_class", test_mode),
        "nn_corn": get_nn_space(tax, "nn_corn", test_mode),
        "linreg": get_linreg_space(tax, test_mode),
        "rf": get_rf_space(tax, test_mode),
        "trac": get_trac_space(tax, test_mode),
    }
