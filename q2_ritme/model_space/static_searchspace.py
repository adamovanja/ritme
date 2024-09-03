from typing import Any, Dict, Optional


def _get_dependent_data_eng_space(trial, data_selection: str) -> None:
    if data_selection.endswith("_ith") or data_selection.endswith("_topi"):
        trial.suggest_int("data_selection_i", 1, 20)
    elif data_selection.endswith("_quantile"):
        trial.suggest_float("data_selection_q", 0.5, 0.9, step=0.1)
    elif data_selection.endswith("_threshold"):
        trial.suggest_float("data_selection_t", 0.00001, 0.01, log=True)


def get_data_eng_space(trial, tax, test_mode: bool = False) -> None:
    if test_mode:
        # note: test mode can be adjusted to whatever one wants to test
        data_selection = trial.suggest_categorical(
            "data_selection", ["abundance_ith", "variance_threshold"]
        )
        if data_selection is not None:
            _get_dependent_data_eng_space(trial, data_selection)

        trial.suggest_categorical("data_aggregation", [None])
        trial.suggest_categorical("data_transform", [None])
        return None

    # feature aggregation
    data_aggregation_options = (
        [None]
        if tax.empty
        else [None, "tax_class", "tax_order", "tax_family", "tax_genus"]
    )
    trial.suggest_categorical("data_aggregation", data_aggregation_options)

    # feature selection
    data_selection_options = [
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
    data_selection = trial.suggest_categorical("data_selection", data_selection_options)
    if data_selection is not None:
        _get_dependent_data_eng_space(trial, data_selection)

    # feature transform
    trial.suggest_categorical("data_transform", [None, "clr", "ilr", "alr", "pa"])
    return None


def get_linreg_space(trial, tax, test_mode: bool = False) -> Dict[str, str]:
    get_data_eng_space(trial, tax, test_mode)

    # alpha controls overall regularization strength, alpha = 0 is equivalent to
    # an ordinary least square. large alpha -> more regularization
    trial.suggest_float("alpha", 0, 1)
    # balance between L1 and L2 reg., when =0 -> L2, when =1 -> L1
    trial.suggest_float("l1_ratio", 0, 1)

    return {"model": "linreg"}


def get_rf_space(trial, tax, test_mode: bool = False) -> Dict[str, str]:
    get_data_eng_space(trial, tax, test_mode)

    # number of trees in forest: the more the higher computational costs
    trial.suggest_int("n_estimators", 40, 200, step=20)

    # max depths of the tree: the higher the higher probab of overfitting
    # ! 4, 8, 16, None
    trial.suggest_categorical("max_depth", [4, 8, 16, 32, None])

    # min number of samples requires to split internal node: small
    # values higher probab of overfitting
    # ! 0.001, 0.01, 0.1
    trial.suggest_float("min_samples_split", 0.001, 0.1, log=True)

    # a leaf should have at least this fraction of samples within - larger
    # values can help reduce overfitting
    # ! 0.0001, 0.001, 0.01
    trial.suggest_float("min_weight_fraction_leaf", 0.0001, 0.01, log=True)

    # min # samples requires at leaf node: small values higher probab
    # of overfitting
    trial.suggest_float("min_samples_leaf", 0.001, 0.1, log=True)

    # max # features to consider when looking for best split: small can
    # reduce overfitting
    # ! None, "sqrt", "log2", 0.1
    trial.suggest_categorical("max_features", [None, "sqrt", "log2", 0.1, 0.2, 0.5])

    # node split occurs if impurity is >= to this value: large values
    # prevent overfitting
    trial.suggest_float("min_impurity_decrease", 0.001, 0.5, log=True)

    # ! True, False
    trial.suggest_categorical("bootstrap", [True, False])

    return {"model": "rf"}


def get_nn_space(
    trial, tax, model_name: str, test_mode: bool = False
) -> Dict[str, str]:
    get_data_eng_space(trial, tax, test_mode)
    # todo: make max_layers configurable parameter!
    max_layers = 30
    # Sample random uniformly between [1,max_layers] rounding to multiples of 5
    trial.suggest_int("n_hidden_layers", 1, max_layers, step=5)
    trial.suggest_categorical(
        "learning_rate", [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    )
    trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    trial.suggest_categorical("epochs", [10, 50, 100, 200])

    # first and last layer are fixed by shape of features and target
    for i in range(max_layers):
        trial.suggest_categorical(f"n_units_hl{i}", [32, 64, 128, 256, 512])

    return {"model": model_name}


def get_xgb_space(trial, tax, test_mode: bool = False) -> Dict[str, Any]:
    get_data_eng_space(trial, tax, test_mode)

    # value between 2 and 6 is often a good starting point
    trial.suggest_int("max_depth", 2, 10)

    # depends on sample size: 0 - 2% # todo make depends on sample number
    trial.suggest_int("min_child_weight", 0, 4)
    trial.suggest_categorical("subsample", [0.7, 0.8, 0.9, 1.0])
    trial.suggest_categorical("eta", [0.01, 0.05, 0.1, 0.2, 0.3])
    trial.suggest_int("num_parallel_tree", 1, 3, step=1)

    return {"model": "xgb"}


def get_trac_space(trial, tax, test_mode: bool = False) -> Dict[str, Any]:
    # no feature_transformation to be used for trac
    # data_aggregate=taxonomy not an option because tax tree does not match with
    # regards to feature IDs here

    # with loguniform: sampled values are more densely concentrated
    # towards the lower end of the range
    trial.suggest_float("lambda", 1e-3, 1.0, log=True)
    data_eng_space_trac = {
        "data_aggregation": None,
        "data_selection": None,
        "data_selection_i": None,
        "data_selection_q": None,
        "data_selection_t": None,
        "data_transform": None,
    }
    return {"model": "trac", **data_eng_space_trac}


def get_search_space(
    trial, model_type: str, tax, test_mode: bool = False
) -> Optional[Dict[str, Any]]:
    """Creates the search space"""
    if model_type in ["xgb", "linreg", "rf", "trac"]:
        space_functions = {
            "xgb": get_xgb_space,
            "linreg": get_linreg_space,
            "rf": get_rf_space,
            "trac": get_trac_space,
        }
        return space_functions[model_type](trial, tax, test_mode)
    elif model_type in ["nn_reg", "nn_class", "nn_corn"]:
        return get_nn_space(trial, tax, model_type, test_mode)
    else:
        raise ValueError(f"Model type {model_type} not supported.")
