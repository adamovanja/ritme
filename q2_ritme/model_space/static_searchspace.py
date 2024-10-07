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


def get_linreg_space(
    trial, tax, test_mode: bool = False, model_hyperparameters: dict = {}
) -> Dict[str, str]:
    get_data_eng_space(trial, tax, test_mode)

    # alpha controls overall regularization strength, alpha = 0 is equivalent to
    # an ordinary least square. large alpha -> more regularization
    alpha = model_hyperparameters.get("alpha", {"min": 0, "max": 1})
    trial.suggest_float("alpha", alpha["min"], alpha["max"])

    # balance between L1 and L2 reg., when =0 -> L2, when =1 -> L1
    l1_ratio = model_hyperparameters.get("l1_ratio", {"min": 0, "max": 1})
    trial.suggest_float("l1_ratio", l1_ratio["min"], l1_ratio["max"])

    return {"model": "linreg"}


def get_rf_space(
    trial, tax, test_mode: bool = False, model_hyperparameters: dict = {}
) -> Dict[str, str]:
    get_data_eng_space(trial, tax, test_mode)

    # number of trees in forest: the more the higher computational costs
    n_estimators = model_hyperparameters.get(
        "n_estimators", {"min": 40, "max": 200, "step": 20}
    )
    trial.suggest_int(
        "n_estimators",
        n_estimators["min"],
        n_estimators["max"],
        step=n_estimators["step"],
    )

    # max depths of the tree: the higher the higher probab of overfitting
    # ! 4, 8, 16, None
    max_depth = model_hyperparameters.get("max_depth", [4, 8, 16, 32, None])
    trial.suggest_categorical("max_depth", max_depth)

    # min number of samples requires to split internal node: small
    # values higher probab of overfitting
    # ! 0.001, 0.01, 0.1
    min_samples_split = model_hyperparameters.get(
        "min_samples_split", {"min": 0.001, "max": 0.1, "log": True}
    )
    trial.suggest_float(
        "min_samples_split",
        min_samples_split["min"],
        min_samples_split["max"],
        log=min_samples_split["log"],
    )

    # a leaf should have at least this fraction of samples within - larger
    # values can help reduce overfitting
    # ! 0.0001, 0.001, 0.01
    min_weight_fraction_leaf = model_hyperparameters.get(
        "min_weight_fraction_leaf", {"min": 0.0001, "max": 0.01, "log": True}
    )
    trial.suggest_float(
        "min_weight_fraction_leaf",
        min_weight_fraction_leaf["min"],
        min_weight_fraction_leaf["max"],
        log=min_weight_fraction_leaf["log"],
    )

    # min # samples requires at leaf node: small values higher probab
    # of overfitting
    min_samples_leaf = model_hyperparameters.get(
        "min_samples_leaf", {"min": 0.001, "max": 0.1, "log": True}
    )
    trial.suggest_float(
        "min_samples_leaf",
        min_samples_leaf["min"],
        min_samples_leaf["max"],
        log=min_samples_leaf["log"],
    )

    # max # features to consider when looking for best split: small can
    # reduce overfitting
    # ! None, "sqrt", "log2", 0.1
    max_features = model_hyperparameters.get(
        "max_features", [None, "sqrt", "log2", 0.1, 0.2, 0.5]
    )
    trial.suggest_categorical("max_features", max_features)

    # node split occurs if impurity is >= to this value: large values
    # prevent overfitting
    min_impurity_decrease = model_hyperparameters.get(
        "min_impurity_decrease", {"min": 0.001, "max": 0.5, "log": True}
    )
    trial.suggest_float(
        "min_impurity_decrease",
        min_impurity_decrease["min"],
        min_impurity_decrease["max"],
        log=min_impurity_decrease["log"],
    )

    # bootstrap: whether bootstrap samples are used when building trees
    # ! True, False
    bootstrap = model_hyperparameters.get("bootstrap", [True, False])
    trial.suggest_categorical("bootstrap", bootstrap)

    return {"model": "rf"}


def get_nn_space(
    trial,
    tax,
    model_name: str,
    test_mode: bool = False,
    model_hyperparameters: dict = {},
) -> Dict[str, str]:
    get_data_eng_space(trial, tax, test_mode)

    # max_layers: in network
    max_layers_options = model_hyperparameters.get(
        "max_layers", {"min": 5, "max": 30, "step": 5}
    )
    max_layers = trial.suggest_int(
        "max_layers",
        max_layers_options["min"],
        max_layers_options["max"],
        step=max_layers_options["step"],
    )

    # Sample random uniformly between [1,max_layers] rounding to multiples of 5
    n_hidden_layers = model_hyperparameters.get(
        "n_hidden_layers", {"min": 1, "max": max_layers, "step": 5}
    )
    trial.suggest_int(
        "n_hidden_layers",
        n_hidden_layers["min"],
        n_hidden_layers["max"],
        step=n_hidden_layers["step"],
    )

    learning_rate = model_hyperparameters.get(
        "learning_rate", [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    )
    trial.suggest_categorical("learning_rate", learning_rate)

    batch_size = model_hyperparameters.get("batch_size", [32, 64, 128, 256])
    trial.suggest_categorical("batch_size", batch_size)

    epochs = model_hyperparameters.get("epochs", [10, 50, 100, 200])
    trial.suggest_categorical("epochs", epochs)

    # first and last layer are fixed by shape of features and target
    n_units_hl = model_hyperparameters.get("n_units_hl", [32, 64, 128, 256, 512])
    for i in range(max_layers):
        trial.suggest_categorical(f"n_units_hl{i}", n_units_hl)

    return {"model": model_name}


def get_xgb_space(
    trial, tax, test_mode: bool = False, model_hyperparameters: dict = {}
) -> Dict[str, Any]:
    get_data_eng_space(trial, tax, test_mode)

    # max_depth: value between 2 and 6 is often a good starting point
    max_depth = model_hyperparameters.get("max_depth", {"min": 2, "max": 10})
    trial.suggest_int("max_depth", max_depth["min"], max_depth["max"])

    # min_child_weight: depends on sample size: 0 - 2%
    # todo make dependable on sample number
    min_child_weight = model_hyperparameters.get(
        "min_child_weight", {"min": 0, "max": 4}
    )
    trial.suggest_int(
        "min_child_weight", min_child_weight["min"], min_child_weight["max"]
    )

    # subsample
    subsample = model_hyperparameters.get("subsample", {"min": 0.7, "max": 1.0})
    trial.suggest_float("subsample", subsample["min"], subsample["max"])

    # eta (learning rate)
    eta = model_hyperparameters.get("eta", {"min": 0.01, "max": 0.3})
    trial.suggest_float("eta", eta["min"], eta["max"], log=True)

    # num_parallel_tree
    num_parallel_tree = model_hyperparameters.get(
        "num_parallel_tree", {"min": 1, "max": 3, "step": 1}
    )
    trial.suggest_int(
        "num_parallel_tree",
        num_parallel_tree["min"],
        num_parallel_tree["max"],
        step=num_parallel_tree["step"],
    )

    return {"model": "xgb"}


def get_trac_space(
    trial, tax, test_mode: bool = False, model_hyperparameters: dict = {}
) -> Dict[str, Any]:
    # no feature_transformation to be used for trac
    # data_aggregate=taxonomy not an option because tax tree does not match with
    # regards to feature IDs here

    # with loguniform: sampled values are more densely concentrated
    # towards the lower end of the range
    lambda_param = model_hyperparameters.get(
        "lambda", {"min": 1e-3, "max": 1.0, "log": True}
    )
    trial.suggest_float(
        "lambda", lambda_param["min"], lambda_param["max"], log=lambda_param["log"]
    )

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
    trial,
    model_type: str,
    tax,
    test_mode: bool = False,
    model_hyperparameters: Dict[str, Any] = {},
) -> Optional[Dict[str, Any]]:
    """Creates the search space"""
    if model_type in ["xgb", "linreg", "rf", "trac"]:
        space_functions = {
            "xgb": get_xgb_space,
            "linreg": get_linreg_space,
            "rf": get_rf_space,
            "trac": get_trac_space,
        }
        return space_functions[model_type](trial, tax, test_mode, model_hyperparameters)
    elif model_type in ["nn_reg", "nn_class", "nn_corn"]:
        return get_nn_space(trial, tax, model_type, test_mode, model_hyperparameters)
    else:
        raise ValueError(f"Model type {model_type} not supported.")
