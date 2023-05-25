"""define hyperparameter search spaces"""
from ray import tune

# ! data engineering
data_eng_space = {
    # adjust to tune.grid_search checks all options:
    # new nb_trials= num_trials * nb of options in data_transform
    "data_transform": tune.grid_search([None, "clr", "ilr", "alr"]),
    # todo: remove manual setting of max number of features (19 now)
    "data_alr_denom_idx": tune.randint(0, 19),
}

# ! adding models
# linear regression
linreg_space = {
    **data_eng_space,
    "fit_intercept": tune.choice([True]),
}

# scikit random forest
rf_space = {
    **data_eng_space,
    "n_estimators": tune.randint(100, 1000),
    "max_depth": tune.randint(2, 32),
    "min_samples_split": tune.choice([0.0001, 0.001, 0.01, 0.1]),
    "min_samples_leaf": tune.choice([0.00001, 0.0001, 0.001]),
    "max_features": tune.choice([None, "sqrt", "log2", 0.1, 0.2, 0.5, 0.8]),
    "min_impurity_decrease": tune.choice([0.0001, 0.001, 0.01]),
    "bootstrap": tune.choice([True, False]),
}

# keras neural network
nn_space = {
    **data_eng_space,
    # Sample random uniformly between [1,9] rounding to multiples of 3
    "n_layers": tune.qrandint(1, 9, 3),
    "learning_rate": tune.loguniform(1e-5, 1e-1),
    "batch_size": tune.choice([32, 64, 128]),
}
for i in range(9):
    nn_space[f"n_units_l{i}"] = tune.randint(3, 64)

# xgb
xgb_space = {
    **data_eng_space,
    # todo: move into function as it does not change and does not need to be tracked
    "objective": "reg:squarederror",
    "max_depth": tune.randint(
        3, 10
    ),  # value between 2 and 6 is often a good starting point
    "min_child_weight": tune.randint(
        0, 4
    ),  # depends on sample size: 0 - 2% # todo make depends on sample number
    "subsample": tune.choice([0.7, 0.8, 0.9, 1.0]),
    "eta": tune.choice([0.01, 0.05, 0.1, 0.2, 0.3]),
    # "n_estimators": tune.choice([5, 10, 20, 50]) # todo add: nb gradient boosted trees
}
