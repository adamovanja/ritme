import json
import os
import unittest

import pandas as pd
from parameterized import parameterized

from ritme.split_train_test import split_train_test


class TestConfigFiles(unittest.TestCase):
    """
    Verify that experiment config files suggested in config folder match the
    newest software code
    """

    def setUp(self):
        super().setUp()
        base_dir = os.path.dirname(__file__)
        self.path_to_config = os.path.join(base_dir, "..", "..", "config")

        self.data_rel = pd.DataFrame(
            {
                "host_id": ["c", "b", "c", "a"],
                "age_months": [1, 2, 5, 7],
                "covariate": [0, 1, 0, 1],
                "F0": [0.83333333333333333, 0.25, 1.0, 1.0],
                "F1": [0.16666666666666666, 0.75, 0.0, 0.0],
            }
        )
        self.data_rel.index = ["SR1", "SR2", "SR3", "SR4"]
        self.md = self.data_rel[["host_id", "age_months", "covariate"]]
        self.ft_rel = self.data_rel[["F0", "F1"]]

    @parameterized.expand(
        [
            "run_config.json",
            "example_linreg.json",
            "example_linreg_py.json",
            "run_config_whparams.json",
        ]
    )
    def test_split_train_test(self, filename_json):
        config_path = os.path.join(self.path_to_config, filename_json)
        with open(config_path) as f:
            run_config = json.load(f)

        try:
            split_train_test(
                md=self.md,
                ft=self.ft_rel,
                group_by_column=run_config["group_by_column"],
                train_size=0.8,
                seed=run_config["seed_data"],
            )
        except Exception as e:
            self.fail(
                f"Failed with {filename_json} " f"and this exception was raised: {e}"
            )

    @parameterized.expand(
        [
            "run_config.json",
            "run_config_whparams.json",
        ]
    )
    def test_find_best_model_config_models_listed(self, filename_json):
        config_path = os.path.join(self.path_to_config, filename_json)
        with open(config_path) as f:
            run_config = json.load(f)

        # Check if all of models_list are valid (i.e., no unknown models)
        config_models = set(run_config["ls_model_types"])
        allowed_models = {
            "linreg",
            "trac",
            "xgb",
            "nn_reg",
            "nn_class",
            "nn_corn",
            "rf",
        }
        self.assertTrue(
            config_models.issubset(allowed_models),
            f"Unknown models found: {config_models - allowed_models}",
        )

    @parameterized.expand(
        [
            (
                "linreg",
                {
                    "alpha": ["min", "max"],
                    "l1_ratio": ["min", "max"],
                    "start_points_to_evaluate": [
                        {
                            "alpha": 0.1,
                            "data_aggregation": "tax_class",
                            "data_enrich": None,
                            "data_selection": "abundance_threshold",
                            "data_transform": "clr",
                            "l1_ratio": 0.5,
                        }
                    ],
                },
            ),
            (
                "rf",
                {
                    "n_estimators": ["min", "max"],
                    "max_depth": None,
                    "min_samples_split": ["min", "max", "log"],
                    "min_weight_fraction_leaf": ["min", "max", "log"],
                    "min_samples_leaf": ["min", "max", "log"],
                    "max_features": None,
                    "min_impurity_decrease": ["min", "max", "log"],
                    "bootstrap": None,
                },
            ),
            (
                "nn_all_types",
                {
                    "n_hidden_layers": ["min", "max"],
                    "learning_rate": None,
                    "batch_size": None,
                    "epochs": None,
                    "n_units_hl": None,
                    "dropout_rate": ["min", "max"],
                    "weight_decay": ["min", "max", "log"],
                    "early_stopping_patience": ["min", "max"],
                    "early_stopping_min_delta": ["min", "max", "log"],
                },
            ),
            (
                "xgb",
                {
                    "n_estimators": ["min", "max"],
                    "max_depth": ["min", "max"],
                    "min_child_weight": ["min", "max"],
                    "subsample": ["min", "max"],
                    "eta": ["min", "max"],
                    "num_parallel_tree": ["min", "max"],
                    "gamma": ["min", "max"],
                    "reg_alpha": ["min", "max", "log"],
                    "reg_lambda": ["min", "max", "log"],
                    "colsample_bytree": ["min", "max"],
                },
            ),
            (
                "trac",
                {"lambda": ["min", "max", "log"]},
            ),
        ]
    )
    def test_custom_template_hparams_for_find_best_model(self, model, hparams):
        config_path = os.path.join(self.path_to_config, "run_config_whparams.json")
        with open(config_path) as f:
            run_config = json.load(f)

        # Check if all keys in hparams are present in run_config
        for k, v in hparams.items():
            self.assertIn(k, run_config["model_hyperparameters"][model])
            if v is not None:
                for vi in v:
                    self.assertIn(vi, run_config["model_hyperparameters"][model][k])

        # Check if all keys in run_config are present in hparams
        for k_config, v_config in run_config["model_hyperparameters"][model].items():
            self.assertIn(k_config, hparams)
            if hparams[k_config] is not None:
                for vi_config in hparams[k_config]:
                    self.assertIn(vi_config, v_config)
