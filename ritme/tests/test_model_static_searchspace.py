import unittest
from unittest.mock import patch

import pandas as pd
from parameterized import parameterized

from ritme.model_space import static_searchspace as ss


class MockTrial:
    def __init__(self):
        self.params = {}

    def suggest_categorical(self, name, categories):
        self.params[name] = categories[0] if categories else None
        return self.params[name]

    def suggest_int(self, name, low, high, step=1):
        self.params[name] = low
        return low

    def suggest_float(self, name, low, high, step=None, log=False):
        self.params[name] = low
        return low


class TestStaticSearchSpace(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.tax = pd.DataFrame()
        self.data_params_prefix = "data_"

    @parameterized.expand(
        [
            ("abundance_ith", "i"),
            ("variance_quantile", "q"),
            ("abundance_threshold", "t"),
        ]
    )
    def test_get_dependent_data_eng_space(self, data_selection, expected_suffix):
        trial = MockTrial()
        ss._get_dependent_data_eng_space(trial, data_selection)

        hyperparam = f"data_selection_{expected_suffix}"
        self.assertIn(hyperparam, trial.params)
        self.assertIsNotNone(trial.params[hyperparam])

    def test_get_data_eng_space(self):
        trial = MockTrial()
        ss.get_data_eng_space(trial, self.tax)
        expected_params = {"data_selection", "data_aggregation", "data_transform"}
        self.assertTrue(expected_params.issubset(trial.params.keys()))

    def test_get_linreg_space(self):
        trial = MockTrial()
        linreg_space = ss.get_linreg_space(trial, self.tax)
        self.assertIsInstance(linreg_space, dict)
        self.assertEqual(linreg_space["model"], "linreg")

        trial_model_params = {
            k for k in trial.params.keys() if not k.startswith(self.data_params_prefix)
        }
        expected_params = {"alpha", "l1_ratio"}
        self.assertSetEqual(trial_model_params, expected_params)

    def test_get_rf_space(self):
        trial = MockTrial()
        rf_space = ss.get_rf_space(trial, self.tax)
        self.assertIsInstance(rf_space, dict)
        self.assertEqual(rf_space["model"], "rf")

        trial_model_params = {
            k for k in trial.params.keys() if not k.startswith(self.data_params_prefix)
        }
        expected_params = {
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_weight_fraction_leaf",
            "min_samples_leaf",
            "max_features",
            "min_impurity_decrease",
            "bootstrap",
        }
        self.assertSetEqual(trial_model_params, expected_params)

    @parameterized.expand(
        [
            ("nn_reg",),
            ("nn_class",),
            ("nn_corn",),
        ]
    )
    def test_get_nn_space(self, model_type):
        trial = MockTrial()
        nn_space = ss.get_nn_space(trial, self.tax, model_type)

        self.assertIsInstance(nn_space, dict)
        self.assertEqual(nn_space["model"], model_type)

        trial_model_params = {
            k for k in trial.params.keys() if not k.startswith(self.data_params_prefix)
        }
        expected_params = {
            "n_hidden_layers",
            "learning_rate",
            "batch_size",
            "epochs",
        }.union({f"n_units_hl{i}" for i in range(5)})
        self.assertSetEqual(trial_model_params, expected_params)

        self.assertTrue(any(f"n_units_hl{i}" in trial.params for i in range(30)))

    def test_get_xgb_space(self):
        trial = MockTrial()
        xgb_space = ss.get_xgb_space(trial, self.tax)
        self.assertIsInstance(xgb_space, dict)
        self.assertEqual(xgb_space["model"], "xgb")

        trial_model_params = {
            k for k in trial.params.keys() if not k.startswith(self.data_params_prefix)
        }
        expected_params = {
            "max_depth",
            "min_child_weight",
            "subsample",
            "eta",
            "num_parallel_tree",
            "gamma",
            "reg_alpha",
            "reg_lambda",
            "colsample_bytree",
        }
        self.assertSetEqual(trial_model_params, expected_params)

    def test_get_trac_space(self):
        trial = MockTrial()
        trac_space = ss.get_trac_space(trial, self.tax)
        self.assertIsInstance(trac_space, dict)
        self.assertEqual(trac_space["model"], "trac")
        self.assertIn("lambda", trial.params)

    @parameterized.expand(
        [
            ("xgb",),
            ("nn_reg",),
            ("nn_class",),
            ("nn_corn",),
            ("linreg",),
            ("rf",),
            ("trac",),
        ]
    )
    def test_get_search_space(self, model_type):
        trial = MockTrial()
        search_space = ss.get_search_space(trial, model_type, self.tax)
        self.assertIsInstance(search_space, dict)
        self.assertEqual(search_space["model"], model_type)

    def test_get_search_space_model_not_supported(self):
        model_type = "FakeModel"
        trial = MockTrial()
        with self.assertRaisesRegex(ValueError, "Model type FakeModel not supported."):
            ss.get_search_space(trial, model_type, self.tax)

    @parameterized.expand(
        [
            (
                "linreg",
                {
                    "alpha": {"min": 0.1, "max": 0.9},
                    "l1_ratio": {"min": 0.2, "max": 0.8},
                },
            ),
            (
                "rf",
                {
                    "n_estimators": {"min": 50, "max": 150, "step": 10},
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": {"min": 0.01, "max": 0.05, "log": True},
                    "min_weight_fraction_leaf": {
                        "min": 0.0005,
                        "max": 0.005,
                        "log": True,
                    },
                    "min_samples_leaf": {"min": 0.01, "max": 0.05, "log": True},
                    "max_features": [None, "sqrt", "log2", 0.2],
                    "min_impurity_decrease": {"min": 0.01, "max": 0.1, "log": True},
                    "bootstrap": [True, False],
                },
            ),
            (
                "xgb",
                {
                    "max_depth": {"min": 3, "max": 7},
                    "min_child_weight": {"min": 1, "max": 3},
                    "subsample": {"min": 0.8, "max": 1.0},
                    "eta": {"min": 0.05, "max": 0.2, "log": True},
                    "num_parallel_tree": {"min": 2, "max": 4, "step": 1},
                },
            ),
            (
                "nn_reg",
                {
                    "n_hidden_layers": {"min": 2, "max": 10, "step": 2},
                    "learning_rate": [0.01, 0.001, 0.0001],
                    "batch_size": [64, 128],
                    "epochs": [50, 100],
                    "n_units_hl": [64, 128, 256],
                },
            ),
        ]
    )
    @patch("ritme.model_space.static_searchspace.get_search_space")
    def test_hyperparameter_passing_different_than_default(
        self, model_type, hyperparameters, mock_get_search_space
    ):
        trial = MockTrial()
        ss.get_search_space(
            trial,
            model_type,
            self.tax,
            model_hyperparameters=hyperparameters,
        )

        # Verify that the get_search_space function was called with the correct
        # hyperparameters
        mock_get_search_space.assert_called_once_with(
            trial,
            model_type,
            self.tax,
            model_hyperparameters=hyperparameters,
        )

    def _verify_trial_params(self, trial, expected_defaults):
        for param, config in expected_defaults.items():
            if param == "n_units_hl":
                self._verify_n_units_hl(trial, config)
            else:
                self._verify_param(trial, param, config)

    def _verify_param(self, trial, param, config):
        self.assertIn(param, trial.params)
        if isinstance(config, dict):
            self.assertGreaterEqual(trial.params[param], config["min"])
            self.assertLessEqual(trial.params[param], config["max"])
        else:
            self.assertIn(trial.params[param], config)

    def _verify_n_units_hl(self, trial, config):
        n_hidden_layers_selected = trial.params["n_hidden_layers"]
        for i in range(n_hidden_layers_selected):
            param_name = f"n_units_hl{i}"
            self.assertIn(param_name, trial.params)
            self.assertIn(trial.params[param_name], config)

    @parameterized.expand(
        [
            (
                "linreg",
                {"alpha": {"min": 0, "max": 1}, "l1_ratio": {"min": 0, "max": 1}},
            ),
            (
                "rf",
                {
                    "n_estimators": {"min": 40, "max": 200, "step": 20},
                    "max_depth": [4, 8, 16, 32, None],
                    "min_samples_split": {"min": 0.001, "max": 0.1, "log": True},
                    "min_weight_fraction_leaf": {
                        "min": 0.0,
                        "max": 0.01,
                        "log": False,
                    },
                    "min_samples_leaf": {"min": 0.001, "max": 0.1, "log": True},
                    "max_features": [None, "sqrt", "log2", 0.1, 0.2, 0.3, 0.5],
                    "min_impurity_decrease": {"min": 0.0, "max": 0.5, "log": False},
                    "bootstrap": [True, False],
                },
            ),
            (
                "xgb",
                {
                    "max_depth": {"min": 2, "max": 10},
                    "min_child_weight": {"min": 0, "max": 4},
                    "subsample": {"min": 0.7, "max": 1.0},
                    "eta": {"min": 0.01, "max": 0.3, "log": True},
                    "num_parallel_tree": {"min": 1, "max": 3, "step": 1},
                },
            ),
            (
                "nn_reg",
                {
                    "n_hidden_layers": {"min": 1, "max": 30, "step": 5},
                    "learning_rate": [
                        0.01,
                        0.005,
                        0.001,
                        0.0005,
                        0.0001,
                        0.00005,
                        0.00001,
                    ],
                    "batch_size": [32, 64, 128, 256],
                    "epochs": [10, 50, 100, 200],
                    "n_units_hl": [32, 64, 128, 256, 512],
                },
            ),
        ]
    )
    def test_hyperparameter_default_used(self, model_type, expected_defaults):
        """Verifies that the default hyperparameters are used when not passed."""
        trial = MockTrial()

        # Call the actual function without passing model_hyperparameters
        _ = ss.get_search_space(trial, model_type, self.tax)

        # Verify that the trial parameters match the expected defaults
        self._verify_trial_params(trial, expected_defaults)
