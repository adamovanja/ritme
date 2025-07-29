import datetime
import unittest
import unittest.mock as mock

import pandas as pd
from optuna.trial import FrozenTrial
from optuna.trial._state import TrialState
from optuna.trial._trial import Trial
from parameterized import parameterized

from ritme.model_space import static_searchspace as ss


class MockTrial(Trial):
    def __init__(self):
        self._study = mock.MagicMock()
        self._study._storage = mock.MagicMock()
        real_frozen_trial = FrozenTrial(
            number=1,
            trial_id=1,
            state=TrialState.RUNNING,
            value=None,
            values=None,
            datetime_start=datetime.datetime.now(),
            datetime_complete=None,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            intermediate_values={},
        )
        self._study._storage.get_trial.return_value = real_frozen_trial
        super().__init__(study=self._study, trial_id=1)


class TestStaticSearchSpace(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.tax = pd.DataFrame()
        self.model_independent_params = {
            "data_aggregation": "tax_class",
            "data_selection": None,
            "data_transform": "clr",
            "data_enrich": None,
        }
        self.train_val = pd.DataFrame(
            {
                "F1": [0.1, 0.5, 0.85],
                "F2": [0.8, 0.2, 0.05],
                "F3": [0.1, 0.3, 0.1],
                "test_target": [7, 8, 9],
            }
        )

    @parameterized.expand(
        [
            ("abundance_ith", "i"),
            ("variance_quantile", "q"),
            ("abundance_threshold", "t"),
        ]
    )
    def test_get_dependent_data_eng_space(self, data_selection, expected_suffix):
        trial = MockTrial()
        ss._get_dependent_data_eng_space(trial, self.train_val, data_selection)

        hyperparam = f"data_selection_{expected_suffix}"
        self.assertIn(hyperparam, trial.params)
        self.assertIsNotNone(trial.params[hyperparam])

    @parameterized.expand(
        [
            ("abundance", 0.5, 1.45),
            ("variance", 0.01333333333333333, 0.15750000000000003),
        ]
    )
    def test_get_dependent_data_eng_space_data_dep_abundance_threshold(
        self, method, min, max
    ):
        trial = mock.Mock(
            suggest_float=mock.MagicMock(),
        )

        ss._get_dependent_data_eng_space(trial, self.train_val, f"{method}_threshold")

        trial.suggest_float.assert_called_once_with(
            "data_selection_t", min, max, log=True
        )

    @parameterized.expand(
        [
            ("nn_reg",),
            ("nn_class",),
            ("nn_corn",),
        ]
    )
    def test_get_nn_space(self, model_type):
        exp_model_params = {
            "n_hidden_layers": 2,
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 100,
            "n_units_hl0": 64,
            "n_units_hl1": 128,
            "dropout_rate": 0.1,
            "weight_decay": 0.0001,
            "early_stopping_patience": 5,
            "early_stopping_min_delta": 1e-4,
        }
        # init
        trial = MockTrial()
        trial._study.sampler.sample_independent.side_effect = list(
            self.model_independent_params.values()
        ) + list(exp_model_params.values())
        # call
        nn_space = ss.get_nn_space(trial, self.train_val, self.tax, model_type)

        # assert
        self.assertIsInstance(nn_space, dict)
        self.assertEqual(nn_space["model"], model_type)

        expected_params = {**exp_model_params, **self.model_independent_params}
        self.assertDictEqual(trial.params, expected_params)

    def test_get_trac_space(self):
        trial = MockTrial()
        trac_space = ss.get_trac_space(trial, self.train_val, self.tax)
        self.assertIsInstance(trac_space, dict)
        self.assertEqual(trac_space["model"], "trac")
        self.assertIn("lambda", trial.params)

    @parameterized.expand(
        [
            ("linreg", ss.get_linreg_space, {"alpha": 0.001, "l1_ratio": 0.2}),
            (
                "rf",
                ss.get_rf_space,
                {
                    "n_estimators": 100,
                    "max_depth": 16,
                    "min_samples_split": 0.01,
                    "min_weight_fraction_leaf": 0.0,
                    "min_samples_leaf": 0.01,
                    "max_features": "sqrt",
                    "min_impurity_decrease": 0.1,
                    "bootstrap": True,
                },
            ),
            (
                "xgb",
                ss.get_xgb_space,
                {
                    "max_depth": 6,
                    "min_child_weight": 2,
                    "subsample": 0.9,
                    "eta": 0.05,
                    "num_parallel_tree": 2,
                    "gamma": 0.1,
                    "reg_alpha": 0.2,
                    "reg_lambda": 0.5,
                    "colsample_bytree": 0.7,
                },
            ),
        ]
    )
    def test_get_others_space(self, model_name, func_get_space, exp_model_params):
        # init
        trial = MockTrial()
        trial._study.sampler.sample_independent.side_effect = list(
            self.model_independent_params.values()
        ) + list(exp_model_params.values())

        # call
        model_space = func_get_space(trial, self.train_val, self.tax)

        # assert
        self.assertIsInstance(model_space, dict)
        self.assertEqual(model_space["model"], model_name)

        expected_params = {**exp_model_params, **self.model_independent_params}
        self.assertDictEqual(trial.params, expected_params)

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
        trial = mock.Mock(
            suggest_categorical=mock.MagicMock(),
            suggest_int=mock.MagicMock(),
            suggest_float=mock.MagicMock(),
        )
        search_space = ss.get_search_space(trial, model_type, self.tax, self.train_val)
        self.assertIsInstance(search_space, dict)
        self.assertEqual(search_space["model"], model_type)

    def test_get_search_space_model_not_supported(self):
        model_type = "FakeModel"
        trial = MockTrial()
        with self.assertRaisesRegex(ValueError, "Model type FakeModel not supported."):
            ss.get_search_space(trial, model_type, self.tax, self.train_val)

    def _list_mocked_params_as_calls(self, exp_defaults):
        mocked_params = []
        for param_name, param_config in exp_defaults.items():
            if "log" in param_config:
                # suggest_float or int
                mocked_params.append(
                    mock.call(
                        param_name,
                        param_config["min"],
                        param_config["max"],
                        log=param_config["log"],
                    )
                )
            elif "step" in param_config:
                # suggest_float or int
                mocked_params.append(
                    mock.call(
                        param_name,
                        param_config["min"],
                        param_config["max"],
                        step=param_config["step"],
                    )
                )
            elif type(param_config) is list:
                # suggest_categorical

                # special case for NN hyperparameters
                i = 0
                if param_name == "n_units_hl":
                    param_name = f"n_units_hl{i}"
                    i += 1
                mocked_params.append(mock.call(param_name, param_config))
            else:
                # suggest_float or int
                mocked_params.append(
                    mock.call(param_name, param_config["min"], param_config["max"])
                )
        return mocked_params

    @parameterized.expand(
        [
            (
                "linreg",
                {
                    "alpha": {"min": 0.00001, "max": 100, "log": True},
                    "l1_ratio": {"min": 0, "max": 1, "step": 0.1},
                },
                {},
                {},
            ),
            (
                "rf",
                {
                    "min_samples_split": {"min": 0.001, "max": 0.1, "log": True},
                    "min_weight_fraction_leaf": {
                        "min": 0.0,
                        "max": 0.01,
                        "log": False,
                    },
                    "min_samples_leaf": {"min": 0.001, "max": 0.1, "log": True},
                    "min_impurity_decrease": {"min": 0.0, "max": 0.5, "log": False},
                },
                {"n_estimators": {"min": 40, "max": 200, "step": 20}},
                {
                    "max_depth": [4, 8, 16, 32, None],
                    "max_features": [None, "sqrt", "log2", 0.1, 0.2, 0.3, 0.5],
                    "bootstrap": [True, False],
                },
            ),
            (
                "xgb",
                {
                    "subsample": {"min": 0.7, "max": 1.0},
                    "eta": {"min": 0.01, "max": 0.3, "log": True},
                    "gamma": {"min": 0.0, "max": 5.0, "step": 0.1},
                    "reg_alpha": {"min": 1e-10, "max": 1.0, "log": True},
                    "reg_lambda": {"min": 1e-10, "max": 1.0, "log": True},
                    "colsample_bytree": {"min": 0.3, "max": 1.0},
                },
                {
                    "max_depth": {"min": 2, "max": 10},
                    "min_child_weight": {"min": 0, "max": 4},
                    "num_parallel_tree": {"min": 1, "max": 3, "step": 1},
                },
                {},
            ),
            (
                "nn_reg",
                {
                    "dropout_rate": {"min": 0.0, "max": 0.5, "step": 0.05},
                    "weight_decay": {"min": 1e-6, "max": 1e-2, "log": True},
                    "early_stopping_min_delta": {"min": 1e-5, "max": 1e-2, "log": True},
                },
                {
                    "n_hidden_layers": {"min": 5, "max": 30, "step": 5},
                    "early_stopping_patience": {"min": 2, "max": 10, "step": 1},
                },
                {
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
                    "n_units_hl0": [32, 64, 128, 256, 512],
                    "n_units_hl1": [32, 64, 128, 256, 512],
                },
            ),
        ]
    )
    def test_hyperparameter_default_used(self, model_type, exp_float, exp_int, exp_cat):
        """Verifies that the default hyperparameters are used when not passed."""
        trial = mock.Mock(
            suggest_categorical=mock.MagicMock(),
            suggest_int=mock.MagicMock(),
            suggest_float=mock.MagicMock(),
        )
        # if NN return n_hidden_layers = 2
        if model_type in ["nn_reg", "nn_class", "nn_corn"]:
            trial.suggest_int.side_effect = [2, 100, 100]

        # Call the actual function without passing model_hyperparameters
        _ = ss.get_search_space(trial, model_type, self.tax, self.train_val)

        # Verify that the trial parameters match the expected defaults
        if len(exp_float) > 0:
            mocked_params_float = self._list_mocked_params_as_calls(exp_float)
            trial.suggest_float.assert_has_calls(mocked_params_float)
        if len(exp_int) > 0:
            mocked_params_int = self._list_mocked_params_as_calls(exp_int)
            trial.suggest_int.assert_has_calls(mocked_params_int)
        if len(exp_cat) > 0:
            mocked_params_cat = self._list_mocked_params_as_calls(exp_cat)
            trial.suggest_categorical.assert_has_calls(mocked_params_cat)

    @parameterized.expand(
        [
            (
                "linreg",
                {
                    "alpha": {"min": 0.05, "max": 1, "log": True},
                    "l1_ratio": {"min": 0.2, "max": 0.5, "step": 0.1},
                },
                {},
                {},
            ),
            (
                "rf",
                {
                    "min_samples_split": {"min": 0.01, "max": 0.05, "log": True},
                    "min_weight_fraction_leaf": {
                        "min": 0.0005,
                        "max": 0.005,
                        "log": True,
                    },
                    "min_samples_leaf": {"min": 0.01, "max": 0.05, "log": True},
                    "min_impurity_decrease": {"min": 0.01, "max": 0.1, "log": True},
                },
                {"n_estimators": {"min": 50, "max": 150, "step": 10}},
                {
                    "max_depth": [5, 10, 15, None],
                    "max_features": [None, "sqrt", "log2", 0.2],
                    "bootstrap": [True],
                },
            ),
            (
                "xgb",
                {
                    "subsample": {"min": 0.8, "max": 1.0},
                    "eta": {"min": 0.05, "max": 0.2, "log": True},
                    "gamma": {"min": 0.1, "max": 3.0, "step": 0.1},
                    "reg_alpha": {"min": 0.1, "max": 0.5, "log": True},
                    "reg_lambda": {"min": 0.1, "max": 0.5, "log": True},
                    "colsample_bytree": {"min": 0.7, "max": 0.8},
                },
                {
                    "max_depth": {"min": 3, "max": 7},
                    "min_child_weight": {"min": 1, "max": 3},
                    "num_parallel_tree": {"min": 2, "max": 4, "step": 1},
                },
                {},
            ),
            (
                "nn_reg",
                {
                    "dropout_rate": {"min": 0.1, "max": 0.3, "step": 0.1},
                    "weight_decay": {"min": 0.0001, "max": 0.001, "log": True},
                    "early_stopping_min_delta": {
                        "min": 0.001,
                        "max": 0.01,
                        "log": True,
                    },
                },
                {
                    "n_hidden_layers": {"min": 2, "max": 10, "step": 2},
                    "early_stopping_patience": {"min": 5, "max": 20, "step": 5},
                },
                {
                    "learning_rate": [0.01, 0.001, 0.0001],
                    "batch_size": [64, 128],
                    "epochs": [2],
                    "n_units_hl": [32, 64],
                },
            ),
        ]
    )
    def test_hyperparameter_passing_different_than_default(
        self, model_type, exp_float, exp_int, exp_cat
    ):
        """Verifies that custom hyperparameters are used when passed."""
        trial = mock.Mock(
            suggest_categorical=mock.MagicMock(),
            suggest_int=mock.MagicMock(),
            suggest_float=mock.MagicMock(),
        )
        # if NN return n_hidden_layers = 2
        if model_type in ["nn_reg", "nn_class", "nn_corn"]:
            trial.suggest_int.side_effect = [2, 2, 2]

        # Call the actual function with passing model_hyperparameters
        exp_params = {**exp_float, **exp_int, **exp_cat}
        _ = ss.get_search_space(
            trial,
            model_type,
            self.tax,
            self.train_val,
            model_hyperparameters=exp_params,
        )

        # Verify that the trial parameters match the expected defaults
        if len(exp_float) > 0:
            mocked_params_float = self._list_mocked_params_as_calls(exp_float)
            trial.suggest_float.assert_has_calls(mocked_params_float)
        if len(exp_int) > 0:
            mocked_params_int = self._list_mocked_params_as_calls(exp_int)
            trial.suggest_int.assert_has_calls(mocked_params_int)
        if len(exp_cat) > 0:
            mocked_params_cat = self._list_mocked_params_as_calls(exp_cat)
            trial.suggest_categorical.assert_has_calls(mocked_params_cat)
