import pandas as pd
from parameterized import parameterized
from qiime2.plugin.testing import TestPluginBase

from q2_ritme.model_space import static_searchspace as ss


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


class TestStaticSearchSpace(TestPluginBase):
    package = "q2_ritme.tests"

    def setUp(self):
        super().setUp()
        self.tax = pd.DataFrame()

    @parameterized.expand(
        [
            ("abundance_ith", "i"),
            ("variance_quantile", "q"),
            ("abundance_threshold", "t"),
        ]
    )
    def test_get_dependent_data_eng_space(self, data_selection, expected_non_none):
        trial = MockTrial()
        ss._get_dependent_data_eng_space(trial, data_selection)

        expected_params = {"data_selection_i", "data_selection_q", "data_selection_t"}
        self.assertTrue(expected_params.issubset(trial.params.keys()))

        for suffix in "iqt":
            param = f"data_selection_{suffix}"
            if suffix == expected_non_none:
                self.assertIsNotNone(trial.params[param], f"{param} should not be None")
            else:
                self.assertIsNone(trial.params[param], f"{param} should be None")

    def test_get_data_eng_space_test_mode(self):
        trial = MockTrial()
        ss.get_data_eng_space(trial, self.tax, test_mode=True)
        expected_params = {"data_selection", "data_aggregation", "data_transform"}
        self.assertTrue(expected_params.issubset(trial.params.keys()))
        self.assertIn(trial.params["data_selection"], [None, "abundance_ith"])
        self.assertEqual(trial.params["data_aggregation"], None)
        self.assertEqual(trial.params["data_transform"], None)

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
        expected_params = {"alpha", "l1_ratio"}
        self.assertTrue(expected_params.issubset(trial.params.keys()))

    def test_get_rf_space(self):
        trial = MockTrial()
        rf_space = ss.get_rf_space(trial, self.tax)
        self.assertIsInstance(rf_space, dict)
        self.assertEqual(rf_space["model"], "rf")
        expected_params = {
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "min_impurity_decrease",
            "bootstrap",
        }
        self.assertTrue(expected_params.issubset(trial.params.keys()))

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

        expected_params = {"n_hidden_layers", "learning_rate", "batch_size", "epochs"}
        self.assertTrue(expected_params.issubset(trial.params.keys()))

        self.assertTrue(any(f"n_units_hl{i}" in trial.params for i in range(30)))

    def test_get_xgb_space(self):
        trial = MockTrial()
        xgb_space = ss.get_xgb_space(trial, self.tax)
        self.assertIsInstance(xgb_space, dict)
        self.assertEqual(xgb_space["model"], "xgb")
        expected_params = {
            "max_depth",
            "min_child_weight",
            "subsample",
            "eta",
            "n_estimators",
        }
        self.assertTrue(expected_params.issubset(trial.params.keys()))

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

    def test_get_optuna_points_to_evaluate(self):
        params_ls = ss.get_optuna_points_to_evaluate()
        self.assertEqual(len(params_ls), 225)
