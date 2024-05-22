import pandas as pd
from qiime2.plugin.testing import TestPluginBase

from q2_ritme.model_space import static_searchspace as ss


class TestFindNonzeroFeatureIdx(TestPluginBase):
    package = "q2_ritme.tests"

    def setUp(self):
        super().setUp()

    def test_find_most_nonzero_feature_idx_with_nonzero_feature(self):
        data = pd.DataFrame(
            {"F1": [0.1, 0.2, 0.7], "F2": [0, 0, 0], "F3": [0.9, 0.8, 0.3]}
        )
        expected_idx = 0
        self.assertEqual(ss.find_most_nonzero_feature_idx(data), expected_idx)

    def test_find_most_nonzero_feature_idx_with_all_zero_features(self):
        data = pd.DataFrame({"F1": [0, 0, 0], "F2": [0, 0, 0], "F3": [0, 0, 0]})
        with self.assertRaises(ValueError):
            ss.find_most_nonzero_feature_idx(data)

    def test_find_most_nonzero_feature_idx_with_empty_dataframe(self):
        data = pd.DataFrame()
        with self.assertRaises(ValueError):
            ss.find_most_nonzero_feature_idx(data)


class TestStaticSearchSpace(TestPluginBase):
    package = "q2_ritme.tests"

    def setUp(self):
        super().setUp()
        self.train_val = pd.DataFrame(
            {"F1": [1, 2, 3], "F2": [4, 5, 6], "F3": [7, 8, 9]}
        )

    def test_get_search_space(self):
        search_space = ss.get_search_space(self.train_val)

        self.assertIsInstance(search_space, dict)
        self.assertIn("xgb", search_space)
        self.assertIn("nn_reg", search_space)
        self.assertIn("nn_class", search_space)
        self.assertIn("nn_corn", search_space)
        self.assertIn("linreg", search_space)
        self.assertIn("rf", search_space)

    def test_get_data_eng_space(self):
        data_eng_space = ss.get_data_eng_space(self.train_val)

        self.assertIsInstance(data_eng_space, dict)
        self.assertIn("data_transform", data_eng_space)
        self.assertIn("data_alr_denom_idx", data_eng_space)

    def test_get_data_alr_denom_idx_space(self):
        data_alr_denom_idx_space = ss.get_alr_denom_idx_space(self.train_val)

        self.assertIsInstance(data_alr_denom_idx_space, int)
        self.assertEqual(data_alr_denom_idx_space, 0)

    def test_get_linreg_space(self):
        linreg_space = ss.get_linreg_space(self.train_val)

        self.assertIsInstance(linreg_space, dict)
        self.assertEqual(linreg_space["model"], "linreg")
        self.assertIn("data_transform", linreg_space)
        self.assertIn("data_alr_denom_idx", linreg_space)
        self.assertIn("fit_intercept", linreg_space)
        self.assertIn("alpha", linreg_space)
        self.assertIn("l1_ratio", linreg_space)

    def test_get_rf_space(self):
        rf_space = ss.get_rf_space(self.train_val)

        self.assertIsInstance(rf_space, dict)
        self.assertEqual(rf_space["model"], "rf")
        self.assertIn("data_transform", rf_space)
        self.assertIn("data_alr_denom_idx", rf_space)
        self.assertIn("n_estimators", rf_space)
        self.assertIn("max_depth", rf_space)
        self.assertIn("min_samples_split", rf_space)
        self.assertIn("min_samples_leaf", rf_space)
        self.assertIn("max_features", rf_space)
        self.assertIn("min_impurity_decrease", rf_space)
        self.assertIn("bootstrap", rf_space)

    def test_get_xgb_space(self):
        xgb_space = ss.get_xgb_space(self.train_val)

        self.assertIsInstance(xgb_space, dict)
        self.assertEqual(xgb_space["model"], "xgb")
        self.assertIn("data_transform", xgb_space)
        self.assertIn("data_alr_denom_idx", xgb_space)
        self.assertIn("objective", xgb_space)
        self.assertIn("max_depth", xgb_space)
        self.assertIn("min_child_weight", xgb_space)
        self.assertIn("subsample", xgb_space)
        self.assertIn("eta", xgb_space)

    def test_get_nn_space(self):
        nn_space = ss.get_nn_space(self.train_val, "nn_reg")

        self.assertIsInstance(nn_space, dict)
        self.assertEqual(nn_space["model"], "nn_reg")
        self.assertIn("data_transform", nn_space)
        self.assertIn("data_alr_denom_idx", nn_space)
        self.assertIn("n_hidden_layers", nn_space)
        self.assertIn("learning_rate", nn_space)
        self.assertIn("batch_size", nn_space)
        self.assertIn("epochs", nn_space)
