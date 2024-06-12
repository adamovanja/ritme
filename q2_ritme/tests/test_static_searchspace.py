import pandas as pd
from qiime2.plugin.testing import TestPluginBase
from ray import tune

from q2_ritme.model_space import static_searchspace as ss


class TestStaticSearchSpace(TestPluginBase):
    package = "q2_ritme.tests"

    def setUp(self):
        super().setUp()
        self.train_val = pd.DataFrame(
            {"F1": [1, 2, 3], "F2": [4, 5, 6], "F3": [7, 8, 9]}
        )
        self.tax = pd.DataFrame()

    def test_get_search_space(self):
        search_space = ss.get_search_space(self.train_val, self.tax)

        self.assertIsInstance(search_space, dict)
        self.assertIn("xgb", search_space)
        self.assertIn("nn_reg", search_space)
        self.assertIn("nn_class", search_space)
        self.assertIn("nn_corn", search_space)
        self.assertIn("linreg", search_space)
        self.assertIn("rf", search_space)

    def test_get_data_eng_space_w_tax(self):
        tax = pd.DataFrame({"Taxon": ["Bacteria", "Firmicutes", "Clostridia"]})
        data_eng_space = ss.get_data_eng_space(self.train_val, tax)

        self.assertIsInstance(data_eng_space, dict)
        self.assertEqual(
            data_eng_space["data_aggregation"],
            tune.grid_search(
                [None, "tax_class", "tax_order", "tax_family", "tax_genus"]
            ),
        )
        self.assertEqual(
            data_eng_space["data_selection"],
            tune.grid_search([None, "abundance_ith", "variance_ith"]),
        )
        self.assertEqual(
            data_eng_space["data_selection_i"].categories,
            [1, 3, 10],
        )
        self.assertEqual(
            data_eng_space["data_transform"],
            tune.grid_search([None, "clr", "ilr", "alr", "pa"]),
        )

    def test_get_data_eng_space_empty_tax(self):
        data_eng_space = ss.get_data_eng_space(self.train_val, self.tax)
        self.assertEqual(data_eng_space["data_aggregation"], None)

    def test_get_linreg_space(self):
        linreg_space = ss.get_linreg_space(self.train_val, self.tax)

        self.assertIsInstance(linreg_space, dict)
        self.assertEqual(linreg_space["model"], "linreg")
        self.assertIn("data_transform", linreg_space)
        self.assertIn("fit_intercept", linreg_space)
        self.assertIn("alpha", linreg_space)
        self.assertIn("l1_ratio", linreg_space)

    def test_get_trac_space(self):
        trac_space = ss.get_trac_space(self.train_val, self.tax)

        self.assertIsInstance(trac_space, dict)
        self.assertEqual(trac_space["model"], "trac")
        self.assertEqual(trac_space["data_transform"], None)
        self.assertIn("lambda", trac_space)

    def test_get_rf_space(self):
        rf_space = ss.get_rf_space(self.train_val, self.tax)

        self.assertIsInstance(rf_space, dict)
        self.assertEqual(rf_space["model"], "rf")
        self.assertIn("data_transform", rf_space)
        self.assertIn("n_estimators", rf_space)
        self.assertIn("max_depth", rf_space)
        self.assertIn("min_samples_split", rf_space)
        self.assertIn("min_samples_leaf", rf_space)
        self.assertIn("max_features", rf_space)
        self.assertIn("min_impurity_decrease", rf_space)
        self.assertIn("bootstrap", rf_space)

    def test_get_xgb_space(self):
        xgb_space = ss.get_xgb_space(self.train_val, self.tax)

        self.assertIsInstance(xgb_space, dict)
        self.assertEqual(xgb_space["model"], "xgb")
        self.assertIn("data_transform", xgb_space)
        self.assertIn("objective", xgb_space)
        self.assertIn("max_depth", xgb_space)
        self.assertIn("min_child_weight", xgb_space)
        self.assertIn("subsample", xgb_space)
        self.assertIn("eta", xgb_space)

    def test_get_nn_space(self):
        nn_space = ss.get_nn_space(self.train_val, self.tax, "nn_reg")

        self.assertIsInstance(nn_space, dict)
        self.assertEqual(nn_space["model"], "nn_reg")
        self.assertIn("data_transform", nn_space)
        self.assertIn("n_hidden_layers", nn_space)
        self.assertIn("learning_rate", nn_space)
        self.assertIn("batch_size", nn_space)
        self.assertIn("epochs", nn_space)
