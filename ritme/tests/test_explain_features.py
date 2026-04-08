import os
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from numpy.testing import assert_allclose

from ritme.evaluate_models import TunedModel
from ritme.explain_features import (
    _build_explainer,
    _get_predict_fn,
    cli_explain_features,
    compute_shap_values,
    plot_shap_bar,
    plot_shap_summary,
)

matplotlib.use("Agg")


class DummySklearnModel:
    def predict(self, X):
        return np.sum(X, axis=1)

    def fit(self, X, y):
        return self


class TestGetPredictFn(unittest.TestCase):
    def test_sklearn_model(self):
        model = DummySklearnModel()
        tmodel = MagicMock(spec=TunedModel)
        tmodel.model = model
        fn = _get_predict_fn(tmodel)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = fn(X)
        assert_allclose(result, [3.0, 7.0])

    def test_xgb_model(self):
        X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_train = np.array([1.0, 2.0, 3.0])
        dtrain = xgb.DMatrix(X_train, label=y_train)
        booster = xgb.train({"max_depth": 1, "verbosity": 0}, dtrain, num_boost_round=2)

        tmodel = MagicMock(spec=TunedModel)
        tmodel.model = booster
        fn = _get_predict_fn(tmodel)
        result = fn(X_train)
        self.assertEqual(result.shape, (3,))


class TestBuildExplainer(unittest.TestCase):
    def test_trac_model_raises(self):
        tmodel = MagicMock(spec=TunedModel)
        tmodel.model = {"matrix_a": pd.DataFrame(), "model": pd.DataFrame()}
        X_bg = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with self.assertRaisesRegex(TypeError, "TRAC"):
            _build_explainer(tmodel, X_bg)

    def test_xgb_model_returns_tree_explainer(self):
        X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_train = np.array([1.0, 2.0, 3.0])
        dtrain = xgb.DMatrix(X_train, label=y_train)
        booster = xgb.train({"max_depth": 1, "verbosity": 0}, dtrain, num_boost_round=2)

        tmodel = MagicMock(spec=TunedModel)
        tmodel.model = booster
        X_bg = pd.DataFrame(X_train, columns=["a", "b"])
        explainer = _build_explainer(tmodel, X_bg)
        self.assertIsInstance(explainer, shap.TreeExplainer)

    def test_sklearn_model_returns_explainer(self):
        tmodel = MagicMock(spec=TunedModel)
        tmodel.model = DummySklearnModel()
        X_bg = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        explainer = _build_explainer(tmodel, X_bg)
        self.assertIsInstance(explainer, shap.Explainer)


class TestBuildDesignMatrix(unittest.TestCase):
    def setUp(self):
        current_dir = os.path.dirname(__file__)
        self.data = pd.read_csv(
            os.path.join(current_dir, "data/example_feature_table.tsv"),
            sep="\t",
            index_col=0,
        )
        self.tax_df = pd.read_csv(
            os.path.join(current_dir, "data/example_taxonomy.tsv"),
            sep="\t",
            index_col=0,
        )
        self.data_config = {
            "data_aggregation": None,
            "data_transform": None,
            "data_selection": None,
            "data_enrich": None,
            "data_enrich_with": None,
        }
        self.model = DummySklearnModel()
        self.tmodel = TunedModel(
            model=self.model,
            data_config=self.data_config,
            tax=self.tax_df,
            path="/fake/path",
        )

    def test_design_matrix_shape(self):
        X = self.tmodel.build_design_matrix(self.data, split="train")
        self.assertEqual(X.shape[0], self.data.shape[0])
        self.assertGreater(X.shape[1], 0)

    def test_design_matrix_matches_predict_input(self):
        X = self.tmodel.build_design_matrix(self.data, split="train")
        preds = self.tmodel.predict(self.data, split="train")
        expected = np.sum(X.values, axis=1)
        assert_allclose(preds, expected)

    def test_design_matrix_train_test_consistency(self):
        data_test = self.data.copy()
        data_test.index = [f"T{i}" for i in range(1, 11)]
        X_train = self.tmodel.build_design_matrix(self.data, split="train")
        X_test = self.tmodel.build_design_matrix(data_test, split="test")
        self.assertEqual(list(X_train.columns), list(X_test.columns))

    def test_design_matrix_multi_snapshot(self):
        df = pd.DataFrame(
            {
                "F0": [1.0, 2.0, 3.0],
                "F1": [1.0, 1.0, 1.0],
                "F0__t-1": [0.5, 1.5, 2.5],
                "F1__t-1": [1.0, 1.0, 1.0],
                "host": ["a", "b", "c"],
                "host__t-1": ["a", "b", "c"],
            },
            index=["S1", "S2", "S3"],
        )
        tmodel = TunedModel(
            model=self.model,
            data_config=self.data_config,
            tax=self.tax_df,
            path="",
        )
        X = tmodel.build_design_matrix(df, split="train")
        self.assertEqual(X.shape[0], 3)
        # t0: F0, F1; t-1: F0__t-1, F1__t-1
        self.assertEqual(X.shape[1], 4)
        suffixed = [c for c in X.columns if "__t-1" in c]
        self.assertEqual(len(suffixed), 2)


class TestComputeShapValues(unittest.TestCase):
    def setUp(self):
        current_dir = os.path.dirname(__file__)
        self.data = pd.read_csv(
            os.path.join(current_dir, "data/example_feature_table.tsv"),
            sep="\t",
            index_col=0,
        )
        self.tax_df = pd.read_csv(
            os.path.join(current_dir, "data/example_taxonomy.tsv"),
            sep="\t",
            index_col=0,
        )

    def _make_tmodel_with_rf(self):
        from sklearn.ensemble import RandomForestRegressor

        data_config = {
            "data_aggregation": None,
            "data_transform": None,
            "data_selection": None,
            "data_enrich": None,
            "data_enrich_with": None,
        }
        tmodel = TunedModel(
            model=RandomForestRegressor(n_estimators=5, random_state=42),
            data_config=data_config,
            tax=self.tax_df,
            path="",
        )
        X = tmodel.build_design_matrix(self.data, split="train")
        y = np.arange(X.shape[0], dtype=float)
        tmodel.model.fit(X.values, y)
        return tmodel

    def test_compute_shap_values_rf(self):
        tmodel = self._make_tmodel_with_rf()
        train = self.data.iloc[:7]
        test = self.data.iloc[7:]
        explanation = compute_shap_values(tmodel, train, test)
        self.assertIsInstance(explanation, shap.Explanation)
        self.assertEqual(explanation.values.shape[0], test.shape[0])
        self.assertEqual(len(explanation.feature_names), len(tmodel.final_feature_cols))

    def test_compute_shap_values_xgb(self):
        data_config = {
            "data_aggregation": None,
            "data_transform": None,
            "data_selection": None,
            "data_enrich": None,
            "data_enrich_with": None,
        }
        tmodel = TunedModel(
            model=None, data_config=data_config, tax=self.tax_df, path=""
        )
        X_train = tmodel.build_design_matrix(self.data, split="train")
        y_train = np.arange(X_train.shape[0], dtype=float)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        booster = xgb.train({"max_depth": 2, "verbosity": 0}, dtrain, num_boost_round=5)
        tmodel.model = booster

        train = self.data.iloc[:7]
        test = self.data.iloc[7:]
        explanation = compute_shap_values(tmodel, train, test)
        self.assertIsInstance(explanation, shap.Explanation)
        self.assertEqual(explanation.values.shape[0], test.shape[0])

    def test_trac_model_raises(self):
        tmodel = MagicMock(spec=TunedModel)
        tmodel.model = {"matrix_a": pd.DataFrame(), "model": pd.DataFrame()}
        X_bg = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        tmodel.build_design_matrix.return_value = X_bg
        with self.assertRaises(TypeError):
            compute_shap_values(tmodel, X_bg, X_bg)

    def test_max_background_samples(self):
        tmodel = self._make_tmodel_with_rf()
        train = self.data.iloc[:7]
        test = self.data.iloc[7:]
        explanation = compute_shap_values(tmodel, train, test, max_background_samples=3)
        self.assertIsInstance(explanation, shap.Explanation)


class TestPlotShapSummary(unittest.TestCase):
    def _make_explanation(self):
        values = np.random.RandomState(42).randn(10, 3)
        data = np.random.RandomState(42).randn(10, 3)
        return shap.Explanation(
            values=values,
            base_values=np.zeros(10),
            data=data,
            feature_names=["feat_a", "feat_b", "feat_c"],
        )

    def test_plot_shap_summary_returns_figure(self):
        explanation = self._make_explanation()
        fig = plot_shap_summary(explanation, show=False)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_shap_summary_bar_type(self):
        explanation = self._make_explanation()
        fig = plot_shap_summary(explanation, plot_type="bar", show=False)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_shap_bar_returns_figure(self):
        explanation = self._make_explanation()
        fig = plot_shap_bar(explanation, show=False)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


class TestCliExplainFeatures(unittest.TestCase):
    @patch("ritme.explain_features.plot_shap_bar")
    @patch("ritme.explain_features.plot_shap_summary")
    @patch("ritme.explain_features.compute_shap_values")
    @patch("ritme.explain_features.load_best_model")
    @patch("pandas.read_pickle")
    def test_cli_saves_outputs(
        self,
        mock_read_pickle,
        mock_load_best_model,
        mock_compute_shap_values,
        mock_plot_summary,
        mock_plot_bar,
    ):
        train_val = pd.DataFrame({"F1": [1, 2, 3], "target": [4, 5, 6]})
        test = pd.DataFrame({"F1": [7, 8], "target": [9, 10]})
        mock_read_pickle.side_effect = [train_val, test]

        mock_tmodel = MagicMock(spec=TunedModel)
        mock_tmodel.model = DummySklearnModel()
        mock_load_best_model.return_value = mock_tmodel

        mock_explanation = shap.Explanation(
            values=np.array([[1.0, 2.0]]),
            base_values=np.array([0.0]),
            data=np.array([[3.0, 4.0]]),
            feature_names=["a", "b"],
        )
        mock_compute_shap_values.return_value = mock_explanation

        mock_plot_summary.return_value = plt.figure()
        mock_plot_bar.return_value = plt.figure()

        with tempfile.TemporaryDirectory() as path_to_exp:
            with patch("builtins.print") as mock_print:
                cli_explain_features(path_to_exp, "rf", "train.pkl", "test.pkl")

            expected_calls = [
                call(f"SHAP values saved in {path_to_exp}/shap_values_rf.pkl."),
                call(f"SHAP summary plot saved in {path_to_exp}/shap_summary_rf.png."),
                call(f"SHAP bar plot saved in {path_to_exp}/shap_bar_rf.png."),
            ]
            mock_print.assert_has_calls(expected_calls)

            self.assertTrue(
                os.path.exists(os.path.join(path_to_exp, "shap_values_rf.pkl"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(path_to_exp, "shap_summary_rf.png"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(path_to_exp, "shap_bar_rf.png"))
            )
        plt.close("all")

    @patch("ritme.explain_features.compute_shap_values")
    @patch("ritme.explain_features.load_best_model")
    @patch("pandas.read_pickle")
    def test_cli_trac_raises(
        self,
        mock_read_pickle,
        mock_load_best_model,
        mock_compute_shap_values,
    ):
        train_val = pd.DataFrame({"F1": [1, 2]})
        test = pd.DataFrame({"F1": [3, 4]})
        mock_read_pickle.side_effect = [train_val, test]

        mock_tmodel = MagicMock(spec=TunedModel)
        mock_tmodel.model = {"matrix_a": pd.DataFrame(), "model": pd.DataFrame()}
        mock_load_best_model.return_value = mock_tmodel

        mock_compute_shap_values.side_effect = TypeError("TRAC")

        with tempfile.TemporaryDirectory() as path_to_exp:
            with self.assertRaises(TypeError):
                cli_explain_features(path_to_exp, "trac", "train.pkl", "test.pkl")


if __name__ == "__main__":
    unittest.main()
