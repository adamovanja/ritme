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
    _extract_coefficients,
    _get_predict_fn,
    _is_coef_model,
    cli_explain_features,
    compute_feature_importance,
    compute_shap_values,
    plot_feature_importance_bar,
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


class TestPlotShapMulticlass(unittest.TestCase):
    """Multi-class (3-D values) plot helpers must label classes and not crash."""

    def _make_multiclass_explanation(self, n_classes=3, output_names=None):
        rng = np.random.RandomState(0)
        values = rng.randn(8, 4, n_classes)
        data = rng.randn(8, 4)
        explanation = shap.Explanation(
            values=values,
            base_values=np.zeros((8, n_classes)),
            data=data,
            feature_names=[f"feat_{i}" for i in range(4)],
        )
        if output_names is not None:
            explanation.output_names = output_names
        return explanation

    def test_summary_plot_multiclass_renders_one_subplot_per_class(self):
        explanation = self._make_multiclass_explanation(
            n_classes=3, output_names=["gut", "tongue", "skin"]
        )
        fig = plot_shap_summary(explanation, max_display=4, show=False)
        self.assertIsInstance(fig, plt.Figure)
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        self.assertIn("Class: gut", titles)
        self.assertIn("Class: tongue", titles)
        self.assertIn("Class: skin", titles)
        plt.close(fig)

    def test_summary_plot_multiclass_falls_back_to_indexed_names(self):
        explanation = self._make_multiclass_explanation(n_classes=3)
        # No output_names attached — helper must still render without raising.
        fig = plot_shap_summary(explanation, max_display=4, show=False)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_bar_plot_multiclass_renders_one_subplot_per_class(self):
        explanation = self._make_multiclass_explanation(
            n_classes=3, output_names=["a", "b", "c"]
        )
        fig = plot_shap_bar(explanation, max_display=4, show=False)
        self.assertIsInstance(fig, plt.Figure)
        # One title per class.
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        self.assertIn("Class: a", titles)
        self.assertIn("Class: b", titles)
        self.assertIn("Class: c", titles)
        plt.close(fig)


class TestClassificationClassNames(unittest.TestCase):
    """``_classification_class_names`` must recover original-target labels."""

    def test_returns_none_for_regression_sklearn_model(self):
        from ritme.explain_features import _classification_class_names

        tmodel = MagicMock(spec=TunedModel)
        tmodel.model = DummySklearnModel()  # has no classes_ attribute
        tmodel.label_encoder = None
        self.assertIsNone(_classification_class_names(tmodel))

    def test_recovers_string_labels_via_label_encoder(self):
        from sklearn.preprocessing import LabelEncoder

        from ritme.explain_features import _classification_class_names

        le = LabelEncoder().fit(["gut", "tongue", "skin"])

        class _SklearnClassifier:
            classes_ = np.array([0, 1, 2])

        tmodel = MagicMock(spec=TunedModel)
        tmodel.model = _SklearnClassifier()
        tmodel.label_encoder = le
        self.assertEqual(
            sorted(_classification_class_names(tmodel)),
            sorted([str(c) for c in le.classes_]),
        )


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

    @patch("ritme.explain_features.load_best_model")
    @patch("pandas.read_pickle")
    def test_cli_trac_writes_coefficient_csv(
        self,
        mock_read_pickle,
        mock_load_best_model,
    ):
        """For TRAC (a coefficient-bearing model), the CLI must skip SHAP and
        emit ``feature_importance_<model_type>.csv`` instead. The real
        coefficient-extraction path is exercised end-to-end on a realistic
        TRAC bundle (no patching of ``compute_feature_importance``)."""
        train_val = pd.DataFrame({"F1": [1, 2]})
        test = pd.DataFrame({"F1": [3, 4]})
        mock_read_pickle.side_effect = [train_val, test]

        mock_tmodel = MagicMock(spec=TunedModel)
        a_df = pd.DataFrame({"n1": [1, 0], "n2": [0, 1]}, index=["F1", "F2"])
        alpha_df = pd.DataFrame(
            {"alpha": [0.1, 0.5, -0.3]}, index=["intercept", "n1", "n2"]
        )
        mock_tmodel.model = {"model": alpha_df, "matrix_a": a_df}
        mock_load_best_model.return_value = mock_tmodel

        with tempfile.TemporaryDirectory() as path_to_exp:
            cli_explain_features(path_to_exp, "trac", "train.pkl", "test.pkl")

            csv_path = os.path.join(path_to_exp, "feature_importance_trac.csv")
            bar_path = os.path.join(path_to_exp, "feature_importance_bar_trac.png")
            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(bar_path))
            # No SHAP outputs should be created for a coefficient-bearing model.
            for shap_name in (
                "shap_values_trac.pkl",
                "shap_summary_trac.png",
                "shap_bar_trac.png",
            ):
                self.assertFalse(os.path.exists(os.path.join(path_to_exp, shap_name)))

            written = pd.read_csv(csv_path)
            self.assertEqual(
                sorted(written.columns),
                sorted(["feature", "coefficient", "abs_coefficient"]),
            )
            # End-to-end: the real ``_extract_coefficients`` path drops
            # ``intercept`` and surfaces only the log-contrast nodes.
            self.assertEqual(sorted(written["feature"].tolist()), ["n1", "n2"])
            self.assertNotIn("intercept", written["feature"].values)
            assert_allclose(
                sorted(written["coefficient"].tolist()), sorted([0.5, -0.3])
            )
        plt.close("all")


class TestIsCoefModel(unittest.TestCase):
    """``_is_coef_model`` must distinguish coefficient-bearing trainables
    (linreg / logreg / trac) from SHAP-only trainables (rf / xgb / nn_*)."""

    def test_trac_dict_is_coef_model(self):
        model = {"matrix_a": pd.DataFrame(), "model": pd.DataFrame()}
        self.assertTrue(_is_coef_model(model))

    def test_linreg_pipeline_is_coef_model(self):
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
        y = np.array([0.0, 1.0, 1.0, 0.0])
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("linreg", ElasticNet(alpha=0.1))]
        ).fit(X, y)
        self.assertTrue(_is_coef_model(pipeline))

    def test_logreg_pipeline_is_coef_model(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
        y = np.array([0, 1, 1, 0])
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("logreg", LogisticRegression())]
        ).fit(X, y)
        self.assertTrue(_is_coef_model(pipeline))

    def test_random_forest_is_not_coef_model(self):
        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(n_estimators=2).fit(
            np.array([[0.0], [1.0]]), np.array([0.0, 1.0])
        )
        self.assertFalse(_is_coef_model(rf))

    def test_xgb_booster_is_not_coef_model(self):
        dtrain = xgb.DMatrix(np.array([[0.0], [1.0]]), label=np.array([0.0, 1.0]))
        booster = xgb.train({"max_depth": 1, "verbosity": 0}, dtrain, num_boost_round=1)
        self.assertFalse(_is_coef_model(booster))

    def test_partial_trac_dict_is_not_coef_model(self):
        """A dict missing one of ``matrix_a`` or ``model`` is not a TRAC bundle
        and must fall through to ``False`` rather than being mis-routed."""
        self.assertFalse(_is_coef_model({"model": pd.DataFrame()}))
        self.assertFalse(_is_coef_model({"matrix_a": pd.DataFrame()}))
        self.assertFalse(_is_coef_model({}))


class TestExtractCoefficients(unittest.TestCase):
    """``_extract_coefficients`` produces a per-feature long-form DataFrame."""

    def test_linreg_pipeline_returns_one_row_per_feature(self):
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(0)
        X = rng.randn(20, 3)
        y = X[:, 0] - 2 * X[:, 1]
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("linreg", ElasticNet(alpha=0.01))]
        ).fit(X, y)

        tmodel = MagicMock(spec=TunedModel)
        tmodel.model = pipeline
        tmodel.label_encoder = None
        df = _extract_coefficients(tmodel, ["F1", "F2", "F3"])

        self.assertEqual(list(df["feature"]), ["F1", "F2", "F3"])
        self.assertEqual(
            sorted(df.columns),
            sorted(["feature", "coefficient", "abs_coefficient"]),
        )
        assert_allclose(df["abs_coefficient"], np.abs(df["coefficient"]))

    def test_logreg_binary_pipeline_returns_one_row_per_feature(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(1)
        X = rng.randn(30, 3)
        y = (X[:, 0] > 0).astype(int)
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("logreg", LogisticRegression())]
        ).fit(X, y)

        tmodel = MagicMock(spec=TunedModel)
        tmodel.model = pipeline
        tmodel.label_encoder = None
        df = _extract_coefficients(tmodel, ["F1", "F2", "F3"])

        # Binary logreg has a single row of coefficients — long form should
        # have exactly one row per feature, no `class` column.
        self.assertEqual(len(df), 3)
        self.assertNotIn("class", df.columns)
        self.assertEqual(list(df["feature"]), ["F1", "F2", "F3"])

    def test_logreg_multiclass_pipeline_returns_per_class_rows(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(2)
        X = rng.randn(60, 3)
        y = rng.randint(0, 3, size=60)
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=500)),
            ]
        ).fit(X, y)

        tmodel = MagicMock(spec=TunedModel)
        tmodel.model = pipeline
        tmodel.label_encoder = None
        df = _extract_coefficients(tmodel, ["F1", "F2", "F3"])

        self.assertEqual(len(df), 3 * 3)
        self.assertIn("class", df.columns)
        self.assertEqual(sorted(df["feature"].unique()), ["F1", "F2", "F3"])

    def test_raises_when_coef_length_mismatches_feature_names(self):
        """Mismatched coefficient/feature-name lengths must raise — this guard
        is what catches future drift between ``build_design_matrix`` and the
        fitted estimator."""

        class _FakeEstimator:
            coef_ = np.array([0.1, 0.2])

        tmodel = MagicMock(spec=TunedModel)
        tmodel.model = _FakeEstimator()
        tmodel.label_encoder = None
        with self.assertRaisesRegex(ValueError, "Number of coefficients"):
            _extract_coefficients(tmodel, ["F1"])

    def test_raises_when_multiclass_coef_columns_mismatch(self):
        class _FakeMulticlassEstimator:
            coef_ = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            classes_ = np.array([0, 1])

        tmodel = MagicMock(spec=TunedModel)
        tmodel.model = _FakeMulticlassEstimator()
        tmodel.label_encoder = None
        with self.assertRaisesRegex(ValueError, "Number of coefficient columns"):
            _extract_coefficients(tmodel, ["F1"])

    def test_trac_dict_drops_intercept_and_uses_matrix_a_labels(self):
        a_df = pd.DataFrame({"node_1": [1, 0], "node_2": [0, 1]}, index=["F1", "F2"])
        alpha_df = pd.DataFrame(
            {"alpha": [0.1, 0.7, -0.4]},
            index=["intercept", "node_1", "node_2"],
        )
        tmodel = MagicMock(spec=TunedModel)
        tmodel.model = {"model": alpha_df, "matrix_a": a_df}
        df = _extract_coefficients(tmodel, ["F1", "F2"])

        self.assertEqual(list(df["feature"]), ["node_1", "node_2"])
        self.assertNotIn("intercept", df["feature"].values)
        assert_allclose(df["coefficient"], [0.7, -0.4])
        assert_allclose(df["abs_coefficient"], [0.7, 0.4])


class TestComputeFeatureImportance(unittest.TestCase):
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

    def _make_tmodel_with_linreg(self):
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("linreg", ElasticNet(alpha=0.01))]
        )
        tmodel = TunedModel(
            model=pipeline,
            data_config=self.data_config,
            tax=self.tax_df,
            path="",
            model_type="linreg",
        )
        X = tmodel.build_design_matrix(self.data, split="train")
        y = np.arange(X.shape[0], dtype=float)
        tmodel.model.fit(X.values, y)
        return tmodel, X

    def test_compute_feature_importance_linreg(self):
        tmodel, X = self._make_tmodel_with_linreg()
        df = compute_feature_importance(tmodel, self.data)
        self.assertEqual(len(df), X.shape[1])
        self.assertEqual(sorted(df["feature"]), sorted(X.columns.tolist()))

    def test_compute_feature_importance_rejects_rf(self):
        from sklearn.ensemble import RandomForestRegressor

        tmodel = TunedModel(
            model=RandomForestRegressor(n_estimators=2),
            data_config=self.data_config,
            tax=self.tax_df,
            path="",
            model_type="rf",
        )
        X = tmodel.build_design_matrix(self.data, split="train")
        tmodel.model.fit(X.values, np.arange(X.shape[0], dtype=float))
        with self.assertRaisesRegex(TypeError, "coefficient"):
            compute_feature_importance(tmodel, self.data)


class TestPlotFeatureImportanceBar(unittest.TestCase):
    def test_single_class_returns_figure(self):
        df = pd.DataFrame(
            {
                "feature": ["F1", "F2", "F3"],
                "coefficient": [0.5, -0.3, 0.1],
                "abs_coefficient": [0.5, 0.3, 0.1],
            }
        )
        fig = plot_feature_importance_bar(df, show=False)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_multiclass_renders_one_subplot_per_class(self):
        df = pd.DataFrame(
            {
                "feature": ["F1", "F2"] * 3,
                "class": ["a", "a", "b", "b", "c", "c"],
                "coefficient": [0.5, -0.3, 0.1, 0.2, -0.7, 0.4],
                "abs_coefficient": [0.5, 0.3, 0.1, 0.2, 0.7, 0.4],
            }
        )
        fig = plot_feature_importance_bar(df, show=False)
        self.assertIsInstance(fig, plt.Figure)
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        self.assertIn("Class: a", titles)
        self.assertIn("Class: b", titles)
        self.assertIn("Class: c", titles)
        plt.close(fig)


class TestCliExplainFeaturesCoefDispatch(unittest.TestCase):
    """The CLI must dispatch to the coefficient path for linreg/logreg/trac
    and write ``feature_importance_<model_type>.csv`` instead of SHAP files."""

    @patch("ritme.explain_features.compute_shap_values")
    @patch("ritme.explain_features.load_best_model")
    @patch("pandas.read_pickle")
    def test_cli_linreg_skips_shap_and_writes_csv(
        self,
        mock_read_pickle,
        mock_load_best_model,
        mock_compute_shap_values,
    ):
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        current_dir = os.path.dirname(__file__)
        data = pd.read_csv(
            os.path.join(current_dir, "data/example_feature_table.tsv"),
            sep="\t",
            index_col=0,
        )
        tax_df = pd.read_csv(
            os.path.join(current_dir, "data/example_taxonomy.tsv"),
            sep="\t",
            index_col=0,
        )
        mock_read_pickle.side_effect = [data, data]

        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("linreg", ElasticNet(alpha=0.01))]
        )
        tmodel = TunedModel(
            model=pipeline,
            data_config={
                "data_aggregation": None,
                "data_transform": None,
                "data_selection": None,
                "data_enrich": None,
                "data_enrich_with": None,
            },
            tax=tax_df,
            path="",
            model_type="linreg",
        )
        X = tmodel.build_design_matrix(data, split="train")
        tmodel.model.fit(X.values, np.arange(X.shape[0], dtype=float))
        mock_load_best_model.return_value = tmodel

        with tempfile.TemporaryDirectory() as path_to_exp:
            cli_explain_features(path_to_exp, "linreg", "train.pkl", "test.pkl")
            csv_path = os.path.join(path_to_exp, "feature_importance_linreg.csv")
            bar_path = os.path.join(path_to_exp, "feature_importance_bar_linreg.png")
            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(bar_path))
            for shap_name in (
                "shap_values_linreg.pkl",
                "shap_summary_linreg.png",
                "shap_bar_linreg.png",
            ):
                self.assertFalse(os.path.exists(os.path.join(path_to_exp, shap_name)))

        mock_compute_shap_values.assert_not_called()
        plt.close("all")

    @patch("ritme.explain_features.compute_shap_values")
    @patch("ritme.explain_features.load_best_model")
    @patch("pandas.read_pickle")
    def test_cli_logreg_skips_shap_and_writes_csv(
        self,
        mock_read_pickle,
        mock_load_best_model,
        mock_compute_shap_values,
    ):
        """End-to-end CLI dispatch for binary logreg: confirms the coefficient
        path writes a CSV (no ``class`` column for binary) and never reaches
        SHAP."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        current_dir = os.path.dirname(__file__)
        data = pd.read_csv(
            os.path.join(current_dir, "data/example_feature_table.tsv"),
            sep="\t",
            index_col=0,
        )
        tax_df = pd.read_csv(
            os.path.join(current_dir, "data/example_taxonomy.tsv"),
            sep="\t",
            index_col=0,
        )
        mock_read_pickle.side_effect = [data, data]

        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("logreg", LogisticRegression())]
        )
        tmodel = TunedModel(
            model=pipeline,
            data_config={
                "data_aggregation": None,
                "data_transform": None,
                "data_selection": None,
                "data_enrich": None,
                "data_enrich_with": None,
            },
            tax=tax_df,
            path="",
            model_type="logreg",
        )
        X = tmodel.build_design_matrix(data, split="train")
        y = np.array([0, 1] * (X.shape[0] // 2) + [0] * (X.shape[0] % 2))
        tmodel.model.fit(X.values, y)
        mock_load_best_model.return_value = tmodel

        with tempfile.TemporaryDirectory() as path_to_exp:
            cli_explain_features(path_to_exp, "logreg", "train.pkl", "test.pkl")
            csv_path = os.path.join(path_to_exp, "feature_importance_logreg.csv")
            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(
                os.path.exists(
                    os.path.join(path_to_exp, "feature_importance_bar_logreg.png")
                )
            )
            written = pd.read_csv(csv_path)
            # Binary logreg collapses (1, F) coef to one row per feature.
            self.assertNotIn("class", written.columns)
            self.assertEqual(len(written), X.shape[1])

        mock_compute_shap_values.assert_not_called()
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
