import os
import pickle
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from ray.air.result import Result

from ritme.evaluate_models import (
    _MODEL_SIMPLICITY_KNOBS,
    TunedModel,
    _get_checkpoint_path,
    _select_best_with_one_se,
    _trial_simplicity_key,
    get_data_processing,
    get_predictions,
    load_best_model,
    load_experiment_config,
    load_nn_model,
    load_sklearn_model,
    load_trac_model,
    load_xgb_model,
    retrieve_n_init_best_models,
    save_best_models,
)
from ritme.model_space import static_searchspace as ss
from ritme.tune_models import _RecordingTrial


class TestEvaluateModels(unittest.TestCase):
    def setUp(self):
        self.result = MagicMock(spec=Result)
        self.result.checkpoint = MagicMock()
        self.result.checkpoint.to_directory.return_value = "/test/checkpoint_dir"

        self.result.metrics = {"model_path": "/fake/model/path"}

        self.result.path = "/fake/trial/dir"
        self.result.config = {
            "data_aggregation": "tax_class",
            "data_transform": "ilr",
            "data_alr_denom_idx": 0,
        }

        self.result_grid = MagicMock()
        self.result_grid.get_best_result.return_value = self.result
        self.mock_taxonomy_df = pd.DataFrame({"species": ["A", "B"], "count": [10, 20]})
        self.mock_model = MagicMock()

        self.model_type = "xgb"
        self.path_to_exp = "/fake/experiment/dir"
        self.data = pd.DataFrame(
            {"feature1": [1, 2], "feature2": [3, 4], "target": [5, 6]}
        )

        self.tmodel = MagicMock(spec=TunedModel)
        self.tmodel.predict.return_value = [5.5, 6.5]

    def test_get_checkpoint_path(self):
        checkpoint_path = _get_checkpoint_path(self.result)
        self.assertEqual(checkpoint_path, "/test/checkpoint_dir/checkpoint")

    @patch("ritme.evaluate_models.load", return_value=MagicMock())
    def test_load_sklearn_model(self, mock_load):
        _ = load_sklearn_model(self.result)
        mock_load.assert_called_once()

    @patch(
        "ritme.evaluate_models.open",
        new_callable=unittest.mock.mock_open,
        read_data=pickle.dumps({"key": "value"}),
    )
    @patch("pickle.load", return_value={"key": "value"})
    def test_load_trac_model(self, mock_pickle_load, mock_open):
        _ = load_trac_model(self.result)
        mock_pickle_load.assert_called_once()

    @patch("xgboost.Booster", return_value=MagicMock())
    def test_load_xgb_model(self, mock_booster):
        _ = load_xgb_model(self.result)
        mock_booster.assert_called_once()

    @patch(
        "ritme.model_space.static_trainables.NeuralNet.load_from_checkpoint",
        return_value=MagicMock(),
    )
    def test_load_nn_model(self, mock_load_from_checkpoint):
        _ = load_nn_model(self.result)
        mock_load_from_checkpoint.assert_called_once()

    def test_get_data_processing(self):
        data_processing = get_data_processing(self.result)
        self.assertEqual(data_processing, self.result.config)

    @patch("ritme.evaluate_models.get_model", return_value=MagicMock())
    @patch("ritme.evaluate_models.get_data_processing")
    @patch("ritme.evaluate_models.get_taxonomy")
    @patch("ritme.evaluate_models.TunedModel.predict")
    def test_retrieve_n_init_best_models(
        self,
        mock_tuned_predict,
        mock_get_taxonomy,
        mock_get_data_processing,
        mock_get_model,
    ):
        mock_get_data_processing.return_value = {
            "data_aggregation": None,
            "data_transform": None,
            "data_selection": None,
            "data_alr_denom_idx": 0,
        }
        result_dic = {"xgb": self.result_grid, "nn_reg": self.result_grid}

        best_models = retrieve_n_init_best_models(result_dic, self.data)

        self.assertIsInstance(best_models, dict)
        self.assertEqual(len(best_models), 2)
        for model_type, tuned_model in best_models.items():
            self.assertIn(model_type, ["xgb", "nn_reg"])
            self.assertIsInstance(tuned_model, TunedModel)
        # assert that tuned model was called with predict for each model type
        self.assertEqual(mock_tuned_predict.call_count, 2)

    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    @patch("pickle.dump")
    def test_save_best_models(self, mock_pickle_dump, mock_open):
        best_model_dic = {"xgb": self.tmodel}
        save_best_models(best_model_dic, "/fake/output/dir")
        mock_pickle_dump.assert_called_once()

    @patch(
        "builtins.open",
        new_callable=unittest.mock.mock_open,
        read_data=b"fake_model_data",
    )
    @patch("pickle.load", return_value=MagicMock())
    def test_load_best_model(self, mock_pickle_load, mock_open):
        _ = load_best_model(self.model_type, self.path_to_exp)
        mock_pickle_load.assert_called_once()

    @patch(
        "builtins.open",
        new_callable=unittest.mock.mock_open,
        read_data='{"key": "value"}',
    )
    def test_load_experiment_config(self, mock_open):
        _ = load_experiment_config(self.path_to_exp)
        mock_open.assert_called_once()

    def test_get_predictions(self):
        predictions = get_predictions(self.data, self.tmodel, "target", "train")
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertIn("true", predictions.columns)
        self.assertIn("pred", predictions.columns)
        self.assertIn("split", predictions.columns)


class DummySklearnModel:
    """
    A minimal model that imitates an sklearn-like model.
    """

    def predict(self, X):
        return np.max(X, axis=1)


class ColumnCountModel:
    """Simple model returning the number of input columns per sample."""

    def predict(self, X):
        return np.full((X.shape[0],), X.shape[1])


class TestTunedModelImplementation(unittest.TestCase):
    # only testing functionality that was not already tested elsewhere
    def setUp(self):
        super().setUp()
        current_dir = os.path.dirname(__file__)

        # data
        self.data = pd.read_csv(
            os.path.join(current_dir, "data/example_feature_table.tsv"),
            sep="\t",
            index_col=0,
        )
        self.microb_raw_fts = [x for x in self.data.columns if x.startswith("F")]
        self.data_test = self.data.copy()
        self.data_test.index = [f"S{i}" for i in range(11, 21)]

        self.tax_df = pd.read_csv(
            os.path.join(current_dir, "data/example_taxonomy.tsv"),
            sep="\t",
            index_col=0,
        )

        # model
        self.data_config = {
            "data_aggregation": "tax_class",
            "data_transform": "alr",
            "data_selection": "abundance_threshold",
            "data_selection_t": 0.1,
            "data_alr_denom_idx_map": {"t0": 0},
            "data_enrich": "shannon_and_metadata",
            "data_enrich_with": ["md2"],
        }
        self.model = DummySklearnModel()
        # create the TunedModel instance
        self.tmodel = TunedModel(
            model=self.model,
            data_config=self.data_config,
            tax=self.tax_df,
            path="/some/fake/path",
        )

    @patch(
        "ritme.evaluate_models.aggregate_microbial_features",
        return_value=pd.DataFrame(),
    )
    def test_aggregate(self, mock_aggregate):
        _ = self.tmodel.aggregate(self.data)
        mock_aggregate.assert_called_once()

    @patch(
        "ritme.evaluate_models.transform_microbial_features",
        return_value=pd.DataFrame(),
    )
    def test_transform_non_alr_calls_once(self, mock_transform):
        # For non-ALR, transform should be applied directly once
        self.tmodel.data_config["data_transform"] = "clr"
        _ = self.tmodel.transform(self.data, time_label="t0")
        mock_transform.assert_called_once()

    def test_transform_alr_requires_map(self):
        # For ALR, time-suffixed input and denom map required
        self.tmodel.data_config["data_transform"] = "alr"
        # remove map to trigger error
        self.tmodel.data_config.pop("data_alr_denom_idx_map", None)
        df = pd.DataFrame({"F0": [1, 2], "F1": [3, 4]})
        with self.assertRaisesRegex(ValueError, r"ALR transform requires.*map"):
            self.tmodel.transform(df, time_label="t0")

    def test_enrich_train(self):
        other_ft_ls, _ = self.tmodel.enrich(
            self.data,
            self.microb_raw_fts,
            self.data[self.microb_raw_fts],
            self.data_config,
            time_label="t0",
            split="train",
        )
        self.assertEqual(other_ft_ls, ["shannon_entropy", "md2_b"])
        self.assertEqual(
            self.tmodel.snapshot_enriched_other_ft["t0"],
            ["shannon_entropy", "md2_b"],
        )

    def test_enrich_test_fails_if_no_train_run_before(self):
        fresh_tmodel = TunedModel(
            model=self.model,
            data_config=self.data_config,
            tax=self.tax_df,
            path="/some/fake/path",
        )
        with self.assertRaisesRegex(
            ValueError,
            r"Model must be fitted \(enrich\) on train split for this snapshot",
        ):
            fresh_tmodel.enrich(
                self.data,
                self.microb_raw_fts,
                self.data[self.microb_raw_fts],
                self.data_config,
                time_label="t0",
                split="test",
            )

    def test_enrich_test_w_train_run_before(self):
        # run on train
        _, _ = self.tmodel.enrich(
            self.data,
            self.microb_raw_fts,
            self.data[self.microb_raw_fts],
            self.data_config,
            time_label="t0",
            split="train",
        )
        # run on "test"
        enrich_test = self.data_test.copy()
        enrich_test["md2"] = 5 * ["c"] + 5 * ["d"]
        test_fts, df_test_enriched = self.tmodel.enrich(
            enrich_test,
            self.microb_raw_fts,
            enrich_test[self.microb_raw_fts],
            self.data_config,
            time_label="t0",
            split="test",
        )
        self.assertEqual(
            self.tmodel.snapshot_enriched_other_ft["t0"],
            ["shannon_entropy", "md2_b"],
        )
        self.assertEqual(test_fts, ["shannon_entropy", "md2_b"])
        self.assertEqual(df_test_enriched["md2_b"].values.tolist(), 10 * [0])

    @patch("ritme.evaluate_models.select_microbial_features")
    def test_select_train_snapshot(self, mock_select):
        mock_select.return_value = self.data[["F1"]]
        out = self.tmodel.select(
            self.data[self.microb_raw_fts], time_label="t0", split="train"
        )
        assert_frame_equal(out, self.data[["F1"]])
        self.assertIn("t0", self.tmodel.snapshot_selected_map)
        self.assertEqual(self.tmodel.snapshot_selected_map["t0"], ["F1"])

    def test_predict_requires_microbial_features(self):
        # No F-prefixed columns at all should raise
        data_no_micro = pd.DataFrame({"other": [1, 2]}, index=["S1", "S2"])
        with self.assertRaisesRegex(ValueError, r"No time labels detected"):
            _ = self.tmodel.predict(data_no_micro, split="train")

    def test_select_test_fails_if_no_train_run_before(self):
        fresh_tmodel = TunedModel(
            model=self.model,
            data_config=self.data_config,
            tax=self.tax_df,
            path="/some/fake/path",
        )
        with self.assertRaisesRegex(
            ValueError, r"Model must be fitted on train split for this snapshot"
        ):
            fresh_tmodel.select(
                self.data[self.microb_raw_fts], time_label="t0", split="test"
            )

    @patch("ritme.evaluate_models.select_microbial_features")
    def test_select_test_w_train_run_before(self, mock_select):
        # run on train
        mock_select.return_value = self.data[["F1"]]
        _ = self.tmodel.select(
            self.data[self.microb_raw_fts], time_label="t0", split="train"
        )
        # run on "test"
        df_test_selected = self.tmodel.select(
            self.data_test[self.microb_raw_fts], time_label="t0", split="test"
        )
        # Expect selected F1 plus grouped column over remaining microbial features
        other_cols = [c for c in self.microb_raw_fts if c != "F1"]
        expected = pd.DataFrame(index=self.data_test.index)
        expected["F1"] = self.data_test["F1"]
        expected["F_low_abun"] = self.data_test[other_cols].sum(axis=1)
        assert_frame_equal(df_test_selected, expected)

    @patch("ritme.evaluate_models.select_microbial_features")
    def test_select_test_missing_a_selected_feature(self, mock_select):
        # Train selects F1 and F2; test data is missing F2 entirely
        mock_select.return_value = self.data[["F1", "F2"]]
        _ = self.tmodel.select(
            self.data[self.microb_raw_fts], time_label="t0", split="train"
        )
        test_data = self.data_test[["F1", "F3", "F4"]].copy()
        with self.assertWarnsRegex(UserWarning, r"filled with 0.*F2"):
            df_test_selected = self.tmodel.select(
                test_data, time_label="t0", split="test"
            )
        # Missing F2 is filled with 0, present F1 retained, F3/F4 grouped
        self.assertIn("F1", df_test_selected.columns)
        self.assertIn("F2", df_test_selected.columns)
        self.assertIn("F_low_abun", df_test_selected.columns)
        self.assertTrue((df_test_selected["F2"] == 0).all())
        assert_array_equal(
            df_test_selected["F_low_abun"].values,
            test_data[["F3", "F4"]].sum(axis=1).values,
        )

    def test_predict_test_missing_train_features(self):
        # Train with F1-F6, predict on test that is missing F2 and F3
        _ = self.tmodel.predict(self.data, split="train")
        test_data = self.data.drop(columns=["F2", "F3"]).head(3).copy()
        test_data.index = [f"T{i}" for i in range(1, 4)]
        # Must not raise KeyError
        preds = self.tmodel.predict(test_data, split="test")
        self.assertEqual(len(preds), 3)

    def test_predict_sklearn_model_w_all_ft_engineering_options(self):
        # aggregate: F1, ..., F6 -> c__Bacilli, c__Clostridia
        # selected: c__Bacilli, c__Clostridia -> c__Bacilli, c__Clostridia
        # transformed: c__Bacilli, c__Clostridia -> alr_c__Clostridia
        # enriched: alr_c__Clostridia -> alr_c__Clostridia, shannon_entropy, md2_b

        # exp_x was calculated manually to verify that the correct x is used
        # within below class
        exp_x = pd.DataFrame(
            {
                "alr_c__Clostridia": [
                    7.018775e05,
                    1.000000e00,
                    2.482959e-07,
                    1.548812e06,
                    6.156383e01,
                    1.236978e-07,
                    1.000000e00,
                    1.838235e00,
                    3.514286e01,
                    1.000000e00,
                ],
                "shannon_entropy": [
                    0.005097,
                    -0.000000,
                    2.22098738e-02,
                    0.010021,
                    0.364255,
                    3.894847e-02,
                    -0.000000,
                    0.069242,
                    0.031841,
                    -0.000000,
                ],
            },
            index=[f"S{i}" for i in range(1, 11)],
        )
        exp_x["md2_b"] = 3 * [0.0] + 4 * [1.0] + 3 * [0.0]
        exp_pred = np.max(exp_x.values, axis=1).flatten()
        obs_pred = self.tmodel.predict(self.data, split="train")
        assert_allclose(obs_pred, exp_pred, rtol=1e-6)

    def test_predict_sklearn_model_multi_snapshot_shannon_alr(self):
        # Build simple two-snapshot dataset: t0 columns unsuffixed, t-1 suffixed
        df = pd.DataFrame(
            {
                "host": ["a", "b", "c", "d"],
                "host__t-1": ["a", "b", "c", "d"],
                "target": [0, 1, 0, 1],
                "F0": [1, 2, 3, 4],
                "F1": [1, 1, 1, 1],
                "F0__t-1": [2, 3, 4, 5],
                "F1__t-1": [1, 1, 1, 1],
            },
            index=["S1", "S2", "S3", "S4"],
        )
        cfg = {
            "data_aggregation": None,
            "data_transform": "alr",
            "data_alr_denom_idx_map": {"t0": 1, "t-1": 1},
            "data_selection": None,
            "data_enrich": "shannon",
            "data_enrich_with": None,
        }
        model = ColumnCountModel()
        tuned = TunedModel(model=model, data_config=cfg, tax=self.tax_df, path="")
        preds = tuned.predict(df, split="train")
        # Expect: per snapshot ALR on 2 features -> 1 col each
        # plus shannon per snapshot -> 1 each (total 4 columns)
        # total columns = 4
        assert_array_equal(preds, np.array([4, 4, 4, 4]))
        # t0 features are unsuffixed, t-1 features are suffixed
        self.assertTrue(all(not s.endswith("__t0") for s in tuned.final_feature_cols))
        self.assertTrue(any(s.endswith("__t-1") for s in tuned.final_feature_cols))

    @patch("ritme.evaluate_models._preprocess_taxonomy_aggregation")
    @patch.object(TunedModel, "enrich")
    @patch.object(TunedModel, "transform")
    @patch.object(TunedModel, "select")
    @patch.object(TunedModel, "aggregate")
    def test_predict_trac_model(
        self, mock_agg, mock_sel, mock_trans, mock_enrich, mock_preproc
    ):
        # Test that TRAC path uses _preprocess_taxonomy_aggregation correctly
        base = pd.DataFrame({"FSp1": [1, 2], "FSp2": [3, 4]}, index=["S1", "S2"])
        mock_agg.return_value = base
        mock_sel.return_value = base
        mock_trans.return_value = base
        # enriched snapshot equals transformed (t0 columns are unsuffixed)
        enriched = pd.DataFrame({"FSp1": [1, 2], "FSp2": [3, 4]}, index=["S1", "S2"])
        mock_enrich.return_value = ([], enriched)
        mock_preproc.return_value = (np.array([[2], [3]]), None)

        trac_model = {
            "matrix_a": pd.DataFrame(),
            "model": pd.DataFrame({"alpha": [2, 3]}, index=["FSp1", "FSp2"]),
        }
        tuned = TunedModel(
            model=trac_model,
            data_config={"data_transform": None, "data_aggregation": None},
            tax=pd.DataFrame(),
            path="",
        )

        preds = tuned.predict(base, split="train")
        # alpha = [2,3]
        # log_geom = [[2],[3]]
        # predicted = log_geom.dot(alpha[1:]) + alpha[0]
        # => [ (2*3)+2, (3*3)+2 ] => [8,11]
        assert_array_equal(preds, np.array([[8], [11]]))

    def test_predict_multi_snapshot_with_nan_rows(self):
        """NaN rows in past snapshot should flow through without errors.

        This simulates missing_mode='nan' where some past observations are NaN.
        The model should receive NaN for those features and still produce output.
        """
        df = pd.DataFrame(
            {
                "host": ["a", "b", "c", "d"],
                "host__t-1": ["a", "b", "c", "d"],
                "target": [0, 1, 0, 1],
                "F0": [0.5, 0.6, 0.7, 0.8],
                "F1": [0.5, 0.4, 0.3, 0.2],
                # First and third rows have NaN for past snapshot
                "F0__t-1": [np.nan, 0.5, np.nan, 0.7],
                "F1__t-1": [np.nan, 0.5, np.nan, 0.3],
            },
            index=["S1", "S2", "S3", "S4"],
        )
        cfg = {
            "data_aggregation": None,
            "data_transform": None,
            "data_selection": None,
            "data_enrich": None,
            "data_enrich_with": None,
        }
        model = ColumnCountModel()
        tuned = TunedModel(model=model, data_config=cfg, tax=self.tax_df, path="")
        preds = tuned.predict(df, split="train")
        # ColumnCountModel returns number of columns
        # t0 has 2 features (unsuffixed), t-1 has 2 features (suffixed) = 4 total
        self.assertEqual(len(preds), 4)
        # NaN rows produce NaN in features but the model still sees 4 columns
        assert_array_equal(preds, np.array([4, 4, 4, 4]))
        # Verify NaN was preserved in the feature matrix (check final_feature_cols)
        self.assertEqual(len(tuned.final_feature_cols), 4)


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

    def test_returns_dataframe_with_correct_shape(self):
        X = self.tmodel.build_design_matrix(self.data, split="train")
        self.assertIsInstance(X, pd.DataFrame)
        self.assertEqual(X.shape[0], self.data.shape[0])
        self.assertGreater(X.shape[1], 0)

    def test_output_matches_predict_input(self):
        X = self.tmodel.build_design_matrix(self.data, split="train")
        preds = self.tmodel.predict(self.data, split="train")
        expected = np.max(X.values, axis=1).flatten()
        assert_allclose(preds, expected, rtol=1e-6)

    def test_train_test_column_consistency(self):
        data_test = self.data.copy()
        data_test.index = [f"T{i}" for i in range(1, 11)]
        X_train = self.tmodel.build_design_matrix(self.data, split="train")
        X_test = self.tmodel.build_design_matrix(data_test, split="test")
        self.assertEqual(list(X_train.columns), list(X_test.columns))

    def test_multi_snapshot(self):
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
        self.assertEqual(X.shape[1], 4)
        suffixed = [c for c in X.columns if "__t-1" in c]
        self.assertEqual(len(suffixed), 2)

    def test_sets_final_feature_cols_on_train(self):
        self.assertEqual(self.tmodel.final_feature_cols, [])
        self.tmodel.build_design_matrix(self.data, split="train")
        self.assertGreater(len(self.tmodel.final_feature_cols), 0)

    def test_with_full_feature_engineering(self):
        cfg = {
            "data_aggregation": "tax_class",
            "data_transform": "alr",
            "data_selection": "abundance_threshold",
            "data_selection_t": 0.1,
            "data_alr_denom_idx_map": {"t0": 0},
            "data_enrich": "shannon_and_metadata",
            "data_enrich_with": ["md2"],
        }
        tmodel = TunedModel(
            model=self.model,
            data_config=cfg,
            tax=self.tax_df,
            path="",
        )
        X = tmodel.build_design_matrix(self.data, split="train")
        preds = tmodel.predict(self.data, split="train")
        expected = np.max(X.values, axis=1).flatten()
        assert_allclose(preds, expected, rtol=1e-6)


def _make_mock_result(metrics: dict, config: dict):
    """Build a MagicMock standing in for a Ray Tune ``Result`` row."""
    r = MagicMock(spec=Result)
    r.metrics = metrics
    r.config = config
    return r


class TestOneStandardErrorRule(unittest.TestCase):
    """Picking the simplest configuration whose K-fold mean is within one
    standard error of the best mean (Hastie/Tibshirani/Friedman 2009 §7.10;
    glmnet's ``lambda.1se``)."""

    def test_falls_back_to_get_best_when_no_se_present(self):
        # Single-split runs do not report _se fields. The selector must
        # defer to Ray Tune's normal best-result lookup.
        rg = MagicMock()
        sentinel = MagicMock()
        rg.get_best_result.return_value = sentinel
        rg.__iter__ = lambda self: iter(
            [
                _make_mock_result({"rmse_val": 0.5, "nb_features": 10}, {"alpha": 1.0}),
                _make_mock_result(
                    {"rmse_val": 0.4, "nb_features": 50}, {"alpha": 0.01}
                ),
            ]
        )
        chosen = _select_best_with_one_se(rg, "rmse_val", "min", "linreg")
        rg.get_best_result.assert_called_once_with(scope="all")
        self.assertIs(chosen, sentinel)

    def test_picks_simplest_within_band_for_min_metric(self):
        # Three configs: A best mean but high SE; B and C both inside the band.
        # Within the band, the simplest (highest alpha for linreg) should win.
        a = _make_mock_result(
            {"rmse_val_mean": 0.45, "rmse_val_se": 0.05, "nb_features": 200},
            {"alpha": 0.001, "l1_ratio": 0.5},
        )
        b = _make_mock_result(
            {"rmse_val_mean": 0.47, "rmse_val_se": 0.02, "nb_features": 180},
            {"alpha": 0.01, "l1_ratio": 0.5},
        )
        c = _make_mock_result(
            {"rmse_val_mean": 0.49, "rmse_val_se": 0.01, "nb_features": 60},
            {"alpha": 1.0, "l1_ratio": 0.5},
        )
        rg = MagicMock()
        rg.__iter__ = lambda self: iter([a, b, c])
        rg.get_best_result.side_effect = AssertionError("must not fall through")
        chosen = _select_best_with_one_se(rg, "rmse_val", "min", "linreg")
        self.assertIs(chosen, c)  # highest alpha + fewest features

    def test_picks_best_when_band_is_just_the_winner(self):
        # B is far worse than A (gap >> SE). Selector must keep A.
        a = _make_mock_result(
            {"rmse_val_mean": 0.40, "rmse_val_se": 0.001, "nb_features": 100},
            {"alpha": 0.01},
        )
        b = _make_mock_result(
            {"rmse_val_mean": 0.50, "rmse_val_se": 0.001, "nb_features": 50},
            {"alpha": 1.0},
        )
        rg = MagicMock()
        rg.__iter__ = lambda self: iter([a, b])
        chosen = _select_best_with_one_se(rg, "rmse_val", "min", "linreg")
        self.assertIs(chosen, a)

    def test_max_metric_uses_correct_band_direction(self):
        # roc_auc maximised: best_mean - se to best_mean is the band.
        a = _make_mock_result(
            {
                "roc_auc_macro_ovr_val_mean": 0.85,
                "roc_auc_macro_ovr_val_se": 0.04,
                "nb_features": 200,
            },
            {"C": 100.0},
        )
        b = _make_mock_result(
            {
                "roc_auc_macro_ovr_val_mean": 0.82,
                "roc_auc_macro_ovr_val_se": 0.02,
                "nb_features": 50,
            },
            {"C": 0.01},  # smaller C = more regularised = simpler for logreg
        )
        rg = MagicMock()
        rg.__iter__ = lambda self: iter([a, b])
        chosen = _select_best_with_one_se(rg, "roc_auc_macro_ovr_val", "max", "logreg")
        self.assertIs(chosen, b)

    def test_simplicity_key_orders_by_nb_features_first(self):
        k_low = _trial_simplicity_key("linreg", {"nb_features": 10}, {"alpha": 0.001})
        k_high = _trial_simplicity_key("linreg", {"nb_features": 100}, {"alpha": 1.0})
        # nb_features 10 should beat nb_features 100 even when alpha is smaller.
        self.assertLess(k_low, k_high)

    def test_one_se_rule_uses_only_kfold_trial_when_band_overlaps_single_split_one(
        self,
    ):
        """Mixed result grids -- some trials reported K-fold metrics (with
        ``_se``), others single-split (no ``_se``) -- must still enter the
        1-SE band path. The selector must NOT silently fall through to
        ``get_best_result`` just because some trials are missing ``_se``.
        """
        # K-fold trial: has _mean and _se; this is the global best by mean.
        kfold_best = _make_mock_result(
            {
                "rmse_val": 0.42,
                "rmse_val_mean": 0.42,
                "rmse_val_se": 0.10,  # wide band: 0.42 + 0.10 covers 0.50 below
                "nb_features": 200,
            },
            {"alpha": 0.001, "l1_ratio": 0.5},
        )
        # K-fold trial inside the band: simpler config (fewer features, higher
        # alpha) should win the simplicity tiebreak within the 1-SE band.
        kfold_simpler = _make_mock_result(
            {
                "rmse_val": 0.50,
                "rmse_val_mean": 0.50,
                "rmse_val_se": 0.02,
                "nb_features": 60,
            },
            {"alpha": 1.0, "l1_ratio": 0.5},
        )
        # Single-split trial in the same grid -- no _se reported. Its bare
        # ``rmse_val`` (0.55) is OUTSIDE the band (0.42 + 0.10 = 0.52), so it
        # is excluded from the band; the selector still treats it as a
        # candidate with se=0.0 but it isn't selected on simplicity grounds.
        single_split = _make_mock_result(
            {"rmse_val": 0.55, "nb_features": 150},
            {"alpha": 0.01, "l1_ratio": 0.5},
        )
        rg = MagicMock()
        rg.__iter__ = lambda self: iter([kfold_best, kfold_simpler, single_split])
        # If the selector silently fell through to get_best_result, this would
        # be called -- assert that does NOT happen when any trial has _se.
        rg.get_best_result.side_effect = AssertionError(
            "must not fall through to get_best_result when any trial has _se"
        )
        chosen = _select_best_with_one_se(rg, "rmse_val", "min", "linreg")
        # Best mean is kfold_best (0.42); band = best_se = 0.10.
        # kfold_simpler (0.50) is within the band (0.50 - 0.42 = 0.08 <= 0.10).
        # single_split (0.55) is outside (0.55 - 0.42 = 0.13 > 0.10).
        # Within the band {kfold_best, kfold_simpler}, the simpler config
        # (fewer features, higher alpha) wins.
        self.assertIs(chosen, kfold_simpler)

    def test_trial_simplicity_key_secondary_axis_breaks_tie_on_equal_nb_features(
        self,
    ):
        """When two configs have identical ``nb_features``, the secondary axis
        of ``_trial_simplicity_key`` (model-internal regularisation knobs)
        breaks the tie. For linreg, higher alpha -> stronger regularisation
        -> simpler -> sorts first.
        """
        k_strong = _trial_simplicity_key(
            "linreg", {"nb_features": 50}, {"alpha": 1.0, "l1_ratio": 0.5}
        )
        k_weak = _trial_simplicity_key(
            "linreg", {"nb_features": 50}, {"alpha": 0.001, "l1_ratio": 0.5}
        )
        # Higher alpha (stronger regularisation) sorts first under
        # "smaller-is-simpler" ordering.
        self.assertLess(k_strong, k_weak)
        # And the primary axis ties:
        self.assertEqual(k_strong[0], k_weak[0])

    def test_trial_with_nan_se_is_excluded_from_selection(self):
        """A K-fold trial whose primary metric had K-1 NaN folds produces
        ``_se = NaN`` (see ``_aggregate_fold_metrics``). Such a trial's mean
        is a single-fold point estimate dressed up as a K-fold result -- it
        must not be selectable as 'best' even when its mean is numerically
        the smallest. The selector picks among the trials with a finite SE.
        """
        # Mean=0.30 looks best, but SE=NaN means K-1 folds failed: unreliable.
        unreliable = _make_mock_result(
            {"rmse_val_mean": 0.30, "rmse_val_se": float("nan"), "nb_features": 100},
            {"alpha": 0.001, "l1_ratio": 0.5},
        )
        # Reliable K-fold winner with a real, finite SE.
        reliable_best = _make_mock_result(
            {"rmse_val_mean": 0.42, "rmse_val_se": 0.02, "nb_features": 80},
            {"alpha": 1.0, "l1_ratio": 0.5},
        )
        rg = MagicMock()
        rg.__iter__ = lambda self: iter([unreliable, reliable_best])
        rg.get_best_result.side_effect = AssertionError(
            "must not fall through when reliable K-fold trials exist"
        )
        chosen = _select_best_with_one_se(rg, "rmse_val", "min", "linreg")
        self.assertIs(chosen, reliable_best)

    def test_falls_back_when_every_trial_has_nan_se(self):
        """If no trial has a finite SE (all K-fold trials had K-1 NaN folds,
        or the grid has only single-split trials), defer to Ray Tune's
        single-best lookup -- there is no reliable basis for the 1-SE rule.
        """
        a = _make_mock_result(
            {"rmse_val_mean": 0.30, "rmse_val_se": float("nan"), "nb_features": 100},
            {"alpha": 0.001},
        )
        b = _make_mock_result(
            {"rmse_val_mean": 0.40, "rmse_val_se": float("nan"), "nb_features": 80},
            {"alpha": 1.0},
        )
        rg = MagicMock()
        sentinel = MagicMock()
        rg.get_best_result.return_value = sentinel
        rg.__iter__ = lambda self: iter([a, b])
        chosen = _select_best_with_one_se(rg, "rmse_val", "min", "linreg")
        rg.get_best_result.assert_called_once_with(scope="all")
        self.assertIs(chosen, sentinel)

    def test_simplicity_knobs_match_actual_search_space(self):
        """Every entry in ``_MODEL_SIMPLICITY_KNOBS`` names a hyperparameter
        that the corresponding search space actually suggests. Without this
        check, a rename in ``static_searchspace.py`` (e.g. ``alpha`` to
        ``alpha_l2``) would silently disable the simplicity-tiebreak for the
        affected knob -- ``_trial_simplicity_key`` would return ``inf`` for
        the renamed knob and the 1-SE rule would degenerate to "best mean
        only" with no warning.
        """
        # Tiny dummy train_val with a couple of F-features so search-space
        # builders that probe column counts don't choke.
        train_val = pd.DataFrame(
            {
                "F0": np.linspace(0.0, 1.0, 8),
                "F1": np.linspace(0.0, 1.0, 8),
                "target": np.linspace(0.0, 1.0, 8),
                "host_id": list(range(8)),
            }
        )
        tax = pd.DataFrame([])
        for model_type, knobs in _MODEL_SIMPLICITY_KNOBS.items():
            recorder = _RecordingTrial()
            ss.get_search_space(
                recorder,
                model_type=model_type,
                train_val=train_val,
                tax=tax,
                model_hyperparameters={},
            )
            recorded = set(recorder.params.keys())
            for knob_name, _sign in knobs:
                self.assertIn(
                    knob_name,
                    recorded,
                    msg=(
                        f"_MODEL_SIMPLICITY_KNOBS[{model_type!r}] references "
                        f"unknown hyperparameter {knob_name!r}; update the "
                        f"table or the {model_type} search space."
                    ),
                )

    def test_trial_simplicity_key_missing_knob_sorts_last_for_negative_sign(self):
        """Missing/non-numeric knob values must sort LAST regardless of the
        knob's sign convention. For knobs with ``sign=-1`` (most entries in
        ``_MODEL_SIMPLICITY_KNOBS``: linreg.alpha, trac.lambda, rf.min_samples_leaf,
        xgb.gamma/reg_alpha/reg_lambda, nn.dropout_rate/weight_decay), the
        earlier ``sign * inf`` formulation produced ``-inf`` which sorted FIRST,
        silently rewarding malformed configs in the 1-SE band.
        """
        # Knob present vs knob missing for a negative-sign knob (linreg.alpha):
        k_present = _trial_simplicity_key(
            "linreg", {"nb_features": 50}, {"alpha": 1.0, "l1_ratio": 0.5}
        )
        k_missing_alpha = _trial_simplicity_key(
            "linreg", {"nb_features": 50}, {"l1_ratio": 0.5}
        )
        self.assertLess(k_present, k_missing_alpha)

        # Same property for a non-castable value (e.g. accidental string):
        k_garbage = _trial_simplicity_key(
            "linreg", {"nb_features": 50}, {"alpha": "n/a", "l1_ratio": 0.5}
        )
        self.assertLess(k_present, k_garbage)


if __name__ == "__main__":
    unittest.main()
