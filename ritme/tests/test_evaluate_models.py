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
    TunedModel,
    _get_checkpoint_path,
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
        df = pd.DataFrame({"F0__t0": [1, 2], "F1__t0": [3, 4]})
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

    def test_predict_requires_time_suffixes(self):
        # Unsuffixed microbial columns should raise
        with self.assertRaisesRegex(ValueError, r"No time labels detected"):
            _ = self.tmodel.predict(self.data, split="train")

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
        # Provide time-suffixed input (single snapshot t0)
        data_t0 = self.data.copy()
        data_t0.columns = [f"{c}__t0" for c in data_t0.columns]
        obs_pred = self.tmodel.predict(data_t0, split="train")
        assert_allclose(obs_pred, exp_pred, rtol=1e-6)

    def test_predict_sklearn_model_multi_snapshot_shannon_alr(self):
        # Build simple two-snapshot dataset with suffixed columns
        df = pd.DataFrame(
            {
                "host__t0": ["a", "b", "c", "d"],
                "host__t-1": ["a", "b", "c", "d"],
                "target__t0": [0, 1, 0, 1],
                "F0__t0": [1, 2, 3, 4],
                "F1__t0": [1, 1, 1, 1],
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
        # Feature names recorded
        self.assertTrue(
            all(
                s.endswith("__t0") or s.endswith("__t-1")
                for s in tuned.final_feature_cols
            )
        )

    @patch("ritme.evaluate_models._preprocess_taxonomy_aggregation")
    @patch.object(TunedModel, "enrich")
    @patch.object(TunedModel, "transform")
    @patch.object(TunedModel, "select")
    @patch.object(TunedModel, "aggregate")
    def test_predict_trac_model(
        self, mock_agg, mock_sel, mock_trans, mock_enrich, mock_preproc
    ):
        # Test that TRAC path uses _preprocess_taxonomy_aggregation correctly
        base = pd.DataFrame(
            {"FSp1__t0": [1, 2], "FSp2__t0": [3, 4]}, index=["S1", "S2"]
        )
        base_unsuffixed = pd.DataFrame(
            {"FSp1": [1, 2], "FSp2": [3, 4]}, index=["S1", "S2"]
        )
        mock_agg.return_value = base_unsuffixed
        mock_sel.return_value = base_unsuffixed
        mock_trans.return_value = base_unsuffixed
        # enriched snapshot equals transformed with suffix
        enriched = pd.DataFrame(
            {"FSp1__t0": [1, 2], "FSp2__t0": [3, 4]}, index=["S1", "S2"]
        )
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


if __name__ == "__main__":
    unittest.main()
