import pickle
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
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
        predictions = get_predictions(
            self.data, self.tmodel, "target", ["feature1", "feature2"], "train"
        )
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertIn("true", predictions.columns)
        self.assertIn("pred", predictions.columns)
        self.assertIn("split", predictions.columns)


class DummySklearnModel:
    """
    A minimal model that imitates an sklearn-like model.
    """

    def predict(self, X):
        return np.mean(X, axis=1)


class TestTunedModelImplementation(unittest.TestCase):
    # only testing functionality that was not already tested elsewhere

    @classmethod
    def setUpClass(cls):
        # Minimal taxonomy DataFrame
        cls.tax_df = pd.DataFrame(
            {"Taxon": ["c__Bacilli", "c__Clostridia"], "Confidence": [0.9, 0.8]},
            index=["FSp1", "FSp2"],
        )
        # Minimal data_config to exercise aggregator, selector & transformer
        cls.data_config = {
            "data_aggregation": "tax_class",
            "data_transform": "ilr",
            "data_selection": "abundance_ith",
            "data_alr_denom_idx": 0,
        }

    def setUp(self):
        self.data = pd.DataFrame(
            {
                "FSp1": [10, 5],
                "FSp2": [3, 2],
                "Other": [1, 1],
            }
        )
        self.data_test = pd.DataFrame(
            {
                "FSp1": [15, 6],
                "FSp2": [5, 5],
                "Other": [2, 2],
            }
        )
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
        aggregated_df = self.tmodel.aggregate(self.data)
        self.assertIsInstance(aggregated_df, pd.DataFrame)
        mock_aggregate.assert_called_once()

    @patch("ritme.evaluate_models.select_microbial_features")
    def test_select_train(self, mock_select):
        mock_select.return_value = self.data[["FSp1"]]
        _ = self.tmodel.select(self.data, split="train")
        self.assertEqual(self.tmodel.train_selected_fts, ["FSp1"])

    def test_select_test_fails_if_no_train_run_before(self):
        fresh_tmodel = TunedModel(
            model=self.model,
            data_config=self.data_config,
            tax=self.tax_df,
            path="/some/fake/path",
        )
        with self.assertRaisesRegex(
            ValueError,
            "To run tmodel.predict on the test set it has to be run on the train "
            "set first.",
        ):
            fresh_tmodel.select(self.data, split="test")

    @patch("ritme.evaluate_models.select_microbial_features")
    def test_select_test_w_train_run_before(self, mock_select):
        # run on train
        mock_select.return_value = self.data[["FSp1"]]
        _ = self.tmodel.select(self.data, split="train")
        # run on test
        df_test_selected = self.tmodel.select(self.data_test, split="test")
        assert_frame_equal(df_test_selected, self.data_test[["FSp1"]])

    @patch("ritme.evaluate_models._preprocess_taxonomy_aggregation")
    @patch.object(TunedModel, "transform")
    @patch.object(TunedModel, "select")
    @patch.object(TunedModel, "aggregate")
    def test_predict_trac_model(self, mock_agg, mock_sel, mock_trans, mock_preproc):
        # only trac model tested since other trainables are already covered with
        # other tests
        mock_preproc.return_value = (np.array([[2], [3]]), None)

        trac_model = {
            "matrix_a": pd.DataFrame(),
            "model": pd.DataFrame({"alpha": [2, 3]}, index=["FSp1", "FSp2"]),
        }
        tuned = TunedModel(
            model=trac_model,
            data_config={},
            tax=pd.DataFrame(),
            path="",
        )

        preds = tuned.predict(pd.DataFrame(), split="train")
        # alpha = [2,3]
        # log_geom = [[2],[3]]
        # predicted = log_geom.dot(alpha[1:]) + alpha[0]
        # => [ (2*3)+2, (3*3)+2 ] => [8,11]
        assert_array_equal(preds, np.array([[8], [11]]))


if __name__ == "__main__":
    unittest.main()
