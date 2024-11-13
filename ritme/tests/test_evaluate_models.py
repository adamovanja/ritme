import pickle
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
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
    retrieve_best_models,
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
    def test_retrieve_best_models(
        self, mock_get_taxonomy, mock_get_data_processing, mock_get_model
    ):
        result_dic = {"xgb": self.result_grid, "nn_reg": self.result_grid}

        best_models = retrieve_best_models(result_dic)

        self.assertIsInstance(best_models, dict)
        self.assertEqual(len(best_models), 2)
        for model_type, tuned_model in best_models.items():
            self.assertIn(model_type, ["xgb", "nn_reg"])
            self.assertIsInstance(tuned_model, TunedModel)

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


if __name__ == "__main__":
    unittest.main()
