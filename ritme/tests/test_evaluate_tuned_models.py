import tempfile
import unittest
from unittest.mock import ANY, MagicMock, patch

import pandas as pd
from pandas.testing import assert_frame_equal

from ritme.evaluate_models import TunedModel
from ritme.evaluate_tuned_models import (
    _calculate_metrics,
    _load_best_tuned_models,
    _predict_w_tuned_model,
    cli_evaluate_tuned_models,
    evaluate_tuned_models,
)


class TestEvaluateTunedModels(unittest.TestCase):
    def setUp(self):
        self.mock_tuned_model = MagicMock(spec=TunedModel)
        self.exp_config = {"target": "test_target", "feature_prefix": "F"}

        self.train_val = pd.DataFrame(
            {"F1": [1, 2, 3], "F2": [4, 5, 6], "test_target": [7, 8, 9]}
        )
        self.test = pd.DataFrame(
            {"F1": [10, 11], "F2": [12, 13], "test_target": [14, 15]}
        )

        self.model_type = "mock_model"
        self.all_preds = pd.DataFrame(
            {
                "true": [7, 8, 9, 14, 15],
                "pred": [7.1, 8.1, 9.1, 14.1, 15.1],
                "split": ["train", "train", "train", "test", "test"],
            }
        )
        self.exp_metrics = pd.DataFrame(
            {
                "rmse_train": [0.1],
                "r2_train": [0.985],
                "rmse_test": [0.1],
                "r2_test": [0.96],
            },
            index=[self.model_type],
        )

    @patch("ritme.evaluate_tuned_models.get_predictions")
    def test_predict_w_tuned_model(self, mock_get_predictions):
        exp_all_preds = self.all_preds.copy()
        pred_train = exp_all_preds[exp_all_preds["split"] == "train"].copy()
        pred_test = exp_all_preds[exp_all_preds["split"] == "test"].copy()
        mock_get_predictions.side_effect = [
            pred_train,
            pred_test,
        ]

        all_preds = _predict_w_tuned_model(
            self.mock_tuned_model, self.exp_config, self.train_val, self.test
        )
        mock_get_predictions.assert_called_with(
            ANY, ANY, "test_target", ["F1", "F2"], ANY
        )
        assert_frame_equal(all_preds, exp_all_preds)

    def test_calculate_metrics(self):
        metrics = _calculate_metrics(self.all_preds, self.model_type)

        assert_frame_equal(
            self.exp_metrics,
            metrics,
        )

    @patch("os.listdir")
    def test_load_best_tuned_models(self, mock_listdir):
        mock_listdir.return_value = [
            "model1_best_model.pkl",
            "model2_best_model.pkl",
            "some_other_file.txt",
        ]

        model_types = _load_best_tuned_models("mock/path/to/exp")

        self.assertEqual(model_types, ["model1", "model2"])

    @patch("os.listdir")
    def test_load_best_tuned_models_no_models(self, mock_listdir):
        mock_listdir.return_value = ["some_other_file.txt", "another_file.csv"]

        with self.assertRaisesRegex(
            ValueError, "No best tuned models found in mock/path/to/exp"
        ):
            _load_best_tuned_models("mock/path/to/exp")

    @patch("ritme.evaluate_tuned_models._predict_w_tuned_model")
    def test_evaluate_tuned_models(self, mock_predict_w_tuned_model):
        mock_predict_w_tuned_model.side_effect = [
            self.all_preds,
            self.all_preds,
        ]

        dic_tuned_models = {
            "mock_model1": self.mock_tuned_model,
            "mock_model2": self.mock_tuned_model,
        }
        metrics = evaluate_tuned_models(
            dic_tuned_models, self.exp_config, self.train_val, self.test
        )
        exp_metrics = pd.concat([self.exp_metrics, self.exp_metrics])
        exp_metrics.index = ["mock_model1", "mock_model2"]
        assert_frame_equal(exp_metrics, metrics)

    @patch("ritme.evaluate_tuned_models.evaluate_tuned_models")
    @patch("ritme.evaluate_tuned_models.load_experiment_config")
    @patch("ritme.evaluate_tuned_models.load_best_model")
    @patch("ritme.evaluate_tuned_models._load_best_tuned_models")
    @patch("pandas.read_pickle")
    def test_cli_evaluate_tuned_models(
        self,
        mock_read_pickle,
        mock_load_best_tuned_models,
        mock_load_best_model,
        mock_load_experiment_config,
        mock_evaluate_tuned_models,
    ):
        mock_read_pickle.side_effect = [self.train_val, self.test]
        mock_load_best_tuned_models.return_value = ["mock_model"]
        mock_load_best_model.return_value = self.mock_tuned_model
        mock_load_experiment_config.return_value = self.exp_config
        mock_evaluate_tuned_models.return_value = self.exp_metrics

        path_exp = tempfile.TemporaryDirectory()
        path_to_exp = path_exp.name
        with patch("builtins.print") as mock_print:
            cli_evaluate_tuned_models(path_to_exp, "path/to/train_val", "path/to/test")
            mock_print.assert_called_with(
                f"Metrics were saved in {path_to_exp}/best_metrics.csv."
            )


if __name__ == "__main__":
    unittest.main()