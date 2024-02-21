"""Testing static trainables"""
import os
import tempfile
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import torch
from qiime2.plugin.testing import TestPluginBase
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from q2_ritme.model_space import _static_trainables as st


class TestHelperFunctions(TestPluginBase):
    """Test all related to helper functions used by static_trainables"""

    package = "q2_ritme.tests"

    def setUp(self):
        super().setUp()
        self.X = np.array([[1, 2], [3, 4]])
        self.y = np.array([1, 2])
        self.model = LinearRegression().fit(self.X, self.y)

    def test_predict_rmse(self):
        expected = mean_squared_error(self.y, self.model.predict(self.X), squared=False)
        result = st._predict_rmse(self.model, self.X, self.y)
        self.assertEqual(result, expected)

    @patch("ray.tune.get_trial_dir")
    def test_save_sklearn_model(self, mock_get_trial_dir):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_get_trial_dir.return_value = tmpdir

            result = st._save_sklearn_model(self.model)
            self.assertTrue(os.path.exists(result))

    @patch("ray.air.session.report")
    @patch("ray.tune.get_trial_dir")
    def test_report_results_manually(self, mock_get_trial_dir, mock_report):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_get_trial_dir.return_value = tmpdir
            st._report_results_manually(self.model, self.X, self.y, self.X, self.y)
            mock_report.assert_called_once()


class TestTrainables(TestPluginBase):
    package = "q2_ritme.tests"

    def setUp(self):
        super().setUp()
        self.train_val = pd.DataFrame(
            {
                "host_id": [1, 2, 3],
                "target": [192.0, 221.0, 250.0],
                "F1": [0.1, 0.9, 0.0],
                "F2": [0.9, 0.1, 1.0],
            },
            index=["ERR1", "ERR2", "ERR3"],
        )
        self.target = "target"
        self.host_id = "host_id"
        self.seed_data = 0
        self.seed_model = 0

    @patch("q2_ritme.model_space._static_trainables.process_train")
    @patch("q2_ritme.model_space._static_trainables.LinearRegression")
    @patch("q2_ritme.model_space._static_trainables._report_results_manually")
    def test_train_linreg(self, mock_report, mock_linreg, mock_process_train):
        # define input parameters
        config = {"fit_intercept": True}

        mock_process_train.return_value = (None, None, None, None)
        mock_linreg_instance = mock_linreg.return_value

        # run model
        st.train_linreg(
            config,
            self.train_val,
            self.target,
            self.host_id,
            self.seed_data,
            self.seed_model,
        )

        # assert
        mock_process_train.assert_called_once_with(
            config, self.train_val, self.target, self.host_id, self.seed_data
        )
        mock_linreg.assert_called_once_with(fit_intercept=config["fit_intercept"])
        mock_linreg_instance.fit.assert_called_once()
        mock_report.assert_called_once()

    @patch("q2_ritme.model_space._static_trainables.process_train")
    @patch("q2_ritme.model_space._static_trainables.RandomForestRegressor")
    @patch("q2_ritme.model_space._static_trainables._report_results_manually")
    def test_train_rf(self, mock_report, mock_rf, mock_process_train):
        # Arrange
        config = {"n_estimators": 100, "max_depth": 10}

        mock_process_train.return_value = (None, None, None, None)
        mock_rf_instance = mock_rf.return_value

        # Act
        st.train_rf(
            config,
            self.train_val,
            self.target,
            self.host_id,
            self.seed_data,
            self.seed_model,
        )

        # Assert
        mock_process_train.assert_called_once_with(
            config, self.train_val, self.target, self.host_id, self.seed_data
        )
        mock_rf.assert_called_once_with(
            n_estimators=config["n_estimators"], max_depth=config["max_depth"]
        )
        mock_rf_instance.fit.assert_called_once()
        mock_report.assert_called_once()

    # def test_train_nn(self, mock_adam, mock_neural_net, mock_process_train):
    #     # todo: add unit test for pytorch NN

    @patch("q2_ritme.model_space._static_trainables.process_train")
    @patch("q2_ritme.model_space._static_trainables.xgb.DMatrix")
    @patch("q2_ritme.model_space._static_trainables.xgb.train")
    @patch("q2_ritme.model_space._static_trainables.xgb_cc")
    def test_train_xgb(
        self, mock_checkpoint, mock_xgb_train, mock_dmatrix, mock_process_train
    ):
        # Arrange
        config = {
            "max_depth": 6,
            "eta": 0.3,
            "objective": "multi:softprob",
            "num_class": 3,
        }

        mock_train = self.train_val.iloc[:2, :]
        mock_test = self.train_val.iloc[2:, :]
        mock_train_x = mock_train[["F1", "F2"]].values
        mock_train_y = mock_train[self.target].values
        mock_test_x = mock_test[["F1", "F2"]].values
        mock_test_y = mock_test[self.target].values
        mock_process_train.return_value = (
            mock_train_x,
            mock_train_y,
            mock_test_x,
            mock_test_y,
        )
        mock_dmatrix.return_value = None

        # Act
        st.train_xgb(
            config,
            self.train_val,
            self.target,
            self.host_id,
            self.seed_data,
            self.seed_model,
        )

        # Assert
        mock_process_train.assert_called_once_with(
            config, self.train_val, self.target, self.host_id, self.seed_data
        )
        mock_dmatrix.assert_has_calls(
            [
                call(mock_train_x, label=mock_train_y),
                call(mock_test_x, label=mock_test_y),
            ]
        )
        mock_xgb_train.assert_called_once()
        mock_checkpoint.assert_called_once()

    @patch("q2_ritme.model_space._static_trainables.seed_everything")
    @patch("q2_ritme.model_space._static_trainables.process_train")
    @patch("q2_ritme.model_space._static_trainables.load_data")
    @patch("q2_ritme.model_space._static_trainables.NeuralNet")
    @patch("q2_ritme.model_space._static_trainables.Trainer")
    def test_train_nn(
        self,
        mock_trainer,
        mock_neural_net,
        mock_load_data,
        mock_process_train,
        mock_seed_everything,
    ):
        # Setup mock return values
        mock_process_train.return_value = (
            torch.rand(10, 5),
            torch.rand(10),
            torch.rand(3, 5),
            torch.rand(3),
        )
        mock_load_data.return_value = (MagicMock(), MagicMock())
        mock_trainer_instance = mock_trainer.return_value

        # Define dummy config and parameters
        config = {
            "n_hidden_layers": 2,
            "n_units_hl0": 10,
            "n_units_hl1": 5,
            "learning_rate": 0.01,
            "epochs": 5,
            "checkpoint_dir": "checkpoints",
        }
        train_val = MagicMock()
        target = "target"
        host_id = "host_id"
        seed_data = 42
        seed_model = 42

        # Call the function under test
        st.train_nn(config, train_val, target, host_id, seed_data, seed_model)

        # Assertions to verify the expected behavior
        mock_seed_everything.assert_called_once_with(seed_model, workers=True)
        mock_process_train.assert_called_once_with(
            config, train_val, target, host_id, seed_data
        )
        mock_load_data.assert_called()
        mock_neural_net.assert_called_once_with(
            n_units=[5, 10, 5, 1], learning_rate=0.01
        )
        mock_trainer_instance.fit.assert_called()