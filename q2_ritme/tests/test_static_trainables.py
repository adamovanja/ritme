"""Testing static trainables"""

import os
import tempfile
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import skbio
import torch
from qiime2.plugin.testing import TestPluginBase
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from q2_ritme.model_space import static_trainables as st


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

    @patch("ray.train.get_context")
    def test_save_sklearn_model(self, mock_get_context):
        mock_trial_context = MagicMock()
        mock_trial_context.get_trial_id.return_value = "mock_trial_id"
        mock_get_context.return_value = mock_trial_context
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_trial_context.get_trial_dir.return_value = tmpdir

            result = st._save_sklearn_model(self.model)
            self.assertTrue(os.path.exists(result))

    @patch("ray.air.session.report")
    @patch("ray.train.get_context")
    def test_report_results_manually(self, mock_get_context, mock_report):
        mock_trial_context = MagicMock()
        mock_trial_context.get_trial_id.return_value = "mock_trial_id"
        mock_get_context.return_value = mock_trial_context
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_trial_context.get_trial_dir.return_value = tmpdir

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

    @patch("q2_ritme.model_space.static_trainables.process_train")
    @patch("q2_ritme.model_space.static_trainables.ElasticNet")
    @patch("q2_ritme.model_space.static_trainables._report_results_manually")
    def test_train_linreg(self, mock_report, mock_linreg, mock_process_train):
        # define input parameters
        config = {"fit_intercept": True, "alpha": 0.1, "l1_ratio": 0.5}

        mock_process_train.return_value = (None, None, None, None, None)
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
        mock_linreg.assert_called_once_with(
            alpha=config["alpha"],
            l1_ratio=config["l1_ratio"],
            fit_intercept=config["fit_intercept"],
        )
        mock_linreg_instance.fit.assert_called_once()
        mock_report.assert_called_once()

    @patch("q2_ritme.model_space.static_trainables.process_train")
    @patch("q2_ritme.model_space.static_trainables.create_matrix_from_tree")
    @patch("q2_ritme.model_space.static_trainables._preprocess_taxonomy_aggregation")
    @patch("q2_ritme.model_space.static_trainables.Classo")
    @patch("q2_ritme.model_space.static_trainables.min_least_squares_solution")
    @patch("q2_ritme.model_space.static_trainables._report_results_manually_trac")
    def test_train_trac(
        self,
        mock_report,
        mock_min_least_squares,
        mock_classo,
        mock_preprocess_taxonomy,
        mock_create_matrix,
        mock_process_train,
    ):
        # Arrange
        config = {"lambda": 0.1}
        mock_process_train.return_value = (None, None, None, None, None)
        mock_create_matrix.return_value = pd.DataFrame()
        mock_preprocess_taxonomy.side_effect = [
            (np.array([[1, 2], [3, 4]]), 2),
            (np.array([[5, 6], [7, 8]]), 2),
        ]
        mock_classo.return_value = np.array([0.1, 0.2])
        mock_min_least_squares.return_value = np.array([0.1, 0.2])

        # Act
        st.train_trac(
            config,
            self.train_val,
            self.target,
            self.host_id,
            self.seed_data,
            self.seed_model,
            pd.DataFrame(),
            skbio.TreeNode(),
        )

        # Assert
        mock_process_train.assert_called_once_with(
            config, self.train_val, self.target, self.host_id, self.seed_data
        )
        mock_create_matrix.assert_called_once()
        assert mock_preprocess_taxonomy.call_count == 2

        # mock_classo.assert_called_once_with doesn't work because matrix is a
        # numpy array
        kwargs = mock_classo.call_args.kwargs

        self.assertTrue(np.array_equal(kwargs["matrix"][0], np.array([[1, 2], [3, 4]])))
        self.assertTrue(np.array_equal(kwargs["matrix"][1], np.ones((1, 2))))
        self.assertIsNone(kwargs["matrix"][2])
        self.assertEqual(kwargs["lam"], config["lambda"])
        self.assertEqual(kwargs["typ"], "R1")
        self.assertEqual(kwargs["meth"], "Path-Alg")
        self.assertEqual(kwargs["w"], 0.5)
        self.assertEqual(kwargs["intercept"], True)

        mock_min_least_squares.assert_called_once()
        mock_report.assert_called_once()

    @patch("q2_ritme.model_space.static_trainables.process_train")
    @patch("q2_ritme.model_space.static_trainables.RandomForestRegressor")
    @patch("q2_ritme.model_space.static_trainables._report_results_manually")
    def test_train_rf(self, mock_report, mock_rf, mock_process_train):
        # Arrange
        config = {"n_estimators": 100, "max_depth": 10}

        mock_process_train.return_value = (None, None, None, None, None)
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

    @patch("q2_ritme.model_space.static_trainables.process_train")
    @patch("q2_ritme.model_space.static_trainables.xgb.DMatrix")
    @patch("q2_ritme.model_space.static_trainables.xgb.train")
    @patch("q2_ritme.model_space.static_trainables.xgb_cc")
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
        mock_ft_cols = ["F1", "F2"]
        mock_process_train.return_value = (
            mock_train_x,
            mock_train_y,
            mock_test_x,
            mock_test_y,
            mock_ft_cols,
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

    @patch("q2_ritme.model_space.static_trainables.seed_everything")
    @patch("q2_ritme.model_space.static_trainables.process_train")
    @patch("q2_ritme.model_space.static_trainables.load_data")
    @patch("q2_ritme.model_space.static_trainables.NeuralNet")
    @patch("q2_ritme.model_space.static_trainables.Trainer")
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
            ["F1", "F2", "F3", "F4", "F5"],
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
            n_units=[5, 10, 5, 1], learning_rate=0.01, nn_type="regression"
        )
        mock_trainer_instance.fit.assert_called()
