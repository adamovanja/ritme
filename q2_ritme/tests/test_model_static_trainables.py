"""Testing static trainables"""

import os
import tempfile
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import skbio
import torch
from parameterized import parameterized
from qiime2.plugin.testing import TestPluginBase
from ray import air, tune
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

from q2_ritme.evaluate_models import (
    TunedModel,
    get_data_processing,
    get_model,
    get_predictions,
    get_taxonomy,
)
from q2_ritme.model_space import static_trainables as st
from q2_ritme.split_train_test import _split_data_stratified
from q2_ritme.tune_models import MODEL_TRAINABLES, _check_for_errors_in_trials


class TestHelperFunctions(TestPluginBase):
    """Test all related to helper functions used by static_trainables"""

    package = "q2_ritme.tests"

    def setUp(self):
        super().setUp()
        self.X = np.array([[1, 2], [3, 4]])
        self.y = np.array([1, 2])
        self.model = LinearRegression().fit(self.X, self.y)
        self.tax = pd.DataFrame()

    def test_predict_rmse(self):
        exp_rmse = root_mean_squared_error(self.y, self.model.predict(self.X))
        exp_r2 = r2_score(self.y, self.model.predict(self.X))
        obs_rmse, obs_r2 = st._predict_rmse_r2(self.model, self.X, self.y)
        self.assertEqual(obs_rmse, exp_rmse)
        self.assertEqual(obs_r2, exp_r2)

    def test_predict_rmse_r2_trac(self):
        alpha = np.array([1.0, 0.1, 0.1])
        obs_rmse, obs_r2 = st._predict_rmse_r2_trac(alpha, self.X, self.y)
        self.assertAlmostEqual(obs_rmse, 0.2999, places=3)
        self.assertAlmostEqual(obs_r2, 0.6400, places=3)

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

            st._report_results_manually(
                self.model, self.X, self.y, self.X, self.y, self.tax
            )
            mock_report.assert_called_once()

    @patch("ray.air.session.report")
    @patch("ray.train.get_context")
    def test_report_results_manually_trac(self, mock_get_context, mock_report):
        mock_trial_context = MagicMock()
        mock_trial_context.get_trial_id.return_value = "mock_trial_id"
        mock_get_context.return_value = mock_trial_context
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_trial_context.get_trial_dir.return_value = tmpdir

            model = {
                "model": pd.DataFrame(
                    {"alpha": [1.0, 0.1, 0.1]}, index=["F1", "F2", "F3"]
                )
            }
            st._report_results_manually_trac(
                model, self.X, self.y, self.X, self.y, self.tax
            )
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
        self.tax = pd.DataFrame([])

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
            self.tax,
        )

        # assert
        mock_process_train.assert_called_once_with(
            config, self.train_val, self.target, self.host_id, self.tax, self.seed_data
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
        mock_create_matrix.return_value = pd.DataFrame(
            {"F1": [1, 0], "F2": [0, 1]}, index=["F1", "F2"]
        )
        mock_preprocess_taxonomy.side_effect = [
            (np.array([[1, 2], [3, 4]]), 2),
            (np.array([[5, 6], [7, 8]]), 2),
        ]
        mock_classo.return_value = np.array([0.1, 0.1, 0.2])
        mock_min_least_squares.return_value = np.array([0.1, 0.1, 0.2])

        # Act
        st.train_trac(
            config,
            self.train_val,
            self.target,
            self.host_id,
            self.seed_data,
            self.seed_model,
            self.tax,
            skbio.TreeNode(),
        )

        # Assert
        mock_process_train.assert_called_once_with(
            config, self.train_val, self.target, self.host_id, self.tax, self.seed_data
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
            self.tax,
        )

        # Assert
        mock_process_train.assert_called_once_with(
            config, self.train_val, self.target, self.host_id, self.tax, self.seed_data
        )
        mock_rf.assert_called_once_with(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            n_jobs=None,
            random_state=0,
        )
        mock_rf_instance.fit.assert_called_once()
        mock_report.assert_called_once()

    # def test_train_nn(self, mock_adam, mock_neural_net, mock_process_train):
    #     # todo: add unit test for pytorch NN

    @patch("q2_ritme.model_space.static_trainables._save_taxonomy")
    @patch("q2_ritme.model_space.static_trainables.process_train")
    @patch("q2_ritme.model_space.static_trainables.xgb.DMatrix")
    @patch("q2_ritme.model_space.static_trainables.xgb.train")
    @patch("q2_ritme.model_space.static_trainables.xgb_cc")
    def test_train_xgb(
        self,
        mock_checkpoint,
        mock_xgb_train,
        mock_dmatrix,
        mock_process_train,
        mock_save_taxonomy,
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
            self.tax,
        )

        # Assert
        mock_process_train.assert_called_once_with(
            config, self.train_val, self.target, self.host_id, self.tax, self.seed_data
        )
        mock_dmatrix.assert_has_calls(
            [
                call(mock_train_x, label=mock_train_y),
                call(mock_test_x, label=mock_test_y),
            ]
        )
        mock_xgb_train.assert_called_once()
        mock_checkpoint.assert_called_once()

    @parameterized.expand(
        [
            ("regression", [5, 10, 5, 1]),
            ("classification", [5, 10, 5, 2]),
            ("ordinal_regression", [5, 10, 5, 1]),
        ]
    )
    @patch("q2_ritme.model_space.static_trainables._save_taxonomy")
    @patch("q2_ritme.model_space.static_trainables.seed_everything")
    @patch("q2_ritme.model_space.static_trainables.process_train")
    @patch("q2_ritme.model_space.static_trainables.load_data")
    @patch("q2_ritme.model_space.static_trainables.NeuralNet")
    @patch("q2_ritme.model_space.static_trainables.Trainer")
    @patch("ray.train.get_context", return_value=MagicMock())
    def test_train_nn(
        self,
        nn_type,
        nb_units,
        mock_get_context,
        mock_trainer,
        mock_neural_net,
        mock_load_data,
        mock_process_train,
        mock_seed_everything,
        mock_save_taxonomy,
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

        # Create a mock context object with a get_trial_dir method
        mock_context = mock_get_context.return_value
        mock_context.get_trial_dir.return_value = tempfile.mkdtemp()

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
        st.train_nn(
            config,
            train_val,
            target,
            host_id,
            self.tax,
            seed_data,
            seed_model,
            nn_type=nn_type,
        )

        # Assertions to verify the expected behavior
        mock_seed_everything.assert_called_once_with(seed_model, workers=True)
        mock_process_train.assert_called_once_with(
            config, train_val, target, host_id, self.tax, seed_data
        )
        mock_load_data.assert_called()
        mock_neural_net.assert_called_once_with(
            n_units=nb_units, learning_rate=0.01, nn_type=nn_type
        )
        mock_trainer_instance.fit.assert_called()


class TestTrainableLogging(TestPluginBase):
    package = "q2_ritme.tests"

    def setUp(self):
        super().setUp()
        np.random.seed(42)
        self.X = np.random.randn(1000, 10)
        self.y = np.sum(self.X, axis=1) + np.random.randn(1000) * 0.1
        self.features = [f"F{i}" for i in range(10)]
        self.train_val = pd.DataFrame(self.X, columns=[f"F{i}" for i in range(10)])
        self.train_val["target"] = self.y
        self.train_val["host_id"] = np.random.randint(0, 5, 1000)
        self.seed_data = 42
        self.seed_model = 42
        self.host_id = "host_id"
        self.target = "target"
        self.tax = pd.DataFrame([])
        self.tree_phylo = skbio.TreeNode()

    def _create_tuned_model(self, model_type, best_result):
        model = get_model(model_type, best_result)
        return TunedModel(
            model,
            get_data_processing(best_result),
            get_taxonomy(best_result),
            best_result.path,
        )

    @parameterized.expand(
        [
            ("linreg",),
            ("xgb",),
            ("nn_reg",),
        ]
    )
    def test_logged_vs_bestresult_rmse(self, model_type):
        """
        Verify that logged rmse values are identical to metrics obtained with
        best result's checkpoint for validation set. This is intentionally
        tested for one model of each type of checkpoint callbacks, namely manual
        reporting (linreg representative for trac, rf), and each of the own
        checkpoint_callbacks (xgb and nn_reg representative for all NNs).

        Note: this test works for train set only for linreg and xgb, NN fails.
        This issue was raised with tune here:
        https://github.com/ray-project/ray/issues/47333

        """
        # fit model
        search_space = {
            "data_selection": None,
            "data_aggregation": None,
            "data_transform": None,
            "data_alr_denom_idx": None,
            "alpha": 0.1,
            "l1_ratio": 0.5,
            "batch_size": 64,
            "n_hidden_layers": 1,
            "epochs": 2,
            "learning_rate": 0.01,
            "max_layers": 2,
        }
        for i in range(search_space["max_layers"]):
            search_space[f"n_units_hl{i}"] = 2
        metric = "rmse_val"
        mode = "min"

        with tempfile.TemporaryDirectory() as tmpdir:
            tuner = tune.Tuner(
                tune.with_parameters(
                    MODEL_TRAINABLES[model_type],
                    train_val=self.train_val,
                    target=self.target,
                    host_id=self.host_id,
                    seed_data=self.seed_data,
                    seed_model=self.seed_model,
                    tax=self.tax,
                    tree_phylo=self.tree_phylo,
                ),
                param_space=search_space,
                tune_config=tune.TuneConfig(metric=metric, mode=mode, num_samples=1),
                run_config=air.RunConfig(storage_path=tmpdir),
            )
            results = tuner.fit()
            _check_for_errors_in_trials(results)

            # get logs
            best_result = results.get_best_result("rmse_val", "min", "all")
            logged_rmse = {
                "train": best_result.metrics["rmse_train"],
                "val": best_result.metrics["rmse_val"],
            }
            # get recreated predictions & assert
            tuned_model = self._create_tuned_model(model_type, best_result)
            # split data with same split as during training - ensures with
            # self.seed_data
            train, val = _split_data_stratified(
                self.train_val, self.host_id, 0.8, self.seed_data
            )

            for split, data in [("train", train), ("val", val)]:
                preds = get_predictions(
                    data, tuned_model, self.target, self.features, split=split
                )
                calculated_rmse = np.sqrt(
                    mean_squared_error(preds["pred"], preds["true"])
                )
                if split == "val":
                    # this test works for train set only for linreg and xgb, NN
                    # fails. This issue was raised with tune here:
                    # https://github.com/ray-project/ray/issues/47333
                    self.assertAlmostEqual(
                        logged_rmse[split], calculated_rmse, places=6
                    )
