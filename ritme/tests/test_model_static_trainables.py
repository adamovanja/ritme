"""Testing static trainables"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

import joblib
import numpy as np
import pandas as pd
import ray
import skbio
from parameterized import parameterized
from ray import tune
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ritme.evaluate_models import (
    TunedModel,
    get_data_processing,
    get_model,
    get_predictions,
    get_taxonomy,
    load_xgb_model,
)
from ritme.feature_space._process_train import KFoldEngineered
from ritme.model_space import static_trainables as st
from ritme.split_train_test import _split_data_grouped
from ritme.tune_models import MODEL_TRAINABLES, _check_for_errors_in_trials


class TestHelperFunctions(unittest.TestCase):
    """Test all related to helper functions used by static_trainables"""

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

    @patch("ray.tune.get_context")
    def test_save_sklearn_model(self, mock_get_context):
        mock_trial_context = MagicMock()
        mock_trial_context.get_trial_id.return_value = "mock_trial_id"
        mock_get_context.return_value = mock_trial_context
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_trial_context.get_trial_dir.return_value = tmpdir

            result = st._save_sklearn_model(self.model)
            self.assertTrue(os.path.exists(result))

    @patch("ray.tune.report")
    @patch("ray.tune.get_context")
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

    @patch("ray.tune.report")
    @patch("ray.tune.get_context")
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


class TestTrainables(unittest.TestCase):
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

    @patch("ritme.model_space.static_trainables.process_train")
    @patch("ritme.model_space.static_trainables.StandardScaler.fit_transform")
    @patch("ritme.model_space.static_trainables.ElasticNet")
    @patch("ritme.model_space.static_trainables._report_results_manually")
    def test_train_linreg(
        self,
        mock_report,
        mock_elasticnet,
        mock_scaler_trf,
        mock_process_train,
    ):
        # define input parameters
        config = {"alpha": 0.1, "l1_ratio": 0.5}

        mock_process_train.return_value = (None, None, None, None)

        # run model
        st.train_linreg(
            config,
            self.train_val,
            self.target,
            self.host_id,
            None,
            self.seed_data,
            self.seed_model,
            self.tax,
        )

        # assert
        mock_process_train.assert_called_once_with(
            config,
            self.train_val,
            self.target,
            self.host_id,
            self.tax,
            self.seed_data,
            stratify_by=None,
        )
        mock_scaler_trf.assert_called_once()
        mock_elasticnet.assert_called_once_with(
            alpha=config["alpha"],
            l1_ratio=config["l1_ratio"],
            fit_intercept=True,
        )
        # Ensure fit() was called on the ElasticNet instance
        mock_elastic_instance = mock_elasticnet.return_value
        mock_elastic_instance.fit.assert_called_once()
        mock_report.assert_called_once()

    @patch("ritme.model_space.static_trainables.process_train")
    @patch("ritme.model_space.static_trainables.create_matrix_from_tree")
    @patch("ritme.model_space.static_trainables._preprocess_taxonomy_aggregation")
    @patch("ritme.model_space.static_trainables.Classo")
    @patch("ritme.model_space.static_trainables.min_least_squares_solution")
    @patch("ritme.model_space.static_trainables._report_results_manually_trac")
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
        mock_process_train.return_value = (None, None, None, None)
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
            None,
            self.seed_data,
            self.seed_model,
            self.tax,
            skbio.TreeNode(),
        )

        # Assert
        mock_process_train.assert_called_once_with(
            config,
            self.train_val,
            self.target,
            self.host_id,
            self.tax,
            self.seed_data,
            stratify_by=None,
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

    @patch("ritme.model_space.static_trainables.process_train")
    @patch("ritme.model_space.static_trainables.RandomForestRegressor")
    @patch("ritme.model_space.static_trainables._report_results_manually")
    def test_train_rf(self, mock_report, mock_rf, mock_process_train):
        # Arrange
        config = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 0.2,
            "min_weight_fraction_leaf": 0.001,
            "min_samples_leaf": 0.1,
            "max_features": "sqrt",
            "min_impurity_decrease": 0.0,
            "bootstrap": True,
        }

        mock_process_train.return_value = (None, None, None, None)
        mock_rf_instance = mock_rf.return_value

        # Act
        st.train_rf(
            config,
            self.train_val,
            self.target,
            self.host_id,
            None,
            self.seed_data,
            self.seed_model,
            self.tax,
        )

        # Assert
        mock_process_train.assert_called_once_with(
            config,
            self.train_val,
            self.target,
            self.host_id,
            self.tax,
            self.seed_data,
            stratify_by=None,
        )
        mock_rf.assert_called_once_with(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            min_weight_fraction_leaf=config["min_weight_fraction_leaf"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
            min_impurity_decrease=config["min_impurity_decrease"],
            bootstrap=config["bootstrap"],
            n_jobs=1,
            random_state=0,
        )
        mock_rf_instance.fit.assert_called_once()
        mock_report.assert_called_once()

    # def test_train_nn(self, mock_adam, mock_neural_net, mock_process_train):
    #     # todo: add unit test for pytorch NN

    @patch("ritme.model_space.static_trainables._save_taxonomy")
    @patch("ritme.model_space.static_trainables.process_train")
    @patch("ritme.model_space.static_trainables.xgb.DMatrix")
    @patch("ritme.model_space.static_trainables.xgb.train")
    @patch("ritme.model_space.static_trainables._RitmeXGBCheckpointCallback")
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
            "n_estimators": 100,
            "max_depth": 6,
            "min_child_weight": 3,
            "subsample": 0.9,
            "eta": 0.3,
            "num_parallel_tree": 2,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "colsample_bytree": 0.9,
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
            None,
            self.seed_data,
            self.seed_model,
            self.tax,
        )

        # Assert nthread is set to default cpus_per_trial and no GPU device
        self.assertEqual(config["nthread"], 1)
        self.assertNotIn("device", config)
        mock_process_train.assert_called_once_with(
            config,
            self.train_val,
            self.target,
            self.host_id,
            self.tax,
            self.seed_data,
            stratify_by=None,
        )
        mock_dmatrix.assert_has_calls(
            [
                call(mock_train_x, label=mock_train_y),
                call(mock_test_x, label=mock_test_y),
            ]
        )
        mock_xgb_train.assert_called_once()
        mock_checkpoint.assert_called_once()

    @patch("ritme.model_space.static_trainables._save_taxonomy")
    @patch("ritme.model_space.static_trainables.process_train")
    @patch("ritme.model_space.static_trainables.xgb.DMatrix")
    @patch("ritme.model_space.static_trainables.xgb.train")
    @patch("ritme.model_space.static_trainables._RitmeXGBCheckpointCallback")
    def test_train_xgb_with_gpu(
        self,
        mock_checkpoint,
        mock_xgb_train,
        mock_dmatrix,
        mock_process_train,
        mock_save_taxonomy,
    ):
        config = {"n_estimators": 100}
        mock_process_train.return_value = (
            np.zeros((2, 2)),
            np.zeros(2),
            np.zeros((1, 2)),
            np.zeros(1),
        )
        mock_dmatrix.return_value = None

        st.train_xgb(
            config,
            self.train_val,
            self.target,
            self.host_id,
            None,
            self.seed_data,
            self.seed_model,
            self.tax,
            cpus_per_trial=4,
            gpus_per_trial=1,
        )

        self.assertEqual(config["nthread"], 4)
        self.assertEqual(config["device"], "cuda")

    @parameterized.expand(
        [
            ("regression", [5, 10, 5, 1], None),
            ("classification", [5, 10, 5, 3], [0, 1, 2]),
            ("ordinal_regression", [5, 10, 5, 2], [0, 1, 2]),
        ]
    )
    @patch("ritme.model_space.static_trainables._save_taxonomy")
    @patch("ritme.model_space.static_trainables.seed_everything")
    @patch("ritme.model_space.static_trainables.process_train")
    @patch("ritme.model_space.static_trainables.load_data")
    @patch("ritme.model_space.static_trainables.NeuralNet")
    @patch("ritme.model_space.static_trainables.Trainer")
    @patch("ray.tune.get_context", return_value=MagicMock())
    def test_train_nn(
        self,
        nn_type,
        nb_units,
        classes,
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
            np.random.rand(10, 5),
            np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int64),
            np.random.rand(3, 5),
            np.array([0, 1, 0], dtype=np.int64),
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
            "dropout_rate": 0.0,
            "weight_decay": 0.0,
            "early_stopping_patience": 3,
            "early_stopping_min_delta": 0.0,
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
            None,
            nn_type=nn_type,
        )

        # Assertions to verify the expected behavior
        mock_seed_everything.assert_called_once_with(seed_model, workers=True)
        mock_process_train.assert_called_once_with(
            config,
            train_val,
            target,
            host_id,
            self.tax,
            seed_data,
            stratify_by=None,
        )
        # Verify DataLoader workers scale with cpus_per_trial (default=1 → 0 workers)
        load_data_call = mock_load_data.call_args
        self.assertEqual(load_data_call.kwargs.get("num_workers"), 0)
        mock_neural_net.assert_called_once_with(
            n_units=nb_units,
            learning_rate=0.01,
            nn_type=nn_type,
            dropout_rate=0.0,
            weight_decay=0.0,
            classes=classes,
            task_type="regression",
        )
        mock_trainer_instance.fit.assert_called()


class TestTrainableLogging(unittest.TestCase):
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
            "data_enrich": None,
            "alpha": 0.1,
            "l1_ratio": 0.5,
            "batch_size": 64,
            "n_hidden_layers": 1,
            "epochs": 2,
            "learning_rate": 0.01,
            "dropout_rate": 0.0,
            "weight_decay": 0.0,
            "early_stopping_patience": 3,
            "early_stopping_min_delta": 0.0,
            "n_estimators": 100,
        }
        for i in range(search_space["n_hidden_layers"]):
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
                    stratify_by=None,
                    tax=self.tax,
                    tree_phylo=self.tree_phylo,
                    cpus_per_trial=1,
                ),
                param_space=search_space,
                tune_config=tune.TuneConfig(metric=metric, mode=mode, num_samples=1),
                run_config=tune.RunConfig(storage_path=tmpdir),
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
            train, val = _split_data_grouped(
                self.train_val, self.host_id, 0.8, self.seed_data
            )

            for split, data in [("train", train), ("val", val)]:
                preds = get_predictions(data, tuned_model, self.target, split=split)
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


class TestRitmeXGBCheckpointCallback(unittest.TestCase):
    """Unit tests for the decoupled XGBoost callback.

    The callback reports metrics on every iteration (cheap, in-memory) but
    only writes Ray Tune checkpoints at two points per trial:

    1. On the *first* validation improvement (safety write so paused-then-killed
       HyperBand trials still have a checkpoint).
    2. At end of training (final write of the best booster seen).

    Improvements between the first and the last are held in-memory only. This
    bounds checkpoint writes to two per trial -- combined with ``num_to_keep=3``
    in ``tune_models.py`` it keeps the experiment-state snapshotter from being
    saturated by concurrent trials.
    """

    def _make_callback(self):
        return st._RitmeXGBCheckpointCallback(
            metrics={
                "r2_train": "train-r2",
                "r2_val": "val-r2",
                "rmse_train": "train-rmse",
                "rmse_val": "val-rmse",
            },
            filename="checkpoint",
            results_postprocessing_fn=lambda r: st.add_nb_features_to_results(r, 7),
            score_attr="rmse_val",
            score_mode="min",
        )

    def _evals_log(self, train_rmses, val_rmses):
        from collections import OrderedDict

        # XGBoost passes evals_log[set_name][metric_name] = list of historical
        # values; the callback reads the last entry of each list as "current".
        return OrderedDict(
            [
                (
                    "train",
                    OrderedDict(
                        [("r2", [0.0] * len(train_rmses)), ("rmse", list(train_rmses))]
                    ),
                ),
                (
                    "val",
                    OrderedDict(
                        [("r2", [0.0] * len(val_rmses)), ("rmse", list(val_rmses))]
                    ),
                ),
            ]
        )

    def test_invalid_score_mode_raises(self):
        with self.assertRaises(ValueError):
            st._RitmeXGBCheckpointCallback(
                metrics={"rmse_val": "val-rmse"},
                filename="checkpoint",
                results_postprocessing_fn=lambda r: r,
                score_attr="rmse_val",
                score_mode="invalid",
            )

    def test_after_iteration_writes_safety_checkpoint_only_on_first_improvement(self):
        # Hybrid: exactly one Ray Tune checkpoint write during iteration (the
        # first improvement). All other improvements are in-memory only.
        # _report_metrics is called every non-write iteration so ASHA can prune.
        callback = self._make_callback()
        model = MagicMock()
        # save_raw is called on improvements; return distinct payloads so we
        # can assert in-memory snapshot was updated to the *latest* improvement.
        model.save_raw.side_effect = [bytearray(b"snap0"), bytearray(b"snap2")]

        val_rmses = [0.50, 0.52, 0.40, 0.45]  # improvements at 0 and 2
        train_rmses = [0.40] * len(val_rmses)

        with patch.object(
            callback, "_save_and_report_checkpoint"
        ) as mock_save, patch.object(callback, "_report_metrics") as mock_report:
            for epoch in range(len(val_rmses)):
                callback.after_iteration(
                    model,
                    epoch,
                    self._evals_log(train_rmses[: epoch + 1], val_rmses[: epoch + 1]),
                )

        # Exactly one Ray Tune checkpoint write -- on the first improvement.
        mock_save.assert_called_once()
        first_call_dict = mock_save.call_args.args[0]
        self.assertEqual(first_call_dict["rmse_val"], 0.50)
        self.assertEqual(first_call_dict["nb_features"], 7)
        # Remaining iterations report metrics only (cheap, no checkpoint).
        self.assertEqual(mock_report.call_count, len(val_rmses) - 1)
        # In-memory best snapshot updated on every improvement (2 total).
        self.assertEqual(model.save_raw.call_count, 2)
        self.assertEqual(callback._best_score, 0.40)
        self.assertEqual(callback._best_model_bytes, bytearray(b"snap2"))
        self.assertEqual(callback._best_report_dict["nb_features"], 7)
        # Safety-write flag flipped exactly once.
        self.assertTrue(callback._wrote_safety_checkpoint)
        # Reported metric dicts always carry nb_features (postprocessing fn).
        for c in mock_report.call_args_list:
            self.assertEqual(c.args[0]["nb_features"], 7)
            self.assertIn("rmse_val", c.args[0])

    def test_after_training_writes_final_checkpoint_from_in_memory_best(self):
        callback = self._make_callback()
        model = MagicMock()
        model.save_raw.return_value = bytearray(b"best_state")

        # Patch _save_and_report_checkpoint and _report_metrics across both
        # the iteration loop and the after_training call so the iter-0 safety
        # write is captured (and doesn't try to call into Ray Tune).
        with patch.object(
            callback, "_save_and_report_checkpoint"
        ) as mock_save, patch.object(callback, "_report_metrics"):
            # One improvement at iter 0 (safety write), then no improvement.
            callback.after_iteration(model, 0, self._evals_log([0.4], [0.5]))
            callback.after_iteration(model, 1, self._evals_log([0.4, 0.4], [0.5, 0.6]))

            # Patch booster construction so we can verify it's loaded from
            # the in-memory snapshot, then handed to _save_and_report_checkpoint.
            loaded_booster = MagicMock()
            with patch(
                "ritme.model_space.static_trainables.xgb.Booster",
                return_value=loaded_booster,
            ) as mock_booster_cls:
                callback.after_training(model)

        # Two Ray Tune checkpoint writes total: safety write + final write.
        self.assertEqual(mock_save.call_count, 2)
        # Safety write (iter 0): handed the *current* booster, val-rmse=0.5.
        first_dict, first_model = mock_save.call_args_list[0].args
        self.assertEqual(first_dict["rmse_val"], 0.5)
        self.assertIs(first_model, model)
        # Final write (after_training): reconstructed booster from in-memory
        # bytes, with the *best* report dict.
        mock_booster_cls.assert_called_once_with()
        loaded_booster.load_model.assert_called_once_with(bytearray(b"best_state"))
        last_dict, last_model = mock_save.call_args_list[1].args
        self.assertEqual(last_dict["rmse_val"], 0.5)
        self.assertEqual(last_dict["nb_features"], 7)
        self.assertIs(last_model, loaded_booster)

    def test_after_training_falls_back_when_no_improvement_recorded(self):
        # If no iteration ever recorded an improvement (e.g. NaN scores
        # throughout), the callback must still emit a checkpoint at end so
        # downstream retrieval works.
        callback = self._make_callback()
        model = MagicMock()
        # Drive one iteration with NaN so _evals_log gets stored.
        with patch.object(callback, "_report_metrics"):
            callback.after_iteration(
                model, 0, self._evals_log([float("nan")], [float("nan")])
            )
        self.assertIsNone(callback._best_model_bytes)

        with patch.object(callback, "_save_and_report_checkpoint") as mock_save:
            callback.after_training(model)
        mock_save.assert_called_once()
        # Falls back to writing the *current* model, not a reloaded best.
        self.assertIs(mock_save.call_args.args[1], model)

    def test_nan_score_does_not_qualify_as_improvement(self):
        callback = self._make_callback()
        model = MagicMock()
        with patch.object(callback, "_report_metrics") as mock_report:
            callback.after_iteration(
                model, 0, self._evals_log([float("nan")], [float("nan")])
            )
        # No in-memory snapshot taken on NaN.
        model.save_raw.assert_not_called()
        self.assertIsNone(callback._best_model_bytes)
        # But metrics are still reported every iter.
        mock_report.assert_called_once()


class TestNNTuneReportCheckpointCallback(unittest.TestCase):
    """Unit tests for the decoupled PyTorch Lightning callback.

    Verifies that ``_handle`` reports metrics on every call (so ASHA can
    prune), persists Lightning checkpoints on every improvement to a per-trial
    scratch dir, and writes Ray Tune checkpoints at exactly two points per
    trial: a safety write on the *first* improvement (so paused-then-killed
    HyperBand trials still have a checkpoint) and a final write at
    ``on_train_end`` of the best validation state seen.
    """

    def _make_callback(self):
        return st.NNTuneReportCheckpointCallback(
            metrics={
                "rmse_val": "val_rmse",
                "rmse_train": "train_rmse",
            },
            filename="checkpoint",
            on="validation_end",
            nb_features=11,
        )

    def _trainer_with_score(self, val_rmse):
        trainer = MagicMock()
        trainer.sanity_checking = False
        # _get_report_dict reads trainer.callback_metrics[metric].item()
        train_metric = MagicMock()
        train_metric.item.return_value = 0.42
        val_metric = MagicMock()
        val_metric.item.return_value = val_rmse
        trainer.callback_metrics = {
            "val_rmse": val_metric,
            "train_rmse": train_metric,
        }
        return trainer

    def test_handle_writes_safety_checkpoint_only_on_first_improvement(self):
        callback = self._make_callback()
        val_rmses = [0.5, 0.6, 0.4, 0.45, 0.42]
        improvements = {0, 2}  # 0.5 < inf; 0.4 < 0.5

        # Track which trainers had save_checkpoint called.
        trainers = [self._trainer_with_score(v) for v in val_rmses]

        ckpt_path = (
            "ritme.model_space.static_trainables.ray.train" ".Checkpoint.from_directory"
        )
        with patch(ckpt_path, return_value="fake_safety_ckpt") as mock_from_dir, patch(
            "ritme.model_space.static_trainables.tune.report"
        ) as mock_report:
            for trainer in trainers:
                callback._handle(trainer, MagicMock())

        # tune.report is called every iteration.
        self.assertEqual(mock_report.call_count, len(val_rmses))

        # Exactly one call carries a Ray Tune checkpoint kwarg -- the very
        # first improvement (idx 0). The other 4 calls are metric-only.
        with_ckpt = [
            i
            for i, c in enumerate(mock_report.call_args_list)
            if "checkpoint" in c.kwargs
        ]
        self.assertEqual(with_ckpt, [0])
        self.assertEqual(
            mock_report.call_args_list[0].kwargs["checkpoint"], "fake_safety_ckpt"
        )
        # ``ray.train.Checkpoint.from_directory`` was invoked exactly once
        # during _handle (the safety write).
        mock_from_dir.assert_called_once()

        # Every report dict carries nb_features.
        for report_call in mock_report.call_args_list:
            self.assertEqual(report_call.args[0]["nb_features"], 11)

        # trainer.save_checkpoint is called only on improvements (per-trial
        # scratch-dir snapshots, no Ray Tune visibility).
        save_calls = [i for i, t in enumerate(trainers) if t.save_checkpoint.called]
        self.assertEqual(set(save_calls), improvements)

        # State after run.
        self.assertEqual(callback._best_score, 0.4)
        self.assertIsNotNone(callback._best_scratch_dir)
        self.assertIsNotNone(callback._best_report_dict)
        self.assertEqual(callback._best_report_dict["rmse_val"], 0.4)
        self.assertTrue(callback._wrote_safety_checkpoint)

        # Cleanup: scratch dir was created on disk.
        if callback._best_scratch_dir and os.path.isdir(callback._best_scratch_dir):
            shutil.rmtree(callback._best_scratch_dir, ignore_errors=True)

    def test_sanity_checking_short_circuits(self):
        callback = self._make_callback()
        trainer = self._trainer_with_score(0.1)
        trainer.sanity_checking = True
        with patch("ritme.model_space.static_trainables.tune.report") as mock_report:
            callback._handle(trainer, MagicMock())
        mock_report.assert_not_called()
        trainer.save_checkpoint.assert_not_called()
        self.assertIsNone(callback._best_scratch_dir)

    def test_on_train_end_reports_best_from_scratch_dir(self):
        callback = self._make_callback()
        # Simulate one prior improvement: scratch dir exists, best dict set.
        scratch = tempfile.mkdtemp(prefix="ritme_nn_best_test_")
        callback._best_scratch_dir = scratch
        callback._best_report_dict = {"rmse_val": 0.4, "nb_features": 11}
        try:
            ckpt_path = (
                "ritme.model_space.static_trainables.ray.train"
                ".Checkpoint.from_directory"
            )
            with patch(
                ckpt_path, return_value="fake_checkpoint"
            ) as mock_from_dir, patch(
                "ritme.model_space.static_trainables.tune.report"
            ) as mock_report:
                callback.on_train_end(self._trainer_with_score(0.99), MagicMock())
            mock_from_dir.assert_called_once_with(scratch)
            mock_report.assert_called_once()
            # Reported dict is the *best* one tracked, not the trainer's
            # current state, and the Ray checkpoint is attached.
            self.assertEqual(mock_report.call_args.args[0]["rmse_val"], 0.4)
            self.assertEqual(
                mock_report.call_args.kwargs.get("checkpoint"), "fake_checkpoint"
            )
            # Scratch dir is cleaned up after the report.
            self.assertFalse(os.path.isdir(scratch))
            self.assertIsNone(callback._best_scratch_dir)
        finally:
            if os.path.isdir(scratch):
                shutil.rmtree(scratch, ignore_errors=True)

    def test_on_train_end_falls_back_when_no_improvement_recorded(self):
        # No prior improvement: must still report a checkpoint built from the
        # current trainer state so downstream retrieval works.
        callback = self._make_callback()
        ckpt_cm = MagicMock()
        ckpt_cm.__enter__ = MagicMock(return_value="fallback_checkpoint")
        ckpt_cm.__exit__ = MagicMock(return_value=False)
        with patch.object(
            callback, "_get_checkpoint", return_value=ckpt_cm
        ) as mock_ckpt, patch(
            "ritme.model_space.static_trainables.tune.report"
        ) as mock_report:
            callback.on_train_end(self._trainer_with_score(0.7), MagicMock())
        mock_ckpt.assert_called_once()
        mock_report.assert_called_once()
        self.assertEqual(
            mock_report.call_args.kwargs.get("checkpoint"), "fallback_checkpoint"
        )


class TestKfoldHelpers(unittest.TestCase):
    """Helpers used by the K-fold path of every sklearn-style trainable."""

    def test_aggregate_fold_metrics_emits_mean_std_se(self):
        per_fold = [
            {"rmse_val": 0.40, "r2_val": 0.5},
            {"rmse_val": 0.50, "r2_val": 0.4},
            {"rmse_val": 0.60, "r2_val": 0.3},
        ]
        out = st._aggregate_fold_metrics(per_fold)
        # Bare key tracks the mean (so existing single-split metric names
        # keep working as Ray Tune's optimisation target).
        self.assertAlmostEqual(out["rmse_val"], 0.50, places=6)
        self.assertAlmostEqual(out["rmse_val_mean"], 0.50, places=6)
        # Sample std (ddof=1) of [0.4, 0.5, 0.6] is 0.1; SE = 0.1 / sqrt(3).
        self.assertAlmostEqual(out["rmse_val_std"], 0.1, places=6)
        self.assertAlmostEqual(out["rmse_val_se"], 0.1 / np.sqrt(3), places=6)
        self.assertEqual(out["n_folds"], 3)

    def test_aggregate_fold_metrics_handles_single_fold(self):
        # Single-fold input cannot support a meaningful SE; emit NaN so the
        # downstream 1-SE rule recognises the trial as unreliable.
        out = st._aggregate_fold_metrics([{"rmse_val": 0.45}])
        self.assertAlmostEqual(out["rmse_val_mean"], 0.45, places=6)
        self.assertTrue(np.isnan(out["rmse_val_std"]))
        self.assertTrue(np.isnan(out["rmse_val_se"]))
        self.assertEqual(out["n_folds"], 1)

    def test_aggregate_fold_metrics_skips_nan_only_keys(self):
        per_fold = [{"rmse_val": np.nan}, {"rmse_val": np.nan}]
        out = st._aggregate_fold_metrics(per_fold)
        self.assertNotIn("rmse_val", out)
        self.assertEqual(out["n_folds"], 2)

    def test_allocate_fold_resources_splits_cpus(self):
        # 14 CPUs, 5 folds -> 5 fold workers, 2 cpus per inner fit.
        n, c = st._allocate_fold_resources(n_splits=5, cpus_per_trial=14)
        self.assertEqual(n, 5)
        self.assertEqual(c, 2)

    def test_allocate_fold_resources_caps_workers_at_cpu_count(self):
        # 4 CPUs, 10 folds -> 4 workers in parallel, 1 cpu each.
        n, c = st._allocate_fold_resources(n_splits=10, cpus_per_trial=4)
        self.assertEqual(n, 4)
        self.assertEqual(c, 1)

    def test_allocate_fold_resources_handles_single_cpu(self):
        n, c = st._allocate_fold_resources(n_splits=5, cpus_per_trial=1)
        self.assertEqual(n, 1)
        self.assertEqual(c, 1)

    def test_fit_one_fold_sklearn_regression_uses_correct_train_val_slices(self):
        """``_fit_one_fold_sklearn_regression`` slices ``X_full`` / ``y_full``
        with ``train_idx`` for training and ``val_idx`` for validation. A
        regression that accidentally swapped the two would compute "val"
        metrics on the train set (or vice versa) -- this test pins the
        slicing identity by recomputing metrics independently from the
        produced slices.
        """
        rng = np.random.default_rng(0)
        n_rows, n_features = 20, 4
        X_full = rng.normal(size=(n_rows, n_features))
        y_full = X_full.sum(axis=1) + rng.normal(size=n_rows) * 0.05
        train_idx = np.arange(0, 15)
        val_idx = np.arange(15, 20)

        out = st._fit_one_fold_sklearn_regression(
            X_full,
            y_full,
            train_idx,
            val_idx,
            estimator_builder=st._build_linreg,
            builder_kwargs={"alpha": 0.01, "l1_ratio": 0.5},
            seed_model=0,
            cpus_per_fold=1,
        )

        # Independently recompute the metrics from the fitted slice
        # definition: train metrics on X_full[train_idx], val on X_full[val_idx].
        fit_model = st._build_linreg(alpha=0.01, l1_ratio=0.5)
        fit_model.fit(X_full[train_idx], y_full[train_idx])
        rmse_tr_exp, r2_tr_exp = st._predict_rmse_r2(
            fit_model, X_full[train_idx], y_full[train_idx]
        )
        rmse_va_exp, r2_va_exp = st._predict_rmse_r2(
            fit_model, X_full[val_idx], y_full[val_idx]
        )
        self.assertAlmostEqual(out["rmse_train"], rmse_tr_exp, places=8)
        self.assertAlmostEqual(out["r2_train"], r2_tr_exp, places=8)
        self.assertAlmostEqual(out["rmse_val"], rmse_va_exp, places=8)
        self.assertAlmostEqual(out["r2_val"], r2_va_exp, places=8)

        # And a sanity contrast: swapping train/val indices must change the
        # numbers (i.e. they are not symmetric / mistakenly computed on the
        # wrong slice in some way).
        swapped = st._fit_one_fold_sklearn_regression(
            X_full,
            y_full,
            val_idx,
            train_idx,
            estimator_builder=st._build_linreg,
            builder_kwargs={"alpha": 0.01, "l1_ratio": 0.5},
            seed_model=0,
            cpus_per_fold=1,
        )
        self.assertNotAlmostEqual(out["rmse_val"], swapped["rmse_val"], places=6)

    @patch("ritme.model_space.static_trainables.ray")
    def test_throttled_fold_dispatch_bounds_in_flight_and_runs_refit_after(
        self, mock_ray
    ):
        """The shared dispatcher submits at most ``n_workers`` fold tasks
        concurrently and submits the refit task only after all folds have
        completed. Earlier the dispatcher fired K+1 tasks at once with
        ``num_cpus=0``, allowing peak threads to roughly double the trial's
        CPU reservation.
        """
        n_folds = 5
        n_workers = 2

        in_flight_history = []  # snapshots of len(in_flight) over time
        submit_order = []
        refit_called_at_step = [None]
        step = [0]

        def fake_submit_fold(i):
            submit_order.append(("fold", i, step[0]))
            step[0] += 1
            return ("fold_ref", i)

        def fake_submit_refit():
            refit_called_at_step[0] = step[0]
            submit_order.append(("refit", None, step[0]))
            step[0] += 1
            return ("refit_ref",)

        # ray.wait returns one ready ref at a time (FIFO order of the
        # in-flight refs we hand it), simulating a worker freeing up.
        def fake_ray_wait(refs, num_returns):
            in_flight_history.append(len(refs))
            return ([refs[0]], refs[1:])

        # ray.get returns a numbered fake metric dict keyed by fold index.
        def fake_ray_get(ref):
            if isinstance(ref, tuple) and ref[0] == "fold_ref":
                return {"rmse_val": 0.1 * ref[1]}
            return {"full": True}

        mock_ray.wait.side_effect = fake_ray_wait
        mock_ray.get.side_effect = fake_ray_get

        fold_results, refit_result = st._dispatch_folds_then_refit(
            submit_fold=fake_submit_fold,
            n_folds=n_folds,
            submit_refit=fake_submit_refit,
            n_workers=n_workers,
        )

        # Folds appear in submission order (one slot per index).
        self.assertEqual(len(fold_results), n_folds)
        for i, r in enumerate(fold_results):
            self.assertAlmostEqual(r["rmse_val"], 0.1 * i, places=6)
        # Refit happens after all folds.
        self.assertEqual(refit_result, {"full": True})
        fold_submit_steps = [s for kind, _, s in submit_order if kind == "fold"]
        self.assertTrue(
            refit_called_at_step[0] > max(fold_submit_steps),
            "refit must be submitted after every fold has been submitted",
        )
        # In-flight count never exceeds n_workers at the moment of any
        # ``ray.wait`` call.
        self.assertTrue(in_flight_history, "ray.wait should be invoked")
        self.assertLessEqual(max(in_flight_history), n_workers)

    def test_aggregate_fold_metrics_one_valid_value_yields_nan_se(self):
        """A single valid observation cannot support a meaningful SE: K-1
        folds returned NaN (degenerate val split, SIGSEGV worker, etc.), so
        the trial's mean is a single-fold point estimate masquerading as a
        K-fold result. Emit ``_se = NaN`` so the downstream 1-SE rule can
        recognise the trial as unreliable and exclude it from selection,
        rather than silently treating it as a zero-noise winner.
        """
        per_fold = [
            {"rmse_val": np.nan},
            {"rmse_val": np.nan},
            {"rmse_val": 0.42},
        ]
        out = st._aggregate_fold_metrics(per_fold)
        self.assertAlmostEqual(out["rmse_val_mean"], 0.42, places=6)
        self.assertTrue(np.isnan(out["rmse_val_std"]))
        self.assertTrue(np.isnan(out["rmse_val_se"]))
        self.assertEqual(out["n_folds"], 3)

    def test_aggregate_fold_metrics_partial_nan_uses_only_valid_for_se(self):
        """SE divisor uses the count of *valid* observations, not the total
        fold count. With 3 real values and 2 NaNs, ``se == std / sqrt(3)``
        (not ``std / sqrt(5)``).
        """
        per_fold = [
            {"rmse_val": 0.40},
            {"rmse_val": np.nan},
            {"rmse_val": 0.50},
            {"rmse_val": np.nan},
            {"rmse_val": 0.60},
        ]
        out = st._aggregate_fold_metrics(per_fold)
        # std (ddof=1) of [0.4, 0.5, 0.6] is 0.1; SE = 0.1 / sqrt(3).
        self.assertAlmostEqual(out["rmse_val_mean"], 0.50, places=6)
        self.assertAlmostEqual(out["rmse_val_std"], 0.1, places=6)
        self.assertAlmostEqual(out["rmse_val_se"], 0.1 / np.sqrt(3), places=6)
        # NOT divided by 5 (the total fold count).
        self.assertNotAlmostEqual(out["rmse_val_se"], 0.1 / np.sqrt(5), places=6)
        self.assertEqual(out["n_folds"], 5)


class TestKfoldTrainables(unittest.TestCase):
    """End-to-end mocked tests for the K-fold path of each sklearn-style
    trainable. Each trainable's K-fold path goes through
    ``process_train_kfold`` -> ``_dispatch_kfold_and_refit_sklearn`` ->
    ``_finalize_and_report_sklearn``. We mock ``process_train_kfold`` and
    ``_dispatch_kfold_and_refit_sklearn`` so the test stays in-process (no
    Ray cluster spin-up), and patch ``ray.tune.get_context`` / ``tune.report``
    to capture what was reported.
    """

    def setUp(self):
        super().setUp()
        np.random.seed(0)
        n_rows = 30
        n_features = 5
        # 30 rows × 5 F-features + host_id grouping column. Targets are a
        # noisy sum so any real model fit produces finite metrics.
        X = np.random.randn(n_rows, n_features)
        feature_cols = [f"F{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_cols)
        df["target"] = X.sum(axis=1) + np.random.randn(n_rows) * 0.1
        df["host_id"] = np.repeat(np.arange(10), 3)  # 10 hosts, 3 rows each
        self.train_val = df
        self.feature_cols = feature_cols
        self.X_full = X
        self.y_reg = df["target"].values
        # Classification targets in {0, 1, 2} so log_loss/roc_auc work.
        self.y_class = np.tile([0, 1, 2], n_rows // 3)[:n_rows].astype(float)
        self.target = "target"
        self.host_id = "host_id"
        self.seed_data = 0
        self.seed_model = 0
        self.tax = pd.DataFrame([])
        self.tree_phylo = skbio.TreeNode()
        # Per-fold metric dicts the mocked dispatcher will return.
        self.reg_fold_metrics = [
            {"rmse_val": 0.40, "rmse_train": 0.30, "r2_val": 0.7, "r2_train": 0.8},
            {"rmse_val": 0.50, "rmse_train": 0.32, "r2_val": 0.6, "r2_train": 0.78},
            {"rmse_val": 0.60, "rmse_train": 0.34, "r2_val": 0.5, "r2_train": 0.76},
        ]
        self.class_fold_metrics = [
            {
                "roc_auc_macro_ovr_val": 0.80,
                "roc_auc_macro_ovr_train": 0.90,
                "f1_macro_val": 0.70,
                "f1_macro_train": 0.85,
                "balanced_accuracy_val": 0.72,
                "balanced_accuracy_train": 0.86,
                "mcc_val": 0.55,
                "mcc_train": 0.75,
                "log_loss_val": 0.65,
                "log_loss_train": 0.40,
            },
            {
                "roc_auc_macro_ovr_val": 0.78,
                "roc_auc_macro_ovr_train": 0.91,
                "f1_macro_val": 0.68,
                "f1_macro_train": 0.86,
                "balanced_accuracy_val": 0.71,
                "balanced_accuracy_train": 0.87,
                "mcc_val": 0.53,
                "mcc_train": 0.76,
                "log_loss_val": 0.66,
                "log_loss_train": 0.41,
            },
            {
                "roc_auc_macro_ovr_val": 0.82,
                "roc_auc_macro_ovr_train": 0.89,
                "f1_macro_val": 0.72,
                "f1_macro_train": 0.84,
                "balanced_accuracy_val": 0.74,
                "balanced_accuracy_train": 0.85,
                "mcc_val": 0.57,
                "mcc_train": 0.74,
                "log_loss_val": 0.64,
                "log_loss_train": 0.39,
            },
        ]

    def _make_kfold_engineered(self, n_splits=3, classification=False):
        """Build a KFoldEngineered NamedTuple with the synthetic fixture data."""
        y = self.y_class if classification else self.y_reg
        # Trivial fold indices; the dispatcher is mocked so these are only used
        # for length / shape sanity by the trainable.
        rows = self.X_full.shape[0]
        per_fold_size = rows // n_splits
        fold_indices = []
        for k in range(n_splits):
            val_start = k * per_fold_size
            val_end = (k + 1) * per_fold_size if k < n_splits - 1 else rows
            val_idx = np.arange(val_start, val_end)
            train_idx = np.setdiff1d(np.arange(rows), val_idx)
            fold_indices.append((train_idx, val_idx))
        return KFoldEngineered(
            X_full=self.X_full,
            y_full=y,
            ft_ls_used=self.feature_cols,
            fold_indices=fold_indices,
        )

    def _assert_kfold_report(self, mock_report, metric_name, n_splits=3):
        """Common assertions for any K-fold sklearn-style trainable."""
        mock_report.assert_called_once()
        # tune.report(metrics=...) keyword-only call from
        # _finalize_and_report_sklearn.
        reported = mock_report.call_args.kwargs.get("metrics")
        if reported is None:
            # Fallback for positional call.
            reported = mock_report.call_args.args[0]
        self.assertIsInstance(reported, dict)
        # Required keys.
        for key in (
            metric_name,
            f"{metric_name}_mean",
            f"{metric_name}_std",
            f"{metric_name}_se",
            "n_folds",
            "model_path",
            "nb_features",
        ):
            self.assertIn(key, reported)
        # Bare key tracks the mean.
        self.assertAlmostEqual(
            reported[metric_name], reported[f"{metric_name}_mean"], places=12
        )
        # SE is a finite real, not NaN.
        se = reported[f"{metric_name}_se"]
        self.assertFalse(np.isnan(se))
        self.assertGreaterEqual(se, 0.0)
        # n_folds matches the requested splits.
        self.assertEqual(reported["n_folds"], n_splits)
        # nb_features matches the design matrix's column count.
        self.assertEqual(reported["nb_features"], self.X_full.shape[1])
        # model_path saved file exists and is loadable.
        model_path = reported["model_path"]
        self.assertTrue(os.path.exists(model_path))
        loaded = joblib.load(model_path)
        self.assertIsInstance(loaded, BaseEstimator)
        # Loaded estimator is already fitted (calling predict must not raise).
        loaded.predict(self.X_full[:1])
        return reported

    @patch("ritme.model_space.static_trainables._dispatch_kfold_and_refit_sklearn")
    @patch("ritme.model_space.static_trainables.process_train_kfold")
    @patch("ritme.model_space.static_trainables.tune.report")
    @patch("ritme.model_space.static_trainables.ray.tune.get_context")
    def test_train_linreg_kfold_reports_aggregated_metrics(
        self,
        mock_get_context,
        mock_report,
        mock_process_train_kfold,
        mock_dispatch,
    ):
        mock_process_train_kfold.return_value = self._make_kfold_engineered(
            n_splits=3, classification=False
        )
        # Real fitted estimator so the saved model is loadable end-to-end.
        fitted = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("linreg", ElasticNet(alpha=0.1, l1_ratio=0.5)),
            ]
        ).fit(self.X_full, self.y_reg)
        mock_dispatch.return_value = (list(self.reg_fold_metrics), fitted)

        config = {"alpha": 0.1, "l1_ratio": 0.5}
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ctx = MagicMock()
            mock_ctx.get_trial_dir.return_value = tmpdir
            mock_ctx.get_trial_id.return_value = "trial_kfold_linreg"
            mock_get_context.return_value = mock_ctx

            st.train_linreg(
                config,
                self.train_val,
                self.target,
                self.host_id,
                None,
                self.seed_data,
                self.seed_model,
                self.tax,
                self.tree_phylo,
                cpus_per_trial=1,
                k_folds=3,
            )

            self._assert_kfold_report(mock_report, "rmse_val", n_splits=3)

    @patch("ritme.model_space.static_trainables._dispatch_kfold_and_refit_sklearn")
    @patch("ritme.model_space.static_trainables.process_train_kfold")
    @patch("ritme.model_space.static_trainables.tune.report")
    @patch("ritme.model_space.static_trainables.ray.tune.get_context")
    def test_train_rf_kfold_reports_aggregated_metrics(
        self,
        mock_get_context,
        mock_report,
        mock_process_train_kfold,
        mock_dispatch,
    ):
        mock_process_train_kfold.return_value = self._make_kfold_engineered(
            n_splits=3, classification=False
        )
        fitted = RandomForestRegressor(n_estimators=5, random_state=0).fit(
            self.X_full, self.y_reg
        )
        mock_dispatch.return_value = (list(self.reg_fold_metrics), fitted)

        config = {
            "n_estimators": 5,
            "max_depth": 3,
            "min_samples_split": 0.2,
            "min_weight_fraction_leaf": 0.001,
            "min_samples_leaf": 0.1,
            "max_features": "sqrt",
            "min_impurity_decrease": 0.0,
            "bootstrap": True,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ctx = MagicMock()
            mock_ctx.get_trial_dir.return_value = tmpdir
            mock_ctx.get_trial_id.return_value = "trial_kfold_rf"
            mock_get_context.return_value = mock_ctx

            st.train_rf(
                config,
                self.train_val,
                self.target,
                self.host_id,
                None,
                self.seed_data,
                self.seed_model,
                self.tax,
                self.tree_phylo,
                cpus_per_trial=1,
                k_folds=3,
            )

            self._assert_kfold_report(mock_report, "rmse_val", n_splits=3)

    @patch("ritme.model_space.static_trainables.tune.report")
    @patch("ritme.model_space.static_trainables.ray.tune.get_context")
    def test_train_xgb_kfold_reports_aggregated_metrics(
        self,
        mock_get_context,
        mock_report,
    ):
        """K-fold path for ``train_xgb`` runs the real fold loop and emits one
        aggregated ``tune.report`` call with mean/std/SE per metric.

        We do NOT mock xgb internals or ``process_train_kfold`` -- the real
        fold loop must run, and the bare metric keys / ``n_folds`` /
        ``nb_features`` must be present in the aggregated dict.
        """
        config = {
            "data_aggregation": None,
            "data_selection": None,
            "data_selection_t": None,
            "data_transform": None,
            "data_enrich": None,
            "n_estimators": 25,
            "max_depth": 3,
            "learning_rate": 0.1,
            "gamma": 0.0,
            "min_child_weight": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "model": "xgb",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ctx = MagicMock()
            mock_ctx.get_trial_dir.return_value = tmpdir
            mock_ctx.get_trial_id.return_value = "trial_kfold_xgb"
            mock_get_context.return_value = mock_ctx

            st.train_xgb(
                config,
                self.train_val,
                self.target,
                self.host_id,
                None,
                self.seed_data,
                self.seed_model,
                self.tax,
                self.tree_phylo,
                cpus_per_trial=1,
                gpus_per_trial=0,
                task_type="regression",
                k_folds=3,
            )

            self.assertEqual(mock_report.call_count, 1)
            reported = mock_report.call_args.kwargs.get("metrics")
            if reported is None:
                reported = mock_report.call_args.args[0]
            self.assertIsInstance(reported, dict)
            for key in ("rmse_val", "rmse_train", "r2_val", "r2_train"):
                self.assertIn(key, reported)
                self.assertIn(f"{key}_mean", reported)
                self.assertIn(f"{key}_std", reported)
                self.assertIn(f"{key}_se", reported)
                # Bare key tracks the mean.
                self.assertAlmostEqual(
                    reported[key], reported[f"{key}_mean"], places=12
                )
            self.assertEqual(reported["n_folds"], 3)
            self.assertEqual(reported["nb_features"], self.X_full.shape[1])

    @patch("ritme.model_space.static_trainables.tune.report")
    @patch("ritme.model_space.static_trainables.ray.tune.get_context")
    def test_train_xgb_class_kfold_reports_aggregated_metrics(
        self,
        mock_get_context,
        mock_report,
    ):
        """K-fold path for ``train_xgb_class`` runs the real fold loop and
        emits one aggregated ``tune.report`` call with mean/std/SE per
        classification metric.

        Mirrors ``test_train_xgb_kfold_reports_aggregated_metrics`` but with
        a 3-class numeric target so ``multi:softprob`` is exercised end-to-end:
        feature engineering, K-fold splitting, per-fold xgb fit + best-iter
        metric eval, aggregation, and the full-data refit all run for real.
        """
        # Overwrite the regression target with a 3-class numeric target so
        # process_train_kfold treats it as numeric (returns floats), and
        # the trainable rounds + LabelEncodes inside the xgb-class path.
        train_val = self.train_val.copy()
        train_val["target"] = self.y_class

        config = {
            "data_aggregation": None,
            "data_selection": None,
            "data_selection_t": None,
            "data_transform": None,
            "data_enrich": None,
            "n_estimators": 25,
            "max_depth": 3,
            "learning_rate": 0.1,
            "gamma": 0.0,
            "min_child_weight": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "model": "xgb_class",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ctx = MagicMock()
            mock_ctx.get_trial_dir.return_value = tmpdir
            mock_ctx.get_trial_id.return_value = "trial_kfold_xgb_class"
            mock_get_context.return_value = mock_ctx

            st.train_xgb_class(
                config,
                train_val,
                self.target,
                self.host_id,
                None,
                self.seed_data,
                self.seed_model,
                self.tax,
                self.tree_phylo,
                cpus_per_trial=1,
                gpus_per_trial=0,
                task_type="classification",
                k_folds=3,
            )

            self.assertEqual(mock_report.call_count, 1)
            reported = mock_report.call_args.kwargs.get("metrics")
            if reported is None:
                reported = mock_report.call_args.args[0]
            self.assertIsInstance(reported, dict)
            # Mirror the metric KEYS that single-split train_xgb_class emits
            # (see its _RitmeXGBCheckpointCallback metrics dict).
            metric_keys = (
                "roc_auc_macro_ovr_train",
                "roc_auc_macro_ovr_val",
                "log_loss_train",
                "log_loss_val",
                "f1_macro_train",
                "f1_macro_val",
                "balanced_accuracy_train",
                "balanced_accuracy_val",
                "mcc_train",
                "mcc_val",
            )
            for key in metric_keys:
                self.assertIn(key, reported)
                self.assertIn(f"{key}_mean", reported)
                self.assertIn(f"{key}_std", reported)
                self.assertIn(f"{key}_se", reported)
                # Bare key tracks the mean.
                self.assertAlmostEqual(
                    reported[key], reported[f"{key}_mean"], places=12
                )
            self.assertEqual(reported["n_folds"], 3)
            self.assertEqual(reported["nb_features"], self.X_full.shape[1])

    @patch("ritme.model_space.static_trainables.tune.report")
    @patch("ritme.model_space.static_trainables.ray.tune.get_context")
    def test_train_xgb_kfold_saves_loadable_checkpoint(
        self,
        mock_get_context,
        mock_report,
    ):
        """K-fold refit booster is surfaced as a Ray Tune ``Checkpoint`` and
        is loadable end-to-end through ``load_xgb_model`` / ``TunedModel``.

        Verifies that the K-fold branch wires the deployable checkpoint
        through the *same* ``tune.report(metrics=..., checkpoint=...)`` API
        the single-split path uses (see
        :class:`_RitmeXGBCheckpointCallback`), so a downstream consumer
        consuming a ``Result`` object can reconstruct the trained booster
        via :func:`load_xgb_model` and predict on a fresh holdout.

        ``_save_xgb_checkpoint`` is a context manager whose temp dir is
        torn down at the close of the ``with`` block in
        ``_run_kfold_xgb``; the real Ray Tune ``tune.report`` persists the
        checkpoint contents to durable storage during the call, so the
        temp dir going away after the report is fine. Under a mocked
        ``tune.report`` that durable-copy never happens, so we use
        ``side_effect`` to materialise the checkpoint to a stable directory
        WHILE the trainable's ``with`` block is still open, then validate
        the load + predict after the trainable returns.
        """
        config = {
            "data_aggregation": None,
            "data_selection": None,
            "data_selection_t": None,
            "data_transform": None,
            "data_enrich": None,
            "n_estimators": 25,
            "max_depth": 3,
            "learning_rate": 0.1,
            "gamma": 0.0,
            "min_child_weight": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "model": "xgb",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ctx = MagicMock()
            mock_ctx.get_trial_dir.return_value = tmpdir
            mock_ctx.get_trial_id.return_value = "trial_kfold_xgb_ckpt"
            mock_get_context.return_value = mock_ctx

            # Persist the checkpoint contents to a stable directory during
            # the patched ``tune.report`` call. Ray's ``Checkpoint`` wraps a
            # temp dir whose lifetime ends with the context manager in
            # ``_save_xgb_checkpoint``, so we must read it now -- not after
            # the trainable returns. Mirrors what real Ray Tune does during
            # ``tune.report``: stage the checkpoint to durable storage.
            persisted = {}
            stable_ckpt_dir = os.path.join(tmpdir, "persisted_checkpoint")

            def _persist(*args, **kwargs):
                checkpoint = kwargs.get("checkpoint")
                if checkpoint is None and len(args) >= 2:
                    checkpoint = args[1]
                self.assertIsNotNone(
                    checkpoint,
                    "K-fold refit must surface a Checkpoint via tune.report",
                )
                # Copy the checkpoint files to a path that outlives the
                # trainable's temp dir.
                src = checkpoint.to_directory()
                shutil.copytree(src, stable_ckpt_dir)
                persisted["metrics"] = kwargs.get("metrics") or (
                    args[0] if args else None
                )

            mock_report.side_effect = _persist

            st.train_xgb(
                config,
                self.train_val,
                self.target,
                self.host_id,
                None,
                self.seed_data,
                self.seed_model,
                self.tax,
                self.tree_phylo,
                cpus_per_trial=1,
                gpus_per_trial=0,
                task_type="regression",
                k_folds=3,
            )

            mock_report.assert_called_once()
            self.assertTrue(
                os.path.isfile(os.path.join(stable_ckpt_dir, "checkpoint")),
                "K-fold refit checkpoint must land at the 'checkpoint' "
                "filename load_xgb_model reads via _get_checkpoint_path.",
            )

            # Exercise the public reload surface. ``load_xgb_model`` reads
            # ``result.checkpoint.to_directory() / 'checkpoint'``; we wrap
            # the stable directory in a fresh Checkpoint so the stub
            # ``Result`` mirrors what Ray Tune would hand back.
            stable_checkpoint = ray.train.Checkpoint.from_directory(stable_ckpt_dir)
            fake_result = MagicMock()
            fake_result.checkpoint = stable_checkpoint
            loaded_booster = load_xgb_model(fake_result)
            self.assertIsInstance(loaded_booster, st.xgb.Booster)

            # Wrap in TunedModel so the public reload surface (the way the
            # orchestrator instantiates it from a Result) is exercised.
            tuned = TunedModel(
                model=loaded_booster,
                data_config={k: v for k, v in config.items() if k.startswith("data_")},
                tax=self.tax,
                path=tmpdir,
                model_type="xgb",
            )
            self.assertIs(tuned.model, loaded_booster)

            # Smoke-predict on a 5-row holdout using the same feature
            # schema the refit booster was trained on. ``_run_kfold_xgb``
            # builds the design matrix via ``process_train_kfold`` from
            # ``self.train_val`` (5 F-columns under this fixture), so a
            # 5-row slice of the synthetic feature matrix is the closest
            # analogue to fresh test data with the same column count.
            holdout = st.xgb.DMatrix(self.X_full[:5])
            preds = loaded_booster.predict(holdout)
            self.assertEqual(preds.shape, (5,))
            self.assertTrue(np.all(np.isfinite(preds)))

    @patch("ritme.model_space.static_trainables._dispatch_kfold_and_refit_sklearn")
    @patch("ritme.model_space.static_trainables.process_train_kfold")
    @patch("ritme.model_space.static_trainables.tune.report")
    @patch("ritme.model_space.static_trainables.ray.tune.get_context")
    def test_train_logreg_kfold_reports_aggregated_metrics(
        self,
        mock_get_context,
        mock_report,
        mock_process_train_kfold,
        mock_dispatch,
    ):
        mock_process_train_kfold.return_value = self._make_kfold_engineered(
            n_splits=3, classification=True
        )
        fitted = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        C=1.0,
                        penalty="l2",
                        solver="saga",
                        max_iter=200,
                        random_state=0,
                    ),
                ),
            ]
        ).fit(self.X_full, np.round(self.y_class).astype(int))
        mock_dispatch.return_value = (list(self.class_fold_metrics), fitted)

        config = {"C": 1.0, "penalty": "l2"}
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ctx = MagicMock()
            mock_ctx.get_trial_dir.return_value = tmpdir
            mock_ctx.get_trial_id.return_value = "trial_kfold_logreg"
            mock_get_context.return_value = mock_ctx

            st.train_logreg(
                config,
                self.train_val,
                self.target,
                self.host_id,
                None,
                self.seed_data,
                self.seed_model,
                self.tax,
                self.tree_phylo,
                cpus_per_trial=1,
                k_folds=3,
            )

            self._assert_kfold_report(mock_report, "roc_auc_macro_ovr_val", n_splits=3)

    @patch("ritme.model_space.static_trainables._dispatch_kfold_and_refit_sklearn")
    @patch("ritme.model_space.static_trainables.process_train_kfold")
    @patch("ritme.model_space.static_trainables.tune.report")
    @patch("ritme.model_space.static_trainables.ray.tune.get_context")
    def test_train_rf_class_kfold_reports_aggregated_metrics(
        self,
        mock_get_context,
        mock_report,
        mock_process_train_kfold,
        mock_dispatch,
    ):
        mock_process_train_kfold.return_value = self._make_kfold_engineered(
            n_splits=3, classification=True
        )
        fitted = RandomForestClassifier(n_estimators=5, random_state=0).fit(
            self.X_full, np.round(self.y_class).astype(int)
        )
        mock_dispatch.return_value = (list(self.class_fold_metrics), fitted)

        config = {
            "n_estimators": 5,
            "max_depth": 3,
            "min_samples_split": 0.2,
            "min_weight_fraction_leaf": 0.001,
            "min_samples_leaf": 0.1,
            "max_features": "sqrt",
            "min_impurity_decrease": 0.0,
            "bootstrap": True,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ctx = MagicMock()
            mock_ctx.get_trial_dir.return_value = tmpdir
            mock_ctx.get_trial_id.return_value = "trial_kfold_rf_class"
            mock_get_context.return_value = mock_ctx

            st.train_rf_class(
                config,
                self.train_val,
                self.target,
                self.host_id,
                None,
                self.seed_data,
                self.seed_model,
                self.tax,
                self.tree_phylo,
                cpus_per_trial=1,
                k_folds=3,
            )

            self._assert_kfold_report(mock_report, "roc_auc_macro_ovr_val", n_splits=3)

    @patch("ritme.model_space.static_trainables._dispatch_kfold_and_refit_trac")
    @patch("ritme.model_space.static_trainables._preprocess_taxonomy_aggregation")
    @patch("ritme.model_space.static_trainables.create_matrix_from_tree")
    @patch("ritme.model_space.static_trainables.process_train_kfold")
    @patch("ritme.model_space.static_trainables.tune.report")
    @patch("ritme.model_space.static_trainables.ray.tune.get_context")
    def test_train_trac_kfold_reports_aggregated_metrics(
        self,
        mock_get_context,
        mock_report,
        mock_process_train_kfold,
        mock_create_matrix,
        mock_preprocess,
        mock_dispatch_trac,
    ):
        # Engineered features & fold indices look the same as for sklearn.
        mock_process_train_kfold.return_value = self._make_kfold_engineered(
            n_splits=3, classification=False
        )
        a_df = pd.DataFrame(
            np.eye(self.X_full.shape[1]),
            index=self.feature_cols,
            columns=self.feature_cols,
        )
        mock_create_matrix.return_value = a_df
        mock_preprocess.return_value = (
            self.X_full.copy(),
            np.ones(self.X_full.shape[1]),
        )
        # Per-fold metrics for three folds + one full-data refit.
        per_fold = [
            {"rmse_val": 0.40, "rmse_train": 0.30, "r2_val": 0.7, "r2_train": 0.8},
            {"rmse_val": 0.50, "rmse_train": 0.32, "r2_val": 0.6, "r2_train": 0.78},
            {"rmse_val": 0.60, "rmse_train": 0.34, "r2_val": 0.5, "r2_train": 0.76},
        ]
        # Match the trac single-split shape: model = {"model": <alpha DF>}.
        alpha_df = pd.DataFrame(
            {"alpha": [0.5, 0.0, 0.3, 0.0, 0.7, 0.2]},
            index=["intercept"] + self.feature_cols,
        )
        full_model = {"model": alpha_df, "matrix_a": a_df}
        mock_dispatch_trac.return_value = (per_fold, full_model)

        config = {"lambda": 0.1}
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ctx = MagicMock()
            mock_ctx.get_trial_dir.return_value = tmpdir
            mock_ctx.get_trial_id.return_value = "trial_kfold_trac"
            mock_get_context.return_value = mock_ctx

            st.train_trac(
                config,
                self.train_val,
                self.target,
                self.host_id,
                None,
                self.seed_data,
                self.seed_model,
                self.tax,
                self.tree_phylo,
                cpus_per_trial=1,
                k_folds=3,
            )

            mock_report.assert_called_once()
            reported = mock_report.call_args.kwargs.get("metrics")
            if reported is None:
                reported = mock_report.call_args.args[0]
            for key in (
                "rmse_val",
                "rmse_val_mean",
                "rmse_val_std",
                "rmse_val_se",
                "n_folds",
                "model_path",
                "nb_features",
            ):
                self.assertIn(key, reported)
            # Bare key tracks the mean.
            self.assertAlmostEqual(
                reported["rmse_val"], reported["rmse_val_mean"], places=12
            )
            self.assertEqual(reported["n_folds"], 3)
            # nb_features is the count of non-zero alpha coefficients in the
            # refit model: alpha_df has 4 non-zero values out of 6.
            n_nonzero = int((alpha_df["alpha"] != 0.0).sum())
            self.assertEqual(reported["nb_features"], n_nonzero)
            # SE is finite and non-negative.
            self.assertFalse(np.isnan(reported["rmse_val_se"]))
            self.assertGreaterEqual(reported["rmse_val_se"], 0.0)
            # The model was pickled to disk by the trainable.
            self.assertTrue(os.path.exists(reported["model_path"]))
