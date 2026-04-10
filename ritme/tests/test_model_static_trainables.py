"""Testing static trainables"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import skbio
from parameterized import parameterized
from ray import tune
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

from ritme.evaluate_models import (
    TunedModel,
    get_data_processing,
    get_model,
    get_predictions,
    get_taxonomy,
)
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
