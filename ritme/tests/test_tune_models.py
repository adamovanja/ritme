# test_script.py

import os
import unittest
from functools import partial
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import skbio
from parameterized import parameterized
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune import ResultGrid
from ray.tune.schedulers import AsyncHyperBandScheduler, HyperBandScheduler
from ray.tune.search.optuna import OptunaSearch

from ritme.model_space import static_searchspace as ss
from ritme.tune_models import (
    MODEL_TRAINABLES,
    OPTUNA_SAMPLER_CLASSES,
    _adaptive_n_startup_trials,
    _check_for_errors_in_trials,
    _define_callbacks,
    _define_scheduler,
    _define_search_algo,
    _get_resources,
    _get_slurm_resource,
    _load_wandb_api_key,
    _load_wandb_entity,
    _RecordingTrial,
    _SafeMLflowLoggerCallback,
    run_all_trials,
    run_trials,
)


class TestHelpersTuneModels(unittest.TestCase):
    @patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "4"})
    def test_get_slurm_resource_present(self):
        # Test when the environment variable is present
        self.assertEqual(_get_slurm_resource("SLURM_CPUS_PER_TASK"), 4)

    @patch.dict(os.environ, {}, clear=True)
    def test_get_slurm_resource_absent(self):
        # Test when the environment variable is absent
        self.assertEqual(_get_slurm_resource("SLURM_CPUS_PER_TASK"), 0)

    @patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "invalid"})
    def test_get_slurm_resource_invalid(self):
        # Test when the environment variable is invalid
        self.assertEqual(_get_slurm_resource("SLURM_CPUS_PER_TASK"), 0)

    def test_check_for_errors_in_trials_no_errors(self):
        # Mock ResultGrid with no errors
        mock_result = MagicMock(spec=ResultGrid)
        mock_result.num_errors = 0
        # Should not raise an exception
        _check_for_errors_in_trials(mock_result)

    def test_check_for_errors_in_trials_with_errors(self):
        # Mock ResultGrid with errors
        mock_result = MagicMock(spec=ResultGrid)
        mock_result.num_errors = 1
        with self.assertRaises(RuntimeError):
            _check_for_errors_in_trials(mock_result)

    @patch("ritme.tune_models._get_slurm_resource")
    def test_get_resources(self, mock_get_slurm_resource):
        # Mock get_slurm_resource to return predefined values
        mock_get_slurm_resource.side_effect = [8, 2]
        resources = _get_resources(2)

        expected_resources = {"cpu": 4, "gpu": 1}
        self.assertEqual(resources, expected_resources)

        # Check that get_slurm_resource was called with correct arguments
        mock_get_slurm_resource.assert_any_call("SLURM_CPUS_PER_TASK", 1)
        mock_get_slurm_resource.assert_any_call("SLURM_GPUS_PER_TASK", 0)

    def test_define_scheduler_not_fully_reproducible(self):
        scheduler_max_t = 100

        scheduler = _define_scheduler(False, 10, scheduler_max_t)

        self.assertIsInstance(scheduler, AsyncHyperBandScheduler)
        self.assertEqual(scheduler._max_t, scheduler_max_t)

    def test_define_scheduler_fully_reproducible(self):
        scheduler_max_t = 100

        scheduler = _define_scheduler(True, 10, scheduler_max_t)

        self.assertIsInstance(scheduler, HyperBandScheduler)
        self.assertEqual(scheduler._max_t_attr, scheduler_max_t)

    @parameterized.expand(
        [
            "RandomSampler",
            "TPESampler",
            "CmaEsSampler",
            "GPSampler",
            "QMCSampler",
        ]
    )
    def test_define_search_algo(self, sampler):
        mock_func_to_get_search_space = Mock()

        # Use a real model type so the adaptive n_startup_trials path (which
        # introspects ss.get_search_space) does not raise for samplers that
        # need it (TPE / CmaEs / GP). The mocked func_to_get_search_space
        # still controls what OptunaSearch sees as its space.
        exp_name = "linreg"
        tax = pd.DataFrame()
        train_val = pd.DataFrame({"F0": [1.0, 2.0, 3.0], "F1": [0.5, 1.5, 2.5]})
        model_hyperparameters = {}
        seed_model = 42
        metric = "accuracy"
        mode = "max"

        search_algo = _define_search_algo(
            mock_func_to_get_search_space,
            exp_name,
            tax,
            train_val,
            model_hyperparameters,
            sampler,
            seed_model,
            metric,
            mode,
        )

        self.assertIsInstance(search_algo, OptunaSearch)

        self.assertTrue(isinstance(search_algo._space, partial))
        search_algo._space()

        mock_func_to_get_search_space.assert_called_once_with(
            model_type=exp_name,
            tax=tax,
            train_val=train_val,
            model_hyperparameters=model_hyperparameters,
        )

        self.assertEqual(search_algo._metric, metric)
        self.assertEqual(search_algo._mode, mode)
        self.assertTrue(
            isinstance(search_algo._sampler, OPTUNA_SAMPLER_CLASSES[sampler])
        )

    def test_define_search_algo_invalid_sampler(self):
        invalid_sampler = "InvalidSampler"
        with self.assertRaisesRegex(
            ValueError, f"Unrecognized sampler '{invalid_sampler}'."
        ):
            _define_search_algo(
                Mock(),
                "linreg",
                pd.DataFrame(),
                pd.DataFrame({"F0": [1.0, 2.0, 3.0], "F1": [0.5, 1.5, 2.5]}),
                {},
                invalid_sampler,
                42,
                "accuracy",
                "max",
            )

    @patch.dict(os.environ, {"WANDB_API_KEY": "test_api_key"})
    def test_load_wandb_api_key(self):
        api_key = _load_wandb_api_key()
        self.assertEqual(api_key, "test_api_key")

    @patch("ritme.tune_models.os.getenv", return_value=None)
    def test_load_wandb_api_key_missing(self, mock_getenv):
        with self.assertRaisesRegex(ValueError, "No WANDB_API_KEY found in .env file."):
            _load_wandb_api_key()

    @patch.dict(os.environ, {"WANDB_ENTITY": "test_entity"})
    def test_load_wandb_entity(self):
        entity = _load_wandb_entity()
        self.assertEqual(entity, "test_entity")

    @patch("ritme.tune_models.os.getenv", return_value=None)
    def test_load_wandb_entity_missing(self, mock_getenv):
        with self.assertRaisesRegex(ValueError, "No WANDB_ENTITY found in .env file."):
            _load_wandb_entity()

    def test_define_callbacks_mlflow(self):
        tracking_uri = "sqlite:///tmp/mlflow.db"
        exp_name = "test_exp"
        experiment_tag = "test_tag"

        callbacks = _define_callbacks(tracking_uri, exp_name, experiment_tag)

        self.assertEqual(len(callbacks), 1)
        self.assertIsInstance(callbacks[0], _SafeMLflowLoggerCallback)
        self.assertIsInstance(callbacks[0], MLflowLoggerCallback)
        self.assertEqual(callbacks[0].tracking_uri, tracking_uri)
        self.assertEqual(callbacks[0].experiment_name, exp_name)
        self.assertEqual(callbacks[0].tags, {"experiment_tag": experiment_tag})

    def test_safe_mlflow_log_trial_end_skips_unknown_trial(self):
        # Trials whose actor died before log_trial_start should not raise.
        cb = _SafeMLflowLoggerCallback(
            tracking_uri="sqlite:///tmp/mlflow.db",
            experiment_name="test_exp",
            tags={"experiment_tag": "test_tag"},
        )
        cb.mlflow_util = MagicMock()
        cb._trial_runs = {}
        unknown_trial = MagicMock()

        cb.log_trial_end(unknown_trial, failed=True)

        cb.mlflow_util.end_run.assert_not_called()

    def test_safe_mlflow_log_trial_end_delegates_known_trial(self):
        # Trials that did start must still be finalized via the parent impl.
        cb = _SafeMLflowLoggerCallback(
            tracking_uri="sqlite:///tmp/mlflow.db",
            experiment_name="test_exp",
            tags={"experiment_tag": "test_tag"},
        )
        known_trial = MagicMock()
        cb._trial_runs = {known_trial: "run-123"}

        with patch.object(MLflowLoggerCallback, "log_trial_end") as mock_super_end:
            cb.log_trial_end(known_trial, failed=True)

        mock_super_end.assert_called_once_with(known_trial, failed=True)

    @patch("ritme.tune_models._load_wandb_api_key")
    @patch("ritme.tune_models._load_wandb_entity")
    def test_define_callbacks_wandb(self, mock_load_entity, mock_load_api_key):
        mock_load_api_key.return_value = "test_api_key"
        mock_load_entity.return_value = "test_entity"
        tracking_uri = "wandb"
        exp_name = "test_exp"
        experiment_tag = "test_tag"

        callbacks = _define_callbacks(tracking_uri, exp_name, experiment_tag)

        self.assertEqual(len(callbacks), 1)
        self.assertIsInstance(callbacks[0], WandbLoggerCallback)
        self.assertEqual(callbacks[0].api_key, "test_api_key")
        self.assertEqual(callbacks[0].kwargs["entity"], "test_entity")
        self.assertEqual(callbacks[0].project, experiment_tag)
        self.assertEqual(callbacks[0].kwargs["tags"], {experiment_tag})
        mock_load_api_key.assert_called_once()
        mock_load_entity.assert_called_once()

    @patch("ritme.tune_models.print")
    def test_define_callbacks_invalid_uri(self, mock_print):
        callbacks = _define_callbacks("invalid_uri", "test_exp", "test_tag")

        self.assertEqual(len(callbacks), 0)
        mock_print.assert_called_once_with(
            "No valid tracking URI provided. Proceeding without logging callbacks."
        )


class TestMainTuneModels(unittest.TestCase):
    def setUp(self):
        # Common variables for all tests. The minimal train_val needs at
        # least one F-prefixed column so the search-space introspection in
        # _adaptive_n_startup_trials does not trip on the empty .str accessor.
        self.train_val = pd.DataFrame({"F0": [1.0, 2.0, 3.0], "F1": [0.5, 1.5, 2.5]})
        self.target = "target_column"
        self.host_id = "host_id_column"
        self.seed_data = 42
        self.seed_model = 42
        self.tax = pd.DataFrame()
        self.tree_phylo = skbio.TreeNode()
        self.path2exp = "/tmp/experiment"
        self.experiment_tag = "test_experiment_tag"
        self.time_budget_s = 5
        self.max_concurrent_trials = 2
        self.model_hyperparameters = {}
        self.mlflow_uri = "sqlite:///tmp/experiment/mlflow.db"

    @patch("ritme.tune_models.init")
    @patch("ritme.tune_models.ray.cluster_resources")
    @patch("ritme.tune_models.tune.Tuner")
    def test_run_trials_uses_passed_experiment_tag_for_callback(
        self, mock_tuner_class, mock_resources, mock_init
    ):
        # Regression test: experiment_tag must come from the caller, not from
        # os.path.basename(path2exp). path2exp here is a throwaway temp dir,
        # but the MLflow tag should still be the user-supplied experiment_tag.
        mock_context = MagicMock()
        mock_context.dashboard_url = "http://localhost:8265"
        mock_init.return_value = mock_context
        mock_resources.return_value = {}
        mock_tuner = MagicMock()
        mock_tuner.fit.return_value = MagicMock(spec=ResultGrid, num_errors=0)
        mock_tuner_class.return_value = mock_tuner

        temp_path2exp = "/tmp/tmp_throwaway_abc123"
        user_tag = "my_real_experiment"

        run_trials(
            tracking_uri=self.mlflow_uri,
            exp_name="linreg",
            trainable=MagicMock(),
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            path2exp=temp_path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=user_tag,
            fully_reproducible=False,
            model_hyperparameters=self.model_hyperparameters,
        )

        run_config = mock_tuner_class.call_args.kwargs["run_config"]
        mlflow_cb = run_config.callbacks[0]
        self.assertEqual(mlflow_cb.tags, {"experiment_tag": user_tag})
        self.assertNotEqual(
            mlflow_cb.tags["experiment_tag"], os.path.basename(temp_path2exp)
        )

    @patch("ritme.tune_models.init")
    @patch("ritme.tune_models.ray.cluster_resources")
    @patch("ritme.tune_models.tune.Tuner")
    def test_run_trials_not_reproducible(
        self, mock_tuner_class, mock_resources, mock_init
    ):
        mock_context = MagicMock()
        mock_context.dashboard_url = "http://localhost:8265"
        mock_init.return_value = mock_context

        mock_resources.return_value = {}

        mock_tuner = MagicMock()
        mock_tuner.fit.return_value = MagicMock(spec=ResultGrid, num_errors=0)
        mock_tuner_class.return_value = mock_tuner

        trainable = MagicMock()

        result = run_trials(
            tracking_uri=self.mlflow_uri,
            exp_name="linreg",
            trainable=trainable,
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            path2exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            fully_reproducible=False,
            model_hyperparameters=self.model_hyperparameters,
        )

        # Assertions
        mock_init.assert_called_once()
        mock_tuner_class.assert_called_once()
        mock_tuner.fit.assert_called_once()
        self.assertIsInstance(result, ResultGrid)

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials(self, mock_run_trials):
        mock_result = MagicMock(spec=ResultGrid)
        mock_run_trials.return_value = mock_result

        model_types = ["xgb", "nn_reg"]
        results = run_all_trials(
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            model_types=model_types,
            model_hyperparameters=self.model_hyperparameters,
        )

        # Assertions
        self.assertEqual(len(results), len(model_types))
        for model in model_types:
            self.assertIn(model, results)
            self.assertEqual(results[model], mock_result)
        self.assertEqual(mock_run_trials.call_count, len(model_types))

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_remove_trac(self, mock_run_trials):
        mock_result = MagicMock(spec=ResultGrid)
        mock_run_trials.return_value = mock_result

        tax = None
        tree_phylo = None
        model_types = ["rf", "trac"]
        results = run_all_trials(
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=tax,
            tree_phylo=tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            model_types=model_types,
            model_hyperparameters=self.model_hyperparameters,
        )

        self.assertNotIn("trac", results)
        self.assertIn("rf", results)

        # Since 'trac' is removed, run_trials should be called once only for
        # 'rf' model
        mock_run_trials.assert_called_once_with(
            self.mlflow_uri,
            "rf",
            MODEL_TRAINABLES["rf"],
            self.train_val,
            self.target,
            self.host_id,
            None,
            self.seed_data,
            self.seed_model,
            tax,
            tree_phylo,
            self.path2exp,
            self.time_budget_s,
            self.max_concurrent_trials,
            self.experiment_tag,
            fully_reproducible=False,
            model_hyperparameters={"data_enrich_with": None},
            optuna_searchspace_sampler="TPESampler",
            task_type="regression",
            k_folds=1,
            nn_corn_max_levels=20,
        )

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_remove_trac_due_to_snapshots(self, mock_run_trials):
        # create dataframe with snapshot columns
        self.train_val = pd.DataFrame(
            {
                "F1": [0.1, 0.2],
                "F2": [0.3, 0.4],
                "F1__t-1": [0.05, 0.15],
                "F2__t-1": [0.25, 0.35],
                "meta": [1, 2],
                self.target: [0.5, 0.6],
                self.host_id: ["a", "b"],
            }
        )
        mock_result = MagicMock(spec=ResultGrid)
        mock_run_trials.return_value = mock_result
        model_types = ["xgb", "trac"]
        results = run_all_trials(
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            model_types=model_types,
            model_hyperparameters=self.model_hyperparameters,
        )
        self.assertIn("xgb", results)
        self.assertNotIn("trac", results)

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_nan_snapshots_rejects_non_xgb(self, mock_run_trials):
        # dataframe with NaN in snapshot features
        self.train_val = pd.DataFrame(
            {
                "F1": [0.1, 0.2],
                "F2": [0.3, 0.4],
                "F1__t-1": [np.nan, 0.15],
                "F2__t-1": [0.25, np.nan],
                "meta": [1, 2],
                self.target: [0.5, 0.6],
                self.host_id: ["a", "b"],
            }
        )
        mock_result = MagicMock(spec=ResultGrid)
        mock_run_trials.return_value = mock_result
        model_types = ["xgb", "rf", "trac"]
        with self.assertRaisesRegex(ValueError, r"NaNs in snapshot features"):
            run_all_trials(
                train_val=self.train_val,
                target=self.target,
                host_id=self.host_id,
                stratify_by=None,
                seed_data=self.seed_data,
                seed_model=self.seed_model,
                tax=self.tax,
                tree_phylo=self.tree_phylo,
                mlflow_uri=self.mlflow_uri,
                path_exp=self.path2exp,
                time_budget_s=self.time_budget_s,
                max_concurrent_trials=self.max_concurrent_trials,
                experiment_tag=self.experiment_tag,
                model_types=model_types,
                model_hyperparameters=self.model_hyperparameters,
            )
        mock_run_trials.assert_not_called()

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_nan_snapshots_xgb_only_ok(self, mock_run_trials):
        # dataframe with NaN in snapshot features; only xgb requested
        self.train_val = pd.DataFrame(
            {
                "F1": [0.1, 0.2],
                "F2": [0.3, 0.4],
                "F1__t-1": [np.nan, 0.15],
                "F2__t-1": [0.25, np.nan],
                "meta": [1, 2],
                self.target: [0.5, 0.6],
                self.host_id: ["a", "b"],
            }
        )
        mock_result = MagicMock(spec=ResultGrid)
        mock_run_trials.return_value = mock_result
        results = run_all_trials(
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            model_types=["xgb"],
            model_hyperparameters=self.model_hyperparameters,
        )
        self.assertEqual(list(results.keys()), ["xgb"])

    @patch("ritme.tune_models.init")
    @patch("ritme.tune_models.ray.cluster_resources")
    @patch("ritme.tune_models.tune.Tuner")
    def test_run_trials_classification(
        self, mock_tuner_class, mock_resources, mock_init
    ):
        mock_context = MagicMock()
        mock_context.dashboard_url = "http://localhost:8265"
        mock_init.return_value = mock_context

        mock_resources.return_value = {}

        mock_tuner = MagicMock()
        mock_tuner.fit.return_value = MagicMock(spec=ResultGrid, num_errors=0)
        mock_tuner_class.return_value = mock_tuner

        trainable = MagicMock()

        result = run_trials(
            tracking_uri=self.mlflow_uri,
            exp_name="logreg",
            trainable=trainable,
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            path2exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            fully_reproducible=False,
            model_hyperparameters=self.model_hyperparameters,
            task_type="classification",
        )

        # Verify Tuner was configured with classification metric/mode
        tuner_call_kwargs = mock_tuner_class.call_args
        tune_config = tuner_call_kwargs.kwargs["tune_config"]
        self.assertEqual(tune_config.metric, "roc_auc_macro_ovr_val")
        self.assertEqual(tune_config.mode, "max")

        # Verify checkpoint config also uses classification metric
        run_config = tuner_call_kwargs.kwargs["run_config"]
        self.assertEqual(
            run_config.checkpoint_config.checkpoint_score_attribute,
            "roc_auc_macro_ovr_val",
        )
        self.assertEqual(run_config.checkpoint_config.checkpoint_score_order, "max")

        self.assertIsInstance(result, ResultGrid)

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_classification(self, mock_run_trials):
        mock_result = MagicMock(spec=ResultGrid)
        mock_run_trials.return_value = mock_result

        model_types = ["logreg", "rf_class"]
        results = run_all_trials(
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            model_types=model_types,
            model_hyperparameters=self.model_hyperparameters,
            task_type="classification",
        )

        # Assertions
        self.assertEqual(len(results), len(model_types))
        for model in model_types:
            self.assertIn(model, results)
            self.assertEqual(results[model], mock_result)
        self.assertEqual(mock_run_trials.call_count, len(model_types))

        # Verify task_type="classification" was passed through to run_trials
        for call in mock_run_trials.call_args_list:
            self.assertEqual(call.kwargs.get("task_type"), "classification")

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_nan_snapshots_rejects_non_xgb_class(self, mock_run_trials):
        # dataframe with NaN in snapshot features
        self.train_val = pd.DataFrame(
            {
                "F1": [0.1, 0.2],
                "F2": [0.3, 0.4],
                "F1__t-1": [np.nan, 0.15],
                "F2__t-1": [0.25, np.nan],
                "meta": [1, 2],
                self.target: [0.5, 0.6],
                self.host_id: ["a", "b"],
            }
        )
        model_types = ["xgb_class", "rf_class", "logreg"]
        with self.assertRaisesRegex(ValueError, r"NaNs in snapshot features"):
            run_all_trials(
                train_val=self.train_val,
                target=self.target,
                host_id=self.host_id,
                stratify_by=None,
                seed_data=self.seed_data,
                seed_model=self.seed_model,
                tax=self.tax,
                tree_phylo=self.tree_phylo,
                mlflow_uri=self.mlflow_uri,
                path_exp=self.path2exp,
                time_budget_s=self.time_budget_s,
                max_concurrent_trials=self.max_concurrent_trials,
                experiment_tag=self.experiment_tag,
                model_types=model_types,
                model_hyperparameters=self.model_hyperparameters,
                task_type="classification",
            )
        mock_run_trials.assert_not_called()

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_nan_snapshots_xgb_class_only_ok(self, mock_run_trials):
        # dataframe with NaN in snapshot features; only xgb_class requested
        self.train_val = pd.DataFrame(
            {
                "F1": [0.1, 0.2],
                "F2": [0.3, 0.4],
                "F1__t-1": [np.nan, 0.15],
                "F2__t-1": [0.25, np.nan],
                "meta": [1, 2],
                self.target: [0.5, 0.6],
                self.host_id: ["a", "b"],
            }
        )
        mock_result = MagicMock(spec=ResultGrid)
        mock_run_trials.return_value = mock_result
        results = run_all_trials(
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            model_types=["xgb_class"],
            model_hyperparameters=self.model_hyperparameters,
            task_type="classification",
        )
        self.assertEqual(list(results.keys()), ["xgb_class"])

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_classification_hparams_fallback_rf_class(
        self, mock_run_trials
    ):
        mock_result = MagicMock(spec=ResultGrid)
        mock_run_trials.return_value = mock_result

        # Provide hparams under "rf" key only - rf_class should fall back to it
        model_hyperparameters = {
            "rf": {"n_estimators": {"min": 10, "max": 200}},
        }
        model_types = ["rf_class"]
        run_all_trials(
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            model_types=model_types,
            model_hyperparameters=model_hyperparameters,
            task_type="classification",
        )

        # Verify run_trials received the "rf" hparams for rf_class
        call_kwargs = mock_run_trials.call_args
        fallback = call_kwargs[0][15] if len(call_kwargs[0]) > 15 else None
        passed_hparams = call_kwargs.kwargs.get("model_hyperparameters", fallback)
        self.assertIn("n_estimators", passed_hparams)
        self.assertEqual(passed_hparams["n_estimators"], {"min": 10, "max": 200})

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_classification_hparams_fallback_xgb_class(
        self, mock_run_trials
    ):
        mock_result = MagicMock(spec=ResultGrid)
        mock_run_trials.return_value = mock_result

        # Provide hparams under "xgb" key only - xgb_class should fall back to it
        model_hyperparameters = {
            "xgb": {"n_estimators": {"min": 50, "max": 500}},
        }
        model_types = ["xgb_class"]
        run_all_trials(
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            model_types=model_types,
            model_hyperparameters=model_hyperparameters,
            task_type="classification",
        )

        # Verify run_trials received the "xgb" hparams for xgb_class
        call_kwargs = mock_run_trials.call_args
        fallback = call_kwargs[0][15] if len(call_kwargs[0]) > 15 else None
        passed_hparams = call_kwargs.kwargs.get("model_hyperparameters", fallback)
        self.assertIn("n_estimators", passed_hparams)
        self.assertEqual(passed_hparams["n_estimators"], {"min": 50, "max": 500})

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_nn_corn_level_cap_rejects_wide_target(
        self, mock_run_trials
    ):
        """``run_all_trials`` validates the ``nn_corn`` level cap up-front by
        rounding the full ``train_val[target]`` before any trial is launched.
        With 30 distinct rounded levels and a cap of 5 the cap check must
        raise ``ValueError`` and ``run_trials`` must never be invoked.
        """
        n_rows = 30
        train_val = pd.DataFrame(
            {
                "F0": np.linspace(0.0, 1.0, n_rows),
                "F1": np.linspace(1.0, 2.0, n_rows),
                self.target: np.arange(n_rows, dtype=float),
                self.host_id: [f"h{i}" for i in range(n_rows)],
            }
        )

        with self.assertRaisesRegex(ValueError, r"nn_corn_max_levels"):
            run_all_trials(
                train_val=train_val,
                target=self.target,
                host_id=self.host_id,
                stratify_by=None,
                seed_data=self.seed_data,
                seed_model=self.seed_model,
                tax=self.tax,
                tree_phylo=self.tree_phylo,
                mlflow_uri=self.mlflow_uri,
                path_exp=self.path2exp,
                time_budget_s=self.time_budget_s,
                max_concurrent_trials=self.max_concurrent_trials,
                experiment_tag=self.experiment_tag,
                model_types=["nn_corn"],
                model_hyperparameters=self.model_hyperparameters,
                nn_corn_max_levels=5,
            )
        mock_run_trials.assert_not_called()

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_nn_corn_level_cap_allows_narrow_target(
        self, mock_run_trials
    ):
        """When the rounded-target level count is at or below
        ``nn_corn_max_levels`` the up-front check passes and the trial is
        launched as usual.
        """
        mock_run_trials.return_value = MagicMock(spec=ResultGrid)
        n_rows = 30
        # 3 distinct rounded levels well below the cap of 5.
        narrow_target = np.tile([0.0, 1.0, 2.0], n_rows // 3)[:n_rows]
        train_val = pd.DataFrame(
            {
                "F0": np.linspace(0.0, 1.0, n_rows),
                "F1": np.linspace(1.0, 2.0, n_rows),
                self.target: narrow_target,
                self.host_id: [f"h{i}" for i in range(n_rows)],
            }
        )

        results = run_all_trials(
            train_val=train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            model_types=["nn_corn"],
            model_hyperparameters=self.model_hyperparameters,
            nn_corn_max_levels=5,
        )
        self.assertIn("nn_corn", results)
        # nn_corn_max_levels must be forwarded to run_trials so the trainable
        # safety net inside train_nn sees the same cap as the up-front check.
        forwarded = mock_run_trials.call_args.kwargs.get("nn_corn_max_levels")
        self.assertEqual(forwarded, 5)

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_nn_corn_level_cap_boundary_exactly_at_cap(
        self, mock_run_trials
    ):
        """Boundary: ``n_levels == nn_corn_max_levels`` must pass (`>` check,
        not `>=`). 5 distinct rounded levels with cap 5 should launch.
        """
        mock_run_trials.return_value = MagicMock(spec=ResultGrid)
        n_rows = 30
        # 5 distinct rounded levels exactly equal to the cap.
        target = np.tile([0.0, 1.0, 2.0, 3.0, 4.0], n_rows // 5)[:n_rows]
        train_val = pd.DataFrame(
            {
                "F0": np.linspace(0.0, 1.0, n_rows),
                "F1": np.linspace(1.0, 2.0, n_rows),
                self.target: target,
                self.host_id: [f"h{i}" for i in range(n_rows)],
            }
        )

        run_all_trials(
            train_val=train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            model_types=["nn_corn"],
            model_hyperparameters=self.model_hyperparameters,
            nn_corn_max_levels=5,
        )
        mock_run_trials.assert_called_once()

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_nn_corn_level_cap_boundary_one_over_cap(
        self, mock_run_trials
    ):
        """Boundary: ``n_levels == nn_corn_max_levels + 1`` must raise."""
        n_rows = 30
        # 6 distinct rounded levels, one over the cap of 5.
        target = np.tile([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], n_rows // 6)[:n_rows]
        train_val = pd.DataFrame(
            {
                "F0": np.linspace(0.0, 1.0, n_rows),
                "F1": np.linspace(1.0, 2.0, n_rows),
                self.target: target,
                self.host_id: [f"h{i}" for i in range(n_rows)],
            }
        )

        with self.assertRaisesRegex(ValueError, r"nn_corn_max_levels"):
            run_all_trials(
                train_val=train_val,
                target=self.target,
                host_id=self.host_id,
                stratify_by=None,
                seed_data=self.seed_data,
                seed_model=self.seed_model,
                tax=self.tax,
                tree_phylo=self.tree_phylo,
                mlflow_uri=self.mlflow_uri,
                path_exp=self.path2exp,
                time_budget_s=self.time_budget_s,
                max_concurrent_trials=self.max_concurrent_trials,
                experiment_tag=self.experiment_tag,
                model_types=["nn_corn"],
                model_hyperparameters=self.model_hyperparameters,
                nn_corn_max_levels=5,
            )
        mock_run_trials.assert_not_called()

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_nn_corn_rejects_non_numeric_target(self, mock_run_trials):
        """Non-numeric target must raise a clear ``ValueError`` instead of a
        cryptic ``TypeError`` from ``np.round`` on an object array.
        """
        n_rows = 12
        train_val = pd.DataFrame(
            {
                "F0": np.linspace(0.0, 1.0, n_rows),
                "F1": np.linspace(1.0, 2.0, n_rows),
                self.target: ["low", "medium", "high"] * (n_rows // 3),
                self.host_id: [f"h{i}" for i in range(n_rows)],
            }
        )

        with self.assertRaisesRegex(ValueError, r"numeric target"):
            run_all_trials(
                train_val=train_val,
                target=self.target,
                host_id=self.host_id,
                stratify_by=None,
                seed_data=self.seed_data,
                seed_model=self.seed_model,
                tax=self.tax,
                tree_phylo=self.tree_phylo,
                mlflow_uri=self.mlflow_uri,
                path_exp=self.path2exp,
                time_budget_s=self.time_budget_s,
                max_concurrent_trials=self.max_concurrent_trials,
                experiment_tag=self.experiment_tag,
                model_types=["nn_corn"],
                model_hyperparameters=self.model_hyperparameters,
            )
        mock_run_trials.assert_not_called()

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_nn_corn_rejects_nan_target(self, mock_run_trials):
        """NaN-bearing target must raise rather than silently coerce NaN to 0
        via ``np.round(...).astype(int)``.
        """
        n_rows = 12
        target = np.array([0.0, 1.0, np.nan, 2.0] * (n_rows // 4), dtype=float)
        train_val = pd.DataFrame(
            {
                "F0": np.linspace(0.0, 1.0, n_rows),
                "F1": np.linspace(1.0, 2.0, n_rows),
                self.target: target,
                self.host_id: [f"h{i}" for i in range(n_rows)],
            }
        )

        with self.assertRaisesRegex(ValueError, r"NaN"):
            run_all_trials(
                train_val=train_val,
                target=self.target,
                host_id=self.host_id,
                stratify_by=None,
                seed_data=self.seed_data,
                seed_model=self.seed_model,
                tax=self.tax,
                tree_phylo=self.tree_phylo,
                mlflow_uri=self.mlflow_uri,
                path_exp=self.path2exp,
                time_budget_s=self.time_budget_s,
                max_concurrent_trials=self.max_concurrent_trials,
                experiment_tag=self.experiment_tag,
                model_types=["nn_corn"],
                model_hyperparameters=self.model_hyperparameters,
            )
        mock_run_trials.assert_not_called()

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_rejects_invalid_cap_values(self, mock_run_trials):
        """``nn_corn_max_levels`` must be an ``int`` >= 2. Garbage values
        (zero, negative, string) raise a clear ``ValueError`` rather than
        silently disabling the guard or rejecting every run.
        """
        n_rows = 12
        train_val = pd.DataFrame(
            {
                "F0": np.linspace(0.0, 1.0, n_rows),
                "F1": np.linspace(1.0, 2.0, n_rows),
                self.target: [0.0, 1.0, 2.0] * (n_rows // 3),
                self.host_id: [f"h{i}" for i in range(n_rows)],
            }
        )

        for invalid_cap in (0, 1, -5, "20", None, 1.5):
            with self.subTest(invalid_cap=invalid_cap):
                with self.assertRaises(ValueError):
                    run_all_trials(
                        train_val=train_val,
                        target=self.target,
                        host_id=self.host_id,
                        stratify_by=None,
                        seed_data=self.seed_data,
                        seed_model=self.seed_model,
                        tax=self.tax,
                        tree_phylo=self.tree_phylo,
                        mlflow_uri=self.mlflow_uri,
                        path_exp=self.path2exp,
                        time_budget_s=self.time_budget_s,
                        max_concurrent_trials=self.max_concurrent_trials,
                        experiment_tag=self.experiment_tag,
                        model_types=["nn_corn"],
                        model_hyperparameters=self.model_hyperparameters,
                        nn_corn_max_levels=invalid_cap,
                    )
        mock_run_trials.assert_not_called()


class TestAdaptiveNStartupTrials(unittest.TestCase):
    """The pre-K-fold default of 1000 startup trials consumed nearly the
    entire time budget on most ritme runs. The adaptive helper sizes the
    random-sampling phase from each model's effective search-space dim,
    counted by introspecting the actual search space along its longest
    conditional path."""

    def setUp(self):
        # Minimal but non-empty train_val: the search space's threshold-bounds
        # path needs at least one F-prefixed feature with non-zero variance
        # in case the recording trial wanders there. Three rows is enough.
        self.train_val = pd.DataFrame(
            {
                "F0": [1.0, 2.0, 3.0],
                "F1": [0.1, 0.5, 0.9],
                "F2": [4.0, 5.0, 6.0],
                "md0": [0, 1, 0],
            }
        )
        self.tax = None

    def _startup(self, model_type: str, hparams: dict | None = None) -> int:
        return _adaptive_n_startup_trials(
            model_type, self.train_val, self.tax, hparams or {}
        )

    def test_floor_for_low_dim_models(self):
        # TRAC has only `lambda` -> 1 dim, falls back to the floor.
        self.assertEqual(self._startup("trac"), 20)

    def test_per_model_defaults_match_search_space_long_path(self):
        # 5 data-eng dims + alpha + l1_ratio -> 7 -> 5 * 7 = 35
        self.assertEqual(self._startup("linreg"), 35)
        # 5 + (C, penalty, l1_ratio) -> 8 -> 40
        self.assertEqual(self._startup("logreg"), 40)
        # 5 + 8 RF model dims -> 13 -> 65
        self.assertEqual(self._startup("rf"), 65)
        self.assertEqual(self._startup("rf_class"), 65)
        # 5 + 10 XGB model dims -> 15 -> 75
        self.assertEqual(self._startup("xgb"), 75)
        self.assertEqual(self._startup("xgb_class"), 75)

    def test_nn_uses_max_n_hidden_layers(self):
        # Default range [1, 30]: longest path has 30 hidden layers
        # -> 5 (data eng) + 8 (fixed nn params) + 30 (per-layer widths) = 43
        # -> 5 * 43 = 215
        self.assertEqual(self._startup("nn_reg"), 215)
        # User-supplied tighter range [1, 5]: longest path has 5 hidden layers
        # -> 5 + 8 + 5 = 18 -> 5 * 18 = 90
        self.assertEqual(
            self._startup("nn_class", {"n_hidden_layers": {"min": 1, "max": 5}}),
            90,
        )
        # min == max collapses to a single layer count: 5 + 8 + 4 = 17 -> 85
        self.assertEqual(
            self._startup("nn_corn", {"n_hidden_layers": {"min": 4, "max": 4}}),
            85,
        )

    def test_unknown_model_type_raises(self):
        # The recording-trial path delegates to ss.get_search_space, which
        # raises ValueError on unknown model types. This is desired: silent
        # fallbacks would mask configuration mistakes.
        from ritme.model_space import static_searchspace  # noqa: F401

        with self.assertRaises(ValueError):
            self._startup("unknown_model")


class TestRecordingTrialSteering(unittest.TestCase):
    """The adaptive ``n_startup_trials`` is sized from the longest conditional
    path of each model's search space, introspected via ``_RecordingTrial``.
    The recording trial's correctness rests on two steering invariants:
    categorical picks that trigger dependent branches (``data_selection`` ->
    ``"abundance_ith"``, ``penalty`` -> ``"elasticnet"``) and returning
    ``high`` for ``n_hidden_layers`` so every per-layer width fires. If a
    search-space parameter is renamed or a conditional branch is restructured,
    the steering can silently undercount dims. These tests lock in the
    presence of the specific parameter names that prove the steering worked,
    so drift fails loudly here rather than silently breaking ``n_startup``.
    """

    def setUp(self):
        # Minimal but non-empty train_val: a few F-prefixed feature columns
        # plus one md column. Matches the fixture used in
        # TestAdaptiveNStartupTrials so the recording-trial path is exercised
        # under realistic-shaped data.
        self.train_val = pd.DataFrame(
            {
                "F0": [1.0, 2.0, 3.0],
                "F1": [0.5, 1.5, 2.5],
                "F2": [0.1, 0.2, 0.3],
                "md0": [0, 1, 0],
            }
        )
        self.tax = None

    def test_linreg_records_data_selection_i_dependent_suggestion(self):
        trial = _RecordingTrial()
        ss.get_search_space(
            trial,
            model_type="linreg",
            tax=self.tax,
            train_val=self.train_val,
            model_hyperparameters={},
        )
        self.assertIn("data_selection_i", trial.params)

    def test_logreg_records_l1_ratio_under_elasticnet_penalty(self):
        trial = _RecordingTrial()
        ss.get_search_space(
            trial,
            model_type="logreg",
            tax=self.tax,
            train_val=self.train_val,
            model_hyperparameters={},
        )
        self.assertIn("l1_ratio", trial.params)

    def test_rf_records_data_selection_dependent_suggestion(self):
        trial = _RecordingTrial()
        ss.get_search_space(
            trial,
            model_type="rf",
            tax=self.tax,
            train_val=self.train_val,
            model_hyperparameters={},
        )
        self.assertIn("data_selection_i", trial.params)

    def test_xgb_records_data_selection_dependent_suggestion(self):
        trial = _RecordingTrial()
        ss.get_search_space(
            trial,
            model_type="xgb",
            tax=self.tax,
            train_val=self.train_val,
            model_hyperparameters={},
        )
        self.assertIn("data_selection_i", trial.params)

    def test_nn_reg_records_all_per_layer_widths_at_default_range(self):
        trial = _RecordingTrial()
        ss.get_search_space(
            trial,
            model_type="nn_reg",
            tax=self.tax,
            train_val=self.train_val,
            model_hyperparameters={},
        )
        expected = {f"n_units_hl{i}" for i in range(30)}
        recorded = {k for k in trial.params if k.startswith("n_units_hl")}
        self.assertEqual(recorded, expected)

    def test_nn_reg_records_widths_for_custom_range(self):
        trial = _RecordingTrial()
        ss.get_search_space(
            trial,
            model_type="nn_reg",
            tax=self.tax,
            train_val=self.train_val,
            model_hyperparameters={"n_hidden_layers": {"min": 1, "max": 5}},
        )
        expected = {f"n_units_hl{i}" for i in range(5)}
        recorded = {k for k in trial.params if k.startswith("n_units_hl")}
        self.assertEqual(recorded, expected)

    def test_nn_reg_records_widths_for_collapsed_range(self):
        trial = _RecordingTrial()
        ss.get_search_space(
            trial,
            model_type="nn_reg",
            tax=self.tax,
            train_val=self.train_val,
            model_hyperparameters={"n_hidden_layers": {"min": 4, "max": 4}},
        )
        expected = {f"n_units_hl{i}" for i in range(4)}
        recorded = {k for k in trial.params if k.startswith("n_units_hl")}
        self.assertEqual(recorded, expected)

    def test_trac_search_space_has_only_lambda(self):
        trial = _RecordingTrial()
        ss.get_search_space(
            trial,
            model_type="trac",
            tax=self.tax,
            train_val=self.train_val,
            model_hyperparameters={},
        )
        self.assertEqual(set(trial.params.keys()), {"lambda"})


if __name__ == "__main__":
    unittest.main()
