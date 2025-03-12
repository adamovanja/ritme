# test_script.py

import os
import unittest
from functools import partial
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import skbio
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune import ResultGrid
from ray.tune.schedulers import AsyncHyperBandScheduler, HyperBandScheduler
from ray.tune.search.optuna import OptunaSearch

from ritme.tune_models import (
    MODEL_TRAINABLES,
    _check_for_errors_in_trials,
    _define_callbacks,
    _define_scheduler,
    _define_search_algo,
    _get_resources,
    _get_slurm_resource,
    _load_wandb_api_key,
    _load_wandb_entity,
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

    def test_define_search_algo(self):
        mock_func_to_get_search_space = Mock()

        exp_name = "test_exp"
        tax = pd.DataFrame()
        model_hyperparameters = {}
        seed_model = 42
        metric = "accuracy"
        mode = "max"

        search_algo = _define_search_algo(
            mock_func_to_get_search_space,
            exp_name,
            tax,
            model_hyperparameters,
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
            model_hyperparameters=model_hyperparameters,
        )

        self.assertEqual(search_algo._seed, seed_model)
        self.assertEqual(search_algo._metric, metric)
        self.assertEqual(search_algo._mode, mode)

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

    @patch("ritme.tune_models.os.path.exists")
    @patch("ritme.tune_models.os.makedirs")
    def test_define_callbacks_mlflow(self, mock_makedirs, mock_exists):
        mock_exists.return_value = False
        tracking_uri = "mlruns"
        exp_name = "test_exp"
        experiment_tag = "test_tag"

        callbacks = _define_callbacks(tracking_uri, exp_name, experiment_tag)

        self.assertEqual(len(callbacks), 1)
        self.assertIsInstance(callbacks[0], MLflowLoggerCallback)
        self.assertEqual(callbacks[0].tracking_uri, tracking_uri)
        self.assertEqual(callbacks[0].experiment_name, exp_name)
        self.assertEqual(callbacks[0].tags, {"experiment_tag": experiment_tag})
        mock_exists.assert_called_once_with(tracking_uri)
        mock_makedirs.assert_called_once_with(tracking_uri)

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
        # Common variables for all tests
        self.train_val = pd.DataFrame()
        self.target = "target_column"
        self.host_id = "host_id_column"
        self.seed_data = 42
        self.seed_model = 42
        self.tax = pd.DataFrame()
        self.tree_phylo = skbio.TreeNode()
        self.path2exp = "/tmp/experiment"
        self.num_trials = 5
        self.max_concurrent_trials = 2
        self.model_hyperparameters = {}
        self.mlflow_uri = "mlruns"

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
            exp_name="test_experiment",
            trainable=trainable,
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            path2exp=self.path2exp,
            num_trials=self.num_trials,
            max_concurrent_trials=self.max_concurrent_trials,
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
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            num_trials=self.num_trials,
            max_concurrent_trials=self.max_concurrent_trials,
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
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=tax,
            tree_phylo=tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            num_trials=self.num_trials,
            max_concurrent_trials=self.max_concurrent_trials,
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
            self.seed_data,
            self.seed_model,
            tax,
            tree_phylo,
            self.path2exp,
            self.num_trials,
            self.max_concurrent_trials,
            fully_reproducible=False,
            model_hyperparameters={},
        )


if __name__ == "__main__":
    unittest.main()
