# test_script.py

import os
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import skbio
from ray.tune import ResultGrid

from q2_ritme.tune_models import (
    MODEL_TRAINABLES,
    check_for_errors_in_trials,
    get_resources,
    get_slurm_resource,
    run_all_trials,
    run_trials,
)


class TestTuneModels(unittest.TestCase):
    @patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "4"})
    def test_get_slurm_resource_present(self):
        # Test when the environment variable is present
        self.assertEqual(get_slurm_resource("SLURM_CPUS_PER_TASK"), 4)

    @patch.dict(os.environ, {}, clear=True)
    def test_get_slurm_resource_absent(self):
        # Test when the environment variable is absent
        self.assertEqual(get_slurm_resource("SLURM_CPUS_PER_TASK"), 0)

    @patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "invalid"})
    def test_get_slurm_resource_invalid(self):
        # Test when the environment variable is invalid
        self.assertEqual(get_slurm_resource("SLURM_CPUS_PER_TASK"), 0)

    def test_check_for_errors_in_trials_no_errors(self):
        # Mock ResultGrid with no errors
        mock_result = MagicMock(spec=ResultGrid)
        mock_result.num_errors = 0
        # Should not raise an exception
        check_for_errors_in_trials(mock_result)

    def test_check_for_errors_in_trials_with_errors(self):
        # Mock ResultGrid with errors
        mock_result = MagicMock(spec=ResultGrid)
        mock_result.num_errors = 1
        with self.assertRaises(RuntimeError):
            check_for_errors_in_trials(mock_result)

    @patch("q2_ritme.tune_models.get_slurm_resource")
    def test_get_resources(self, mock_get_slurm_resource):
        # Mock get_slurm_resource to return predefined values
        mock_get_slurm_resource.side_effect = [8, 2]
        resources = get_resources(2)

        expected_resources = {"cpu": 4, "gpu": 1}
        self.assertEqual(resources, expected_resources)

        # Check that get_slurm_resource was called with correct arguments
        mock_get_slurm_resource.assert_any_call("SLURM_CPUS_PER_TASK", 1)
        mock_get_slurm_resource.assert_any_call("SLURM_GPUS_PER_TASK", 0)

    @patch("q2_ritme.tune_models.init")
    @patch("q2_ritme.tune_models.ray.cluster_resources")
    @patch("q2_ritme.tune_models.tune.Tuner")
    def test_run_trials_not_reproducible(
        self, mock_tuner_class, mock_resources, mock_init
    ):
        # Mocking Ray init and Tuner
        mock_context = MagicMock()
        mock_context.dashboard_url = "http://localhost:8265"
        mock_init.return_value = mock_context

        mock_resources.return_value = {}

        mock_tuner = MagicMock()
        mock_tuner.fit.return_value = MagicMock(spec=ResultGrid, num_errors=0)
        mock_tuner_class.return_value = mock_tuner

        # Mocking other dependencies
        trainable = MagicMock()
        train_val = pd.DataFrame()
        target = "target_column"
        host_id = "host_id_column"
        seed_data = 42
        seed_model = 42
        tax = pd.DataFrame()
        tree_phylo = skbio.TreeNode()
        path2exp = "/tmp/experiment"
        num_trials = 5
        max_concurrent_trials = 2
        model_hyperparameters = {}

        result = run_trials(
            tracking_uri="mlruns",
            exp_name="test_experiment",
            trainable=trainable,
            test_mode=False,
            train_val=train_val,
            target=target,
            host_id=host_id,
            seed_data=seed_data,
            seed_model=seed_model,
            tax=tax,
            tree_phylo=tree_phylo,
            path2exp=path2exp,
            num_trials=num_trials,
            max_concurrent_trials=max_concurrent_trials,
            fully_reproducible=False,
            model_hyperparameters=model_hyperparameters,
        )

        # Assertions
        mock_init.assert_called_once()
        mock_tuner_class.assert_called_once()
        mock_tuner.fit.assert_called_once()
        self.assertIsInstance(result, ResultGrid)

    @patch("q2_ritme.tune_models.run_trials")
    def test_run_all_trials(self, mock_run_trials):
        # Mocking run_trials
        mock_result = MagicMock(spec=ResultGrid)
        mock_run_trials.return_value = mock_result

        # Preparing input data
        train_val = pd.DataFrame()
        target = "target_column"
        host_id = "host_id_column"
        seed_data = 42
        seed_model = 42
        tax = pd.DataFrame()
        tree_phylo = skbio.TreeNode()
        mlflow_uri = "mlruns"
        path_exp = "/tmp/experiment"
        num_trials = 5
        max_concurrent_trials = 2
        model_types = ["xgb", "nn_reg"]
        model_hyperparameters = {}

        results = run_all_trials(
            train_val=train_val,
            target=target,
            host_id=host_id,
            seed_data=seed_data,
            seed_model=seed_model,
            tax=tax,
            tree_phylo=tree_phylo,
            mlflow_uri=mlflow_uri,
            path_exp=path_exp,
            num_trials=num_trials,
            max_concurrent_trials=max_concurrent_trials,
            model_types=model_types,
            model_hyperparameters=model_hyperparameters,
        )

        # Assertions
        self.assertEqual(len(results), len(model_types))
        for model in model_types:
            self.assertIn(model, results)
            self.assertEqual(results[model], mock_result)
        self.assertEqual(mock_run_trials.call_count, len(model_types))

    @patch("q2_ritme.tune_models.run_trials")
    def test_run_all_trials_remove_trac(self, mock_run_trials):
        # Test that 'trac' is removed when tax or tree_phylo is None
        # Mocking run_trials
        mock_result = MagicMock(spec=ResultGrid)
        mock_run_trials.return_value = mock_result

        # Preparing input data
        train_val = pd.DataFrame()
        target = "target_column"
        host_id = "host_id_column"
        seed_data = 42
        seed_model = 42
        tax = None  # No taxonomy provided
        tree_phylo = None  # No phylogeny provided
        mlflow_uri = "mlruns"
        path_exp = "/tmp/experiment"
        num_trials = 5
        max_concurrent_trials = 2
        model_types = ["rf", "trac"]
        model_hyperparameters = {}

        results = run_all_trials(
            train_val=train_val,
            target=target,
            host_id=host_id,
            seed_data=seed_data,
            seed_model=seed_model,
            tax=tax,
            tree_phylo=tree_phylo,
            mlflow_uri=mlflow_uri,
            path_exp=path_exp,
            num_trials=num_trials,
            max_concurrent_trials=max_concurrent_trials,
            model_types=model_types,
            model_hyperparameters=model_hyperparameters,
        )

        # 'trac' should be removed
        self.assertNotIn("trac", results)
        self.assertIn("rf", results)
        # Since 'trac' is removed, run_trials should be called once for 'rf' model
        mock_run_trials.assert_called_once_with(
            mlflow_uri,
            "rf",
            MODEL_TRAINABLES["rf"],
            False,
            train_val,
            target,
            host_id,
            seed_data,
            seed_model,
            tax,
            tree_phylo,
            path_exp,
            num_trials,
            max_concurrent_trials,
            fully_reproducible=False,
            model_hyperparameters={},
        )


if __name__ == "__main__":
    unittest.main()
