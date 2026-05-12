# test_find_best_model_config.py
import json
import os
import tempfile
import unittest
from io import StringIO
from unittest.mock import ANY, MagicMock, patch

import pandas as pd
import skbio
from pandas.testing import assert_frame_equal, assert_series_equal

from ritme.find_best_model_config import (
    _define_experiment_path,
    _define_model_tracker,
    _extract_mlflow_logs_to_csv,
    _load_experiment_config,
    _load_phylogeny,
    _load_taxonomy,
    _process_phylogeny,
    _process_taxonomy,
    _save_config,
    _verify_experiment_config,
    cli_find_best_model_config,
    find_best_model_config,
)


class TestFindBestModelConfig(unittest.TestCase):
    def setUp(self):
        super().setUp()
        # experiment config
        self.config = {
            "tracking_uri": "mlruns",
            "fully_reproducible": False,
            "experiment_tag": "test_experiment",
            "target": "target_column",
            "group_by_column": "group_column",
            "seed_data": 42,
            "seed_model": 42,
            "time_budget_s": 10,
            "max_cuncurrent_trials": 2,
            "ls_model_types": ["model1", "model2"],
            "model_hyperparameters": {},
        }
        # data
        current_dir = os.path.dirname(__file__)
        ft_w_md = pd.read_csv(
            os.path.join(current_dir, "data/example_feature_table.tsv"),
            sep="\t",
            index_col=0,
        )
        self.ft = ft_w_md.drop(columns=["md2"])
        # single-snapshot convention: t0 columns have no suffix
        self.ft_t0 = self.ft.copy()
        self.train_val = self.ft_t0.copy()
        self.train_val["target_column"] = [1, 2, 3, 4, 4, 4, 0, 6, 5, 8]
        self.train_val["stratify_column"] = [
            "a",
            "a",
            "a",
            "a",
            "b",
            "b",
            "b",
            "b",
            "b",
            "b",
        ]

        # taxonomy
        self.tax_renamed = pd.read_csv(
            os.path.join(current_dir, "data/example_taxonomy.tsv"),
            sep="\t",
            index_col=0,
        )
        self.tax_renamed.index.name = "Feature ID"

        self.tax = self.tax_renamed.copy()
        self.tax.index = self.tax.index.map(lambda x: int(x.replace("F", "")))

        # phylogeny
        # this tree has one feature more than the feature table (namely 7) -
        # will be filtered out by _process_phylogeny
        self.tree_str = (
            "(((1:0.1,4:0.2):0.3,(2:0.4,3:0.5):0.6):0.7,((5:0.8,6:0.9):1.0,7:1.1):1.2);"
        )
        self.tree_phylo = skbio.TreeNode.read([self.tree_str])
        # this tree is already filtered to the feature table
        self.tree_str_filtered = (
            "(((F1:0.1,F4:0.2):0.3,(F2:0.4,F3:0.5):0.6):0.7,(F5:0.8,F6:0.9):2.2);"
        )
        self.tree_phylo_filtered = skbio.TreeNode.read([self.tree_str_filtered])

    def test_load_experiment_config(self):
        with tempfile.NamedTemporaryFile(mode="w") as temp_file:
            json.dump(self.config, temp_file)
            temp_file.flush()
            loaded_config = _load_experiment_config(temp_file.name)
            self.assertEqual(loaded_config, self.config)

    def test_verify_experiment_config(self):
        _verify_experiment_config(self.config)

    def test_verify_experiment_config_corrupt(self):
        config = self.config.copy()
        config["tracking_uri"] = "invalid"
        with self.assertRaises(ValueError):
            _verify_experiment_config(config)

    def test_save_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _save_config(self.config, temp_dir, "test_config.json")

            config_path = os.path.join(temp_dir, "test_config.json")
            self.assertTrue(os.path.exists(config_path))

            with open(config_path, "r") as f:
                loaded_config = json.load(f)
                self.assertEqual(loaded_config, self.config)

    def test_load_taxonomy(self):
        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w") as f:
            self.tax.to_csv(f, sep="\t")
            f.flush()
            loaded_tax = _load_taxonomy(f.name)
        assert_series_equal(loaded_tax["Taxon"], self.tax["Taxon"])

    def test_process_taxonomy_rename(self):
        processed_tax = _process_taxonomy(self.tax, self.ft_t0)

        assert_series_equal(processed_tax["Taxon"], self.tax_renamed["Taxon"])

    def test_process_taxonomy_filter(self):
        tax_more_ft = self.tax.copy()
        tax_more_ft.loc[7, :] = tax_more_ft.loc[6, :].copy()
        processed_tax = _process_taxonomy(tax_more_ft, self.ft_t0)

        assert_series_equal(processed_tax["Taxon"], self.tax_renamed["Taxon"])

    def test_process_taxonomy_not_matching(self):
        tax_not_matched = self.tax.copy()
        tax_not_matched.index = tax_not_matched.index.map(lambda x: "GA" + str(x))
        with self.assertRaisesRegex(
            ValueError, "Taxonomy data does not match with feature table."
        ):
            _process_taxonomy(tax_not_matched, self.ft_t0)

    def test_load_phylogeny(self):
        with tempfile.NamedTemporaryFile(suffix=".nwk", mode="w") as f:
            skbio.TreeNode.write(self.tree_phylo, f)
            f.flush()
            loaded_phylo = _load_phylogeny(f.name)
        self.assertEqual(str(loaded_phylo), str(self.tree_phylo))

    def test_process_phylogeny(self):
        tree_phylo = skbio.TreeNode.read([self.tree_str])
        processed_tree = _process_phylogeny(tree_phylo, self.ft_t0)
        self.assertEqual(str(processed_tree).strip(), self.tree_str_filtered)

    def test_process_taxonomy_with_time_suffixes(self):
        # feature table with temporal suffixes (duplicate base features over time)
        # t0 columns have no suffix, t-1 columns get __t-1 suffix
        ft_time_t0 = self.ft.copy()
        ft_time_t1 = self.ft.copy()
        ft_time_t1.columns = [c + "__t-1" for c in ft_time_t1.columns]
        ft_merged = pd.concat([ft_time_t0, ft_time_t1], axis=1)

        processed_tax = _process_taxonomy(self.tax, ft_merged)
        # Expect same taxonomy rows as base (no duplication, suffix stripping works)
        assert_series_equal(processed_tax["Taxon"], self.tax_renamed["Taxon"])

    def test_process_phylogeny_with_time_suffixes(self):
        # feature table with time suffixes: t0 unsuffixed, t-1 with __t-1
        ft_time_t0 = self.ft.copy()
        ft_time_t1 = self.ft.copy()
        ft_time_t1.columns = [c + "__t-1" for c in ft_time_t1.columns]
        ft_merged = pd.concat([ft_time_t0, ft_time_t1], axis=1)

        # Use original tree (with extra leaf 7) so filtering logic is exercised
        tree_phylo = skbio.TreeNode.read([self.tree_str])
        processed_tree = _process_phylogeny(tree_phylo, ft_merged)
        # After suffix stripping the structure should still match filtered tree
        self.assertEqual(str(processed_tree).strip(), self.tree_str_filtered)

    def test_define_model_tracker_mlflow(self):
        with patch("builtins.print") as mock_print:
            path_tracker = _define_model_tracker("mlruns", "experiments/models")
            self.assertEqual(path_tracker, "sqlite:///experiments/models/mlflow.db")
            mock_print.assert_called_once()

    def test_define_model_tracker_wandb(self):
        with patch("builtins.print") as mock_print:
            path_tracker = _define_model_tracker("wandb", "experiments/models")
            self.assertEqual(path_tracker, "wandb")
            mock_print.assert_called_once()

    def test_define_experiment_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path_experiment = _define_experiment_path(self.config, temp_dir)
            self.assertTrue(os.path.exists(path_experiment))

    def test_define_experiment_path_already_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path_exp_exists = os.path.join(temp_dir, self.config["experiment_tag"])
            os.makedirs(path_exp_exists)
            with self.assertRaisesRegex(
                ValueError, f"already exists: {self.config['experiment_tag']}."
            ):
                _define_experiment_path(self.config, temp_dir)

    @patch("ritme.find_best_model_config._extract_mlflow_logs_to_csv")
    @patch("ritme.find_best_model_config.run_all_trials")
    @patch("ritme.find_best_model_config.retrieve_n_init_best_models")
    def test_find_best_model_config(
        self,
        mock_retrieve_n_init_best_models,
        mock_run_all_trials,
        mock_extract_mlflow,
    ):
        # Mock the return values of the functions
        mock_run_all_trials.return_value = {"model1": "result1", "model2": "result2"}
        mock_retrieve_n_init_best_models.return_value = {
            "model1": MagicMock(),
            "model2": MagicMock(),
        }

        # define phylo tree
        tree_phylo = skbio.TreeNode.read([self.tree_str])

        # Call the function under test
        with tempfile.TemporaryDirectory() as temp_dir:
            best_model_dic, path_exp = find_best_model_config(
                self.config, self.train_val, self.tax, tree_phylo, temp_dir
            )

            # assert pandas dataframes - not easily checkable in the
            # assert_called_once method below
            args, _ = mock_run_all_trials.call_args
            assert_frame_equal(args[0], self.train_val)
            # After adding stratify_by positional arg, tax and phylo shift one index
            assert_frame_equal(args[6], self.tax_renamed)
            self.assertEqual(str(args[7]), str(self.tree_phylo_filtered))
            # Trial storage and mlruns are in a temp directory (not path_exp)
            mock_run_all_trials.assert_called_once_with(
                ANY,  # train_val
                self.config["target"],
                self.config["group_by_column"],
                None,  # stratify_by absent
                self.config["seed_data"],
                self.config["seed_model"],
                ANY,  # processed tax
                ANY,  # processed phylo
                ANY,  # path_tracker (temp dir mlruns)
                ANY,  # tmp_storage (temp dir)
                self.config["time_budget_s"],
                self.config["max_cuncurrent_trials"],
                self.config["experiment_tag"],
                model_types=self.config["ls_model_types"],
                fully_reproducible=False,
                model_hyperparameters={},
                optuna_searchspace_sampler="TPESampler",
                task_type="regression",
                k_folds=ANY,
            )
            # Verify temp storage paths are NOT under path_exp
            self.assertNotEqual(args[8], os.path.join(path_exp, "mlruns"))
            self.assertNotEqual(args[9], path_exp)
            # Verify MLflow extraction was called
            mock_extract_mlflow.assert_called_once()

    @patch("ritme.find_best_model_config._extract_mlflow_logs_to_csv")
    @patch("ritme.find_best_model_config.run_all_trials")
    @patch("ritme.find_best_model_config.retrieve_n_init_best_models")
    def test_find_best_model_config_with_stratify_by(
        self,
        mock_retrieve_n_init_best_models,
        mock_run_all_trials,
        mock_extract_mlflow,
    ):
        config_w_strat = self.config.copy()
        config_w_strat["stratify_by"] = ["stratify_column"]
        mock_run_all_trials.return_value = {"model1": "result1", "model2": "result2"}
        mock_tmodel1 = MagicMock()
        mock_tmodel2 = MagicMock()
        mock_retrieve_n_init_best_models.return_value = {
            "model1": mock_tmodel1,
            "model2": mock_tmodel2,
        }
        tree_phylo = skbio.TreeNode.read([self.tree_str])
        with tempfile.TemporaryDirectory() as temp_dir:
            best_model_dic, path_exp = find_best_model_config(
                config_w_strat, self.train_val, self.tax, tree_phylo, temp_dir
            )
            args, _ = mock_run_all_trials.call_args
            self.assertEqual(args[3], ["stratify_column"])  # stratify_by
            mock_run_all_trials.assert_called_once()
            self.assertEqual(set(best_model_dic.keys()), {"model1", "model2"})
            self.assertIs(best_model_dic["model1"], mock_tmodel1)
            self.assertIs(best_model_dic["model2"], mock_tmodel2)
            self.assertTrue(path_exp.startswith(f"{temp_dir}/test_experiment"))

    @patch("ritme.find_best_model_config._load_experiment_config")
    @patch("ritme.find_best_model_config.pd.read_pickle")
    @patch("ritme.find_best_model_config._load_taxonomy")
    @patch("ritme.find_best_model_config._load_phylogeny")
    @patch("ritme.find_best_model_config.find_best_model_config")
    @patch("ritme.find_best_model_config.save_best_models")
    def test_cli_find_best_model_config_w_tax_phylo(
        self,
        mock_best_models,
        mock_find_best_model_config,
        mock_phylo,
        mock_tax,
        mock_train_val,
        mock_config,
    ):
        path_to_logs = "path/to/logs"
        # Mock the return values of the functions
        mock_config.return_value = self.config
        mock_train_val.return_value = self.train_val
        mock_tax.return_value = self.tax
        mock_phylo.return_value = self.tree_phylo
        mock_find_best_model_config.return_value = (
            {"model1": "best_model1", "model2": "best_model2"},
            path_to_logs,
        )
        mock_best_models.return_value = None

        with patch("sys.stdout", new=StringIO()) as stdout:
            cli_find_best_model_config(
                "path/to/config",
                "path/to/train_val",
                "path/to/tax",
                "path/to/tree_phylo",
                path_to_logs,
            )
            self.assertIn(
                f"Best model configurations were saved in {path_to_logs}.",
                stdout.getvalue().strip(),
            )

        # assert called with tax and tree_phylo
        args, _ = mock_find_best_model_config.call_args
        assert_frame_equal(args[2], self.tax)
        self.assertEqual(str(args[3]), str(self.tree_phylo))

    @patch("ritme.find_best_model_config._load_experiment_config")
    @patch("ritme.find_best_model_config.pd.read_pickle")
    @patch("ritme.find_best_model_config.find_best_model_config")
    @patch("ritme.find_best_model_config.save_best_models")
    def test_cli_find_best_model_config_no_tax_phylo(
        self,
        mock_best_models,
        mock_find_best_model_config,
        mock_train_val,
        mock_config,
    ):
        path_to_logs = "path/to/logs"
        # Mock the return values of the functions
        mock_config.return_value = self.config
        mock_train_val.return_value = self.train_val
        mock_find_best_model_config.return_value = (
            {"model1": "best_model1", "model2": "best_model2"},
            path_to_logs,
        )
        mock_best_models.return_value = None

        # call w/o tax and phylo
        cli_find_best_model_config(
            "path/to/config",
            "path/to/train_val",
            path_store_model_logs=path_to_logs,
        )

        # assert called with tax and tree_phylo
        args, _ = mock_find_best_model_config.call_args
        self.assertEqual(args[2], None)
        self.assertEqual(args[3], None)


class TestExtractMlflowLogsToCsv(unittest.TestCase):
    @patch("mlflow.tracking.MlflowClient")
    def test_extract_mlflow_logs_to_csv(self, mock_client_class):
        mock_client = mock_client_class.return_value

        mock_exp = MagicMock()
        mock_exp.experiment_id = "1"
        mock_exp.name = "xgb"
        mock_client.search_experiments.return_value = [mock_exp]

        mock_run = MagicMock()
        mock_run.info.run_id = "run123"
        mock_run.info.experiment_id = "1"
        mock_run.info.status = "FINISHED"
        mock_run.info.start_time = 1000
        mock_run.info.end_time = 2000
        mock_run.data.params = {"model": "xgb", "lr": "0.1"}
        mock_run.data.metrics = {"rmse_val": 0.5, "nb_features": 10.0}
        mock_run.data.tags = {"experiment_tag": "test_exp"}
        mock_client.search_runs.return_value = [mock_run]

        with tempfile.TemporaryDirectory() as temp_dir:
            _extract_mlflow_logs_to_csv("sqlite:///fake/mlflow.db", temp_dir)

            csv_path = os.path.join(temp_dir, "mlflow_logs.csv")
            self.assertTrue(os.path.exists(csv_path))

            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.loc[0, "run_id"], "run123")
            self.assertEqual(df.loc[0, "experiment_name"], "xgb")
            self.assertEqual(df.loc[0, "status"], "FINISHED")
            self.assertAlmostEqual(df.loc[0, "metrics.rmse_val"], 0.5)
            self.assertEqual(df.loc[0, "params.model"], "xgb")
            self.assertEqual(df.loc[0, "tags.experiment_tag"], "test_exp")

    @patch("mlflow.tracking.MlflowClient")
    def test_extract_mlflow_logs_multiple_experiments(self, mock_client_class):
        mock_client = mock_client_class.return_value

        mock_exp1 = MagicMock()
        mock_exp1.experiment_id = "1"
        mock_exp1.name = "xgb"
        mock_exp2 = MagicMock()
        mock_exp2.experiment_id = "2"
        mock_exp2.name = "rf"
        mock_client.search_experiments.return_value = [mock_exp1, mock_exp2]

        mock_run1 = MagicMock()
        mock_run1.info.run_id = "run1"
        mock_run1.info.experiment_id = "1"
        mock_run1.info.status = "FINISHED"
        mock_run1.info.start_time = 1000
        mock_run1.info.end_time = 2000
        mock_run1.data.params = {"model": "xgb"}
        mock_run1.data.metrics = {"rmse_val": 0.5}
        mock_run1.data.tags = {}

        mock_run2 = MagicMock()
        mock_run2.info.run_id = "run2"
        mock_run2.info.experiment_id = "2"
        mock_run2.info.status = "FINISHED"
        mock_run2.info.start_time = 3000
        mock_run2.info.end_time = 4000
        mock_run2.data.params = {"model": "rf"}
        mock_run2.data.metrics = {"rmse_val": 0.3}
        mock_run2.data.tags = {}
        mock_client.search_runs.return_value = [mock_run1, mock_run2]

        with tempfile.TemporaryDirectory() as temp_dir:
            _extract_mlflow_logs_to_csv("sqlite:///fake/mlflow.db", temp_dir)
            df = pd.read_csv(os.path.join(temp_dir, "mlflow_logs.csv"))
            self.assertEqual(len(df), 2)
            self.assertEqual(set(df["experiment_name"]), {"xgb", "rf"})

    @patch("mlflow.tracking.MlflowClient")
    def test_extract_mlflow_logs_paginated(self, mock_client_class):
        # Verifies that all runs are collected when MLflow returns multiple
        # pages (regression test for the search_runs default max_results=1000
        # cap).
        mock_client = mock_client_class.return_value

        mock_exp = MagicMock()
        mock_exp.experiment_id = "1"
        mock_exp.name = "xgb"
        mock_client.search_experiments.return_value = [mock_exp]

        class FakePage(list):
            def __init__(self, items, token=None):
                super().__init__(items)
                self.token = token

        def make_run(run_id):
            run = MagicMock()
            run.info.run_id = run_id
            run.info.experiment_id = "1"
            run.info.status = "FINISHED"
            run.info.start_time = 1000
            run.info.end_time = 2000
            run.data.params = {"model": "xgb"}
            run.data.metrics = {"rmse_val": 0.5}
            run.data.tags = {}
            return run

        page1 = FakePage([make_run(f"run{i}") for i in range(1000)], token="t1")
        page2 = FakePage([make_run(f"run{i}") for i in range(1000, 1500)], token=None)
        mock_client.search_runs.side_effect = [page1, page2]

        with tempfile.TemporaryDirectory() as temp_dir:
            _extract_mlflow_logs_to_csv("sqlite:///fake/mlflow.db", temp_dir)

            df = pd.read_csv(os.path.join(temp_dir, "mlflow_logs.csv"))
            self.assertEqual(len(df), 1500)
            self.assertEqual(mock_client.search_runs.call_count, 2)

            first_call_kwargs = mock_client.search_runs.call_args_list[0].kwargs
            second_call_kwargs = mock_client.search_runs.call_args_list[1].kwargs
            self.assertIsNone(first_call_kwargs.get("page_token"))
            self.assertEqual(second_call_kwargs.get("page_token"), "t1")

    @patch("mlflow.tracking.MlflowClient")
    def test_extract_mlflow_logs_no_experiments(self, mock_client_class):
        mock_client = mock_client_class.return_value
        mock_client.search_experiments.return_value = []

        with tempfile.TemporaryDirectory() as temp_dir:
            _extract_mlflow_logs_to_csv("sqlite:///fake/mlflow.db", temp_dir)
            csv_path = os.path.join(temp_dir, "mlflow_logs.csv")
            self.assertFalse(os.path.exists(csv_path))

    @patch("mlflow.tracking.MlflowClient")
    def test_extract_mlflow_logs_no_runs(self, mock_client_class):
        mock_client = mock_client_class.return_value

        mock_exp = MagicMock()
        mock_exp.experiment_id = "0"
        mock_exp.name = "Default"
        mock_client.search_experiments.return_value = [mock_exp]
        mock_client.search_runs.return_value = []

        with tempfile.TemporaryDirectory() as temp_dir:
            _extract_mlflow_logs_to_csv("sqlite:///fake/mlflow.db", temp_dir)
            csv_path = os.path.join(temp_dir, "mlflow_logs.csv")
            self.assertFalse(os.path.exists(csv_path))

    @patch("ritme.find_best_model_config._extract_mlflow_logs_to_csv")
    @patch("ritme.find_best_model_config.run_all_trials")
    @patch("ritme.find_best_model_config.retrieve_n_init_best_models")
    def test_find_best_model_config_no_mlflow_extraction_for_wandb(
        self,
        mock_retrieve,
        mock_run_all_trials,
        mock_extract_mlflow,
    ):
        config = {
            "tracking_uri": "wandb",
            "fully_reproducible": False,
            "experiment_tag": "test_wandb",
            "target": "target_column",
            "group_by_column": "group_column",
            "seed_data": 42,
            "seed_model": 42,
            "time_budget_s": 10,
            "max_cuncurrent_trials": 2,
            "ls_model_types": ["xgb"],
            "model_hyperparameters": {},
        }
        mock_run_all_trials.return_value = {"xgb": "result"}
        mock_retrieve.return_value = {"xgb": MagicMock()}

        train_val = pd.DataFrame(
            {"F1": [0.1, 0.2], "target_column": [1, 2], "group_column": ["a", "b"]}
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            find_best_model_config(config, train_val, path_store_model_logs=temp_dir)
            mock_extract_mlflow.assert_not_called()

    @patch("ritme.find_best_model_config._extract_mlflow_logs_to_csv")
    @patch("ritme.find_best_model_config.run_all_trials")
    @patch("ritme.find_best_model_config.retrieve_n_init_best_models")
    def test_find_best_model_config_updates_model_paths(
        self,
        mock_retrieve,
        mock_run_all_trials,
        mock_extract_mlflow,
    ):
        mock_run_all_trials.return_value = {"xgb": "result"}

        mock_tmodel = MagicMock()
        mock_tmodel.path = "/tmp/some_temp_trial_dir"
        mock_retrieve.return_value = {"xgb": mock_tmodel}

        train_val = pd.DataFrame(
            {"F1": [0.1, 0.2], "target_column": [1, 2], "group_column": ["a", "b"]}
        )
        config = {
            "tracking_uri": "mlruns",
            "fully_reproducible": False,
            "experiment_tag": "test_path_update",
            "target": "target_column",
            "group_by_column": "group_column",
            "seed_data": 42,
            "seed_model": 42,
            "time_budget_s": 10,
            "max_cuncurrent_trials": 2,
            "ls_model_types": ["xgb"],
            "model_hyperparameters": {},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            best_models, path_exp = find_best_model_config(
                config, train_val, path_store_model_logs=temp_dir
            )
            self.assertEqual(best_models["xgb"].path, path_exp)


if __name__ == "__main__":
    unittest.main()
