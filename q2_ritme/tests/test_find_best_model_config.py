# test_find_best_model_config.py
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import qiime2 as q2
import skbio
from pandas.testing import assert_series_equal
from qiime2.plugin.testing import TestPluginBase

from q2_ritme.find_best_model_config import (
    _define_experiment_path,
    _define_model_tracker,
    _load_experiment_config,
    _load_phylogeny,
    _load_taxonomy,
    _process_phylogeny,
    _process_taxonomy,
    _save_config,
    _verify_experiment_config,
    find_best_model_config,
)


class TestFindBestModelConfig(TestPluginBase):
    package = "q2_ritme.tests"

    def setUp(self):
        super().setUp()
        self.config = {
            "tracking_uri": "mlruns",
            "experiment_tag": "test_experiment",
            "feature_prefix": "F",
            "target": "target_column",
            "stratify_by_column": "stratify_column",
            "seed_data": 42,
            "seed_model": 42,
            "num_trials": 10,
            "max_cuncurrent_trials": 2,
            "ls_model_types": ["model1", "model2"],
            "test_mode": False,
            "model_hyperparameters": {},
        }
        self.ft = pd.read_csv(
            self.get_data_path("example_feature_table.tsv"), sep="\t", index_col=0
        )
        self.train_val = self.ft.copy()
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
        self.tax_renamed = pd.read_csv(
            self.get_data_path("example_taxonomy.tsv"), sep="\t", index_col=0
        )
        self.tax_renamed.index.name = "Feature ID"

        self.tax = self.tax_renamed.copy()
        self.tax.index = self.tax.index.map(lambda x: x.replace("F", ""))
        self.tax_art = q2.Artifact.import_data("FeatureData[Taxonomy]", self.tax)
        self.tree_str_more_fts = (
            "(((1:0.1,4:0.2):0.3,(2:0.4,3:0.5):0.6):0.7,((5:0.8,6:0.9):1.0,7:1.1):1.2);"
        )

        self.tree_phylo = skbio.TreeNode.read([self.tree_str_more_fts])
        self.tree_art = q2.Artifact.import_data("Phylogeny[Rooted]", self.tree_phylo)

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

    @patch("q2_ritme.find_best_model_config.q2.Artifact")
    def test_load_taxonomy(self, mock_artifact):
        mock_artifact.load.return_value = self.tax_art
        loaded_tax = _load_taxonomy("taxonomy.qza")
        # we're only interested in Taxon here - not in confidence which changes
        # type by reformatting
        assert_series_equal(loaded_tax["Taxon"], self.tax["Taxon"])

    def test_process_taxonomy_rename(self):
        processed_tax = _process_taxonomy(self.tax, self.ft)

        assert_series_equal(processed_tax["Taxon"], self.tax_renamed["Taxon"])

    def test_process_taxonomy_filter(self):
        tax_more_ft = self.tax.copy()
        tax_more_ft.loc["7", :] = tax_more_ft.loc["6", :].copy()
        processed_tax = _process_taxonomy(tax_more_ft, self.ft)

        assert_series_equal(processed_tax["Taxon"], self.tax_renamed["Taxon"])

    def test_process_taxonomy_not_matching(self):
        tax_not_matched = self.tax.copy()
        tax_not_matched.index = tax_not_matched.index.map(lambda x: "GA" + str(x))
        with self.assertRaisesRegex(
            ValueError, "Taxonomy data does not match with feature table."
        ):
            _process_taxonomy(tax_not_matched, self.ft)

    @patch("q2_ritme.find_best_model_config.q2.Artifact")
    def test_load_phylogeny(self, mock_artifact):
        mock_artifact.load.return_value = self.tree_art
        loaded_phylo = _load_phylogeny("phylogeny.qza")
        self.assertEqual(str(loaded_phylo), str(self.tree_phylo))

    def test_process_phylogeny(self):
        tree_str_more_fts = (
            "(((1:0.1,4:0.2):0.3,(2:0.4,3:0.5):0.6):0.7,((5:0.8,6:0.9):1.0,7:1.1):1.2);"
        )
        tree_phylo = skbio.TreeNode.read([tree_str_more_fts])
        processed_tree = _process_phylogeny(tree_phylo, self.ft)
        expected_tree = (
            "(((F1:0.1,F4:0.2):0.3,(F2:0.4,F3:0.5):0.6):0.7,(F5:0.8,F6:0.9):2.2);"
        )
        self.assertEqual(str(processed_tree).strip(), expected_tree)

    def test_define_model_tracker_mlflow(self):
        with patch("builtins.print") as mock_print:
            path_tracker = _define_model_tracker("mlruns", "experiments/models")
            self.assertEqual(path_tracker, "experiments/models/mlruns")
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

    @patch("q2_ritme.find_best_model_config.run_all_trials")
    @patch("q2_ritme.find_best_model_config.retrieve_best_models")
    def test_find_best_model_config(
        self, mock_retrieve_best_models, mock_run_all_trials
    ):
        # Mock the return values of the functions
        mock_run_all_trials.return_value = {"model1": "result1", "model2": "result2"}
        mock_retrieve_best_models.return_value = {
            "model1": "best_model1",
            "model2": "best_model2",
        }

        # define phylo tree
        tree_str = (
            "(((1:0.1,4:0.2):0.3,(2:0.4,3:0.5):0.6):0.7,((5:0.8,6:0.9):1.0,7:1.1):1.2);"
        )
        tree_phylo = skbio.TreeNode.read([tree_str])

        # Call the function under test
        with tempfile.TemporaryDirectory() as temp_dir:
            best_model_dic, path_exp = find_best_model_config(
                self.config, self.train_val, self.tax, tree_phylo, temp_dir
            )

            # Assertions
            mock_run_all_trials.assert_called_once()
            mock_retrieve_best_models.assert_called_once_with(
                mock_run_all_trials.return_value
            )
            self.assertEqual(
                best_model_dic, {"model1": "best_model1", "model2": "best_model2"}
            )
            self.assertTrue(path_exp.startswith(f"{temp_dir}/test_experiment"))


if __name__ == "__main__":
    unittest.main()
