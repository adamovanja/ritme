# test_find_best_model_config.py
import json
import os
import tempfile
import unittest
from io import StringIO
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pandas as pd
import skbio
from pandas.testing import assert_frame_equal, assert_series_equal

from ritme.find_best_model_config import (
    _STUB_DIR_ALLOWED_FILES,
    _STUB_DIR_IGNORED_NAMES,
    _define_experiment_path,
    _define_model_tracker,
    _extract_mlflow_logs_to_csv,
    _load_experiment_config,
    _load_phylogeny,
    _load_taxonomy,
    _process_phylogeny,
    _process_taxonomy,
    _save_config,
    _verify_data_enrich_compat,
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
            # Real model names so _validate_run_inputs (called before
            # _define_experiment_path) accepts them. Mock return values
            # below still use synthetic "model1"/"model2" keys to
            # exercise the result-dict plumbing.
            "ls_model_types": ["linreg", "rf"],
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

    def _make_enrich_data(self, bmi_values=(22.0, 23.0, 24.0), site_values=None):
        if site_values is None:
            site_values = ["gut", "skin", "gut"]
        return pd.DataFrame(
            {
                "F1": [0.1, 0.2, 0.3],
                "F2": [0.4, 0.5, 0.6],
                "bmi": list(bmi_values),
                "body-site": pd.Series(site_values, dtype="object"),
                "target_column": [1.0, 2.0, 3.0],
            }
        )

    @staticmethod
    def _enrich_config(
        *,
        data_enrich_with,
        ls_model_types,
        data_enrich_options=None,
        model_hyperparameters_extra=None,
    ):
        mh = dict(model_hyperparameters_extra or {})
        if data_enrich_with is not None:
            mh["data_enrich_with"] = data_enrich_with
        if data_enrich_options is not None:
            mh["data_enrich_options"] = data_enrich_options
        return {
            "ls_model_types": list(ls_model_types),
            "model_hyperparameters": mh,
        }

    def test_verify_data_enrich_compat_no_enrich_with(self):
        # No data_enrich_with means no metadata is injected, so the
        # validator must not fire regardless of requested models.
        for value in (None, []):
            with self.subTest(data_enrich_with=value):
                config = self._enrich_config(
                    data_enrich_with=value, ls_model_types=["linreg"]
                )
                train_val = self._make_enrich_data(bmi_values=[22.0, np.nan, 24.0])
                _verify_data_enrich_compat(config, train_val)
        empty_mh_config = {
            "ls_model_types": ["linreg"],
            "model_hyperparameters": {},
        }
        _verify_data_enrich_compat(empty_mh_config, self._make_enrich_data())

    def test_verify_data_enrich_compat_options_exclude_metadata(self):
        # data_enrich_options explicitly restricted to non-metadata modes ->
        # no NaN can reach the trainable, so validator no-ops.
        config = self._enrich_config(
            data_enrich_with=["bmi"],
            ls_model_types=["linreg"],
            data_enrich_options=[None, "shannon"],
        )
        train_val = self._make_enrich_data(bmi_values=[22.0, np.nan, 24.0])
        _verify_data_enrich_compat(config, train_val)

    def test_verify_data_enrich_compat_no_nan(self):
        config = self._enrich_config(
            data_enrich_with=["bmi"], ls_model_types=["linreg"]
        )
        _verify_data_enrich_compat(config, self._make_enrich_data())

    def test_verify_data_enrich_compat_only_tolerant_models(self):
        config = self._enrich_config(
            data_enrich_with=["bmi"], ls_model_types=["xgb", "rf"]
        )
        train_val = self._make_enrich_data(bmi_values=[22.0, np.nan, 24.0])
        _verify_data_enrich_compat(config, train_val)

    def test_verify_data_enrich_compat_categorical_with_nan_passes(self):
        # body-site is object/categorical -> enrich_features one-hots it via
        # pd.get_dummies, which never propagates NaN to the feature matrix.
        config = self._enrich_config(
            data_enrich_with=["body-site"], ls_model_types=["linreg"]
        )
        train_val = self._make_enrich_data(site_values=["gut", None, "skin"])
        _verify_data_enrich_compat(config, train_val)

    def test_verify_data_enrich_compat_raises_with_nan_and_intolerant(self):
        config = self._enrich_config(
            data_enrich_with=["bmi"], ls_model_types=["xgb", "linreg"]
        )
        train_val = self._make_enrich_data(bmi_values=[22.0, np.nan, np.nan])
        with self.assertRaises(ValueError) as ctx:
            _verify_data_enrich_compat(config, train_val)
        msg = str(ctx.exception)
        self.assertIn("bmi", msg)
        self.assertRegex(msg, r"'bmi'\D*\b2\b")
        # xgb is tolerant: must NOT appear in the intolerant-models list
        # (it still appears in the remediation suggestion downstream).
        self.assertRegex(msg, r"requested model types \[[^\]]*'linreg'[^\]]*\]")
        self.assertNotRegex(msg, r"requested model types \[[^\]]*'xgb'[^\]]*\]")

    def test_verify_data_enrich_compat_multi_column_summary(self):
        train_val = self._make_enrich_data(bmi_values=[22.0, np.nan, np.nan])
        train_val["age"] = [np.nan, 30.0, 40.0]
        config = self._enrich_config(
            data_enrich_with=["bmi", "age"], ls_model_types=["linreg"]
        )
        with self.assertRaises(ValueError) as ctx:
            _verify_data_enrich_compat(config, train_val)
        msg = str(ctx.exception)
        self.assertRegex(msg, r"'bmi'\D*\b2\b")
        self.assertRegex(msg, r"'age'\D*\b1\b")

    def test_verify_data_enrich_compat_skips_missing_columns(self):
        # Columns absent from train_val are delegated to enrich_features
        # (see enrich_features.py:46-52) which raises its own ValueError.
        config = self._enrich_config(
            data_enrich_with=["not_in_df"], ls_model_types=["linreg"]
        )
        _verify_data_enrich_compat(config, self._make_enrich_data())

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

    def test_define_experiment_path_existing_stub_allowed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stub_path = os.path.join(temp_dir, self.config["experiment_tag"])
            os.makedirs(stub_path)
            with open(os.path.join(stub_path, "experiment_config.json"), "w") as f:
                json.dump({}, f)
            returned = _define_experiment_path(
                self.config, temp_dir, allow_existing_tag=True
            )
            self.assertEqual(returned, stub_path)

    def test_define_experiment_path_existing_extras_refuses(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            real_path = os.path.join(temp_dir, self.config["experiment_tag"])
            os.makedirs(real_path)
            with open(os.path.join(real_path, "experiment_config.json"), "w") as f:
                json.dump({}, f)
            with open(os.path.join(real_path, "mlflow_logs.csv"), "w") as f:
                f.write("run_id\n")
            with self.assertRaisesRegex(ValueError, "mlflow_logs.csv"):
                _define_experiment_path(self.config, temp_dir, allow_existing_tag=True)

    def test_define_experiment_path_ignores_hidden_files(self):
        # OS / editor / Jupyter noise (.DS_Store, .ipynb_checkpoints,
        # __pycache__) must not block stub reuse.
        with tempfile.TemporaryDirectory() as temp_dir:
            stub_path = os.path.join(temp_dir, self.config["experiment_tag"])
            os.makedirs(stub_path)
            with open(os.path.join(stub_path, "experiment_config.json"), "w") as f:
                json.dump({}, f)
            for noise in (".DS_Store", ".ipynb_checkpoints"):
                with open(os.path.join(stub_path, noise), "w") as f:
                    f.write("noise")
            os.makedirs(os.path.join(stub_path, "__pycache__"))
            returned = _define_experiment_path(
                self.config, temp_dir, allow_existing_tag=True
            )
            self.assertEqual(returned, stub_path)

    def test_stub_dir_allowed_files_membership(self):
        # Pins the allowlist contents so a future maintainer cannot
        # silently expand the set (e.g. adding mlflow_logs.csv would
        # allow overwriting real artifacts).
        self.assertEqual(_STUB_DIR_ALLOWED_FILES, frozenset({"experiment_config.json"}))

    def test_stub_dir_ignored_names_membership(self):
        # Symmetric pin: silently expanding the ignore set would also
        # let real artifacts slip past the stub gate.
        self.assertEqual(_STUB_DIR_IGNORED_NAMES, frozenset({"__pycache__"}))
    @patch("ritme.find_best_model_config.run_all_trials")
    def test_find_best_model_config_aborts_on_enrich_nan_before_path_creation(
        self, mock_run_all_trials
    ):
        # End-to-end wiring: the validator must fire before run_all_trials
        # is invoked AND before _define_experiment_path creates the log
        # directory on disk.
        config = self.config.copy()
        config["ls_model_types"] = ["linreg"]
        config["model_hyperparameters"] = {"data_enrich_with": ["bmi"]}
        train_val = self.train_val.copy()
        train_val["bmi"] = [
            22.0,
            np.nan,
            24.0,
            25.0,
            26.0,
            27.0,
            28.0,
            29.0,
            30.0,
            31.0,
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, r"data_enrich_with"):
                find_best_model_config(config, train_val, None, None, temp_dir)
            mock_run_all_trials.assert_not_called()
            self.assertFalse(
                os.path.exists(os.path.join(temp_dir, config["experiment_tag"]))
            )

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
                nn_corn_max_levels=20,
                max_trial_failure_rate=0.005,
            )
            # Verify temp storage paths are NOT under path_exp
            self.assertNotEqual(args[8], os.path.join(path_exp, "mlruns"))
            self.assertNotEqual(args[9], path_exp)
            # Verify MLflow extraction was called
            mock_extract_mlflow.assert_called_once()

    @patch("ritme.find_best_model_config._extract_mlflow_logs_to_csv")
    @patch("ritme.find_best_model_config.run_all_trials")
    @patch("ritme.find_best_model_config.retrieve_n_init_best_models")
    def test_find_best_model_config_forwards_custom_nn_corn_max_levels(
        self,
        mock_retrieve_n_init_best_models,
        mock_run_all_trials,
        mock_extract_mlflow,
    ):
        """A non-default ``nn_corn_max_levels`` value in the experiment
        config must be forwarded to ``run_all_trials``. Guards against the
        ``config.get(...)`` line silently degrading to a hardcoded default.
        """
        mock_run_all_trials.return_value = {"model1": MagicMock()}
        mock_retrieve_n_init_best_models.return_value = {"model1": MagicMock()}

        config = self.config.copy()
        config["nn_corn_max_levels"] = 7

        tree_phylo = skbio.TreeNode.read([self.tree_str])
        with tempfile.TemporaryDirectory() as temp_dir:
            find_best_model_config(
                config, self.train_val, self.tax, tree_phylo, temp_dir
            )

        forwarded = mock_run_all_trials.call_args.kwargs["nn_corn_max_levels"]
        self.assertEqual(forwarded, 7)

    @patch("ritme.find_best_model_config._extract_mlflow_logs_to_csv")
    @patch("ritme.find_best_model_config.run_all_trials")
    @patch("ritme.find_best_model_config.retrieve_n_init_best_models")
    def test_find_best_model_config_forwards_max_trial_failure_rate(
        self,
        mock_retrieve_n_init_best_models,
        mock_run_all_trials,
        mock_extract_mlflow,
    ):
        mock_run_all_trials.return_value = {"model1": MagicMock()}
        mock_retrieve_n_init_best_models.return_value = {"model1": MagicMock()}

        config = self.config.copy()
        config["max_trial_failure_rate"] = 0.05

        tree_phylo = skbio.TreeNode.read([self.tree_str])
        with tempfile.TemporaryDirectory() as temp_dir:
            find_best_model_config(
                config, self.train_val, self.tax, tree_phylo, temp_dir
            )
        forwarded = mock_run_all_trials.call_args.kwargs["max_trial_failure_rate"]
        self.assertEqual(forwarded, 0.05)

    @patch("ritme.find_best_model_config.run_all_trials")
    def test_find_best_model_config_invalid_model_leaves_no_stub_dir(
        self, mock_run_all_trials
    ):
        # Pre-flight validators fire BEFORE _define_experiment_path so a
        # config-rejection failure leaves no stub directory on disk
        # (issue_m_existing_dir.md).
        config = self.config.copy()
        config["ls_model_types"] = ["not_a_real_model"]
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                find_best_model_config(config, self.train_val, self.tax, None, temp_dir)
            mock_run_all_trials.assert_not_called()
            self.assertFalse(
                os.path.exists(os.path.join(temp_dir, config["experiment_tag"]))
            )

    @patch("ritme.find_best_model_config._extract_mlflow_logs_to_csv")
    @patch("ritme.find_best_model_config.run_all_trials")
    @patch("ritme.find_best_model_config.retrieve_n_init_best_models")
    def test_find_best_model_config_allow_existing_tag_reuses_stub(
        self,
        mock_retrieve_n_init_best_models,
        mock_run_all_trials,
        mock_extract_mlflow,
    ):
        mock_run_all_trials.return_value = {"model1": MagicMock()}
        mock_retrieve_n_init_best_models.return_value = {"model1": MagicMock()}

        tree_phylo = skbio.TreeNode.read([self.tree_str])
        with tempfile.TemporaryDirectory() as temp_dir:
            stub_dir = os.path.join(temp_dir, self.config["experiment_tag"])
            os.makedirs(stub_dir)
            with open(os.path.join(stub_dir, "experiment_config.json"), "w") as f:
                json.dump({}, f)
            _, path_exp = find_best_model_config(
                self.config,
                self.train_val,
                self.tax,
                tree_phylo,
                temp_dir,
                allow_existing_tag=True,
            )
            self.assertEqual(path_exp, stub_dir)

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
    def test_cli_find_best_model_config_forwards_allow_existing_tag(
        self,
        mock_best_models,
        mock_find_best_model_config,
        mock_train_val,
        mock_config,
    ):
        # Pins the Typer flag wiring: a CLI invocation with the flag
        # must pass allow_existing_tag=True to find_best_model_config.
        # A regression where the parameter default leaks a typer.OptionInfo
        # (instead of a bool) would fail this test.
        mock_config.return_value = self.config
        mock_train_val.return_value = self.train_val
        mock_find_best_model_config.return_value = ({}, "path/to/logs")
        mock_best_models.return_value = None

        cli_find_best_model_config(
            "path/to/config",
            "path/to/train_val",
            allow_existing_tag=True,
        )
        kwargs = mock_find_best_model_config.call_args.kwargs
        self.assertIs(kwargs["allow_existing_tag"], True)

        # And the default path (no flag): must be exactly False, not an
        # OptionInfo.
        mock_find_best_model_config.reset_mock()
        cli_find_best_model_config("path/to/config", "path/to/train_val")
        kwargs = mock_find_best_model_config.call_args.kwargs
        self.assertIs(kwargs["allow_existing_tag"], False)

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
