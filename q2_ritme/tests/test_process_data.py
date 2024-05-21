import os
import tempfile

import pandas as pd
import qiime2 as q2
from pandas.testing import assert_frame_equal
from qiime2.plugin.testing import TestPluginBase

from q2_ritme.process_data import (
    filter_merge_n_sort,
    get_relative_abundance,
    load_data,
    load_n_split_data,
    split_data_by_host,
)


class TestProcessData(TestPluginBase):
    package = "q2_ritme.test"

    def setUp(self):
        super().setUp()
        self.data_abs = pd.DataFrame(
            {
                "host_id": ["c", "b", "c", "a"],
                "F0": [100, 10, 3, 5],
                "F1": [20, 30, 0, 0],
                "X1": [50, 50, 30, 5],
                "X2": [30, 40, 1, 1],
                "supertarget": [1, 2, 5, 7],
                "covariate": [0, 1, 0, 1],
            }
        )
        self.data_rel = pd.DataFrame(
            {
                "host_id": ["c", "b", "c", "a"],
                "F0": [0.83333333333333333, 0.25, 1.0, 1.0],
                "F1": [0.16666666666666666, 0.75, 0.0, 0.0],
                "supertarget": [1, 2, 5, 7],
                "covariate": [0, 1, 0, 1],
            }
        )
        self.data_abs.index = ["SR1", "SR2", "SR3", "SR4"]
        self.data_rel.index = ["SR1", "SR2", "SR3", "SR4"]
        self.md = self.data_rel[["host_id", "supertarget", "covariate"]]
        self.ft_abs = self.data_abs[["F0", "F1"]]
        self.ft_rel = self.data_rel[["F0", "F1"]]

        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_md_path = os.path.join(self.tmpdir.name, "test_md.tsv")
        self.md.to_csv(self.tmp_md_path, sep="\t")

        self.tmp_ft_rel_path = os.path.join(self.tmpdir.name, "test_ft_rel.tsv")
        self.ft_rel.to_csv(self.tmp_ft_rel_path, sep="\t")
        self.tmp_ft_abs_path = os.path.join(self.tmpdir.name, "test_ft_abs.tsv")
        self.ft_abs.to_csv(self.tmp_ft_abs_path, sep="\t")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_get_relative_abundance(self):
        ft_rel_obs = get_relative_abundance(self.ft_abs, feature_prefix="F")

        pd.testing.assert_frame_equal(ft_rel_obs, self.ft_rel)

    def test_get_relative_abundance_no_features(self):
        # Call the get_relative_abundance function with no_features
        ft_rel_obs = get_relative_abundance(self.ft_abs, no_features=["X1", "X2"])

        # Compare the observed and expected DataFrames
        pd.testing.assert_frame_equal(ft_rel_obs, self.ft_rel)

    def test_get_relative_abundance_no_features_prefix(self):
        # Test the case when neither feature_prefix nor no_features is provided
        with self.assertRaises(ValueError):
            get_relative_abundance(self.ft_abs)

    def test_load_data_ft_rel_tsv(self):
        # Load the relative feature abundance table
        ft_rel, md = load_data(self.tmp_md_path, self.tmp_ft_rel_path)

        pd.testing.assert_frame_equal(ft_rel, self.ft_rel)
        pd.testing.assert_frame_equal(md, self.md)

    def test_load_data_ft_abs_tsv(self):
        # Load the absolute feature abundance table -> transformed to relative
        # abundances
        ft, md = load_data(self.tmp_md_path, self.tmp_ft_abs_path)

        pd.testing.assert_frame_equal(ft, self.ft_rel)
        pd.testing.assert_frame_equal(md, self.md)

    def test_load_data_ft_abs_qza(self):
        art_ft = q2.Artifact.import_data("FeatureTable[Frequency]", self.ft_abs)
        tmp_ft_abs_path_art = self.tmp_ft_abs_path.replace(".tsv", ".qza")
        art_ft.save(tmp_ft_abs_path_art)
        # Load the data from the temporary files
        ft, md = load_data(self.tmp_md_path, tmp_ft_abs_path_art)

        pd.testing.assert_frame_equal(ft, self.ft_rel)
        pd.testing.assert_frame_equal(md, self.md)

    def test_load_data_simulated(self):
        ft, md = load_data()
        assert ft.shape[0] == 1000
        assert md.shape[0] == 1000

    def test_load_data_no_feature_prefix(self):
        tmp_ft_path_noprefix = self.tmp_ft_rel_path.replace(".tsv", "_noprefix.tsv")
        ft_noprefix = self.ft_rel.rename(columns={"F0": "0", "F1": "1"})
        ft_noprefix.to_csv(tmp_ft_path_noprefix, sep="\t")

        ft, _ = load_data(self.tmp_md_path, tmp_ft_path_noprefix)
        assert set([i[0] for i in ft.columns.tolist()]) == {"F"}

    def test_filter_merge_n_sort_w_filter(self):
        obs = filter_merge_n_sort(
            self.md,
            self.ft_rel,
            host_id="host_id",
            target="supertarget",
            filter_md_cols=["host_id", "supertarget"],
        )

        exp = self.data_rel.drop("covariate", axis=1).sort_values(
            ["host_id", "supertarget"]
        )
        exp = exp[["host_id", "supertarget", "F0", "F1"]]

        pd.testing.assert_frame_equal(obs, exp)

    def test_filter_merge_n_sort_no_filter(self):
        obs = filter_merge_n_sort(
            self.md,
            self.ft_rel,
            host_id="host_id",
            target="supertarget",
        )

        exp = self.data_rel.sort_values(["host_id", "supertarget"])
        exp = exp[["host_id", "supertarget", "covariate", "F0", "F1"]]

        pd.testing.assert_frame_equal(obs, exp)

    def test_split_data_by_host(self):
        train_obs, test_obs = split_data_by_host(self.data_rel, "host_id", 0.5, 123)

        train_exp = self.data_rel.iloc[[0, 2], :].copy()
        test_exp = self.data_rel.iloc[[1, 3], :].copy()

        assert_frame_equal(train_obs, train_exp)
        assert_frame_equal(test_obs, test_exp)

        overlap = [
            x
            for x in train_obs["host_id"].unique()
            if x in test_obs["host_id"].unique()
        ]
        assert len(overlap) == 0

    def test_split_data_by_host_error_one_host(self):
        data = pd.DataFrame(
            {"host_id": ["c", "c", "c", "c"], "supertarget": [1, 2, 1, 2]}
        )
        with self.assertRaisesRegex(
            ValueError, "Only one unique host available in dataset."
        ):
            split_data_by_host(data, "host_id", 0.5, 123)

    def test_load_n_split_data(self):
        # Call the function with the test paths
        train_val, test, tax, tree_phylo = load_n_split_data(
            self.tmp_md_path,
            self.tmp_ft_rel_path,
            None,
            None,
            host_id="host_id",
            target="supertarget",
            train_size=0.8,
            seed=123,
            filter_md_cols=["host_id", "supertarget"],
        )

        # Check that the train + test dataframes are not empty
        self.assertFalse(train_val.empty)
        self.assertFalse(test.empty)

        # Check that the dataframes do not overlap
        overlap = pd.merge(train_val, test, how="inner")
        self.assertEqual(len(overlap), 0)

        # tax and phylo should be empty in this case
        self.assertTrue(tax.empty)
        self.assertTrue(tree_phylo.children == [])
