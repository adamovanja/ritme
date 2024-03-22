import os
import tempfile

import pandas as pd
import qiime2 as q2
from pandas.testing import assert_frame_equal
from qiime2.plugin.testing import TestPluginBase

from q2_ritme.process_data import (
    filter_merge_n_sort,
    load_data,
    load_n_split_data,
    split_data_by_host,
)


class TestProcessData(TestPluginBase):
    package = "q2_ritme.test"

    def setUp(self):
        super().setUp()
        self.data = pd.DataFrame(
            {
                "host_id": ["c", "b", "c", "a"],
                "F0": [0.12, 0.23, 0.33, 0.44],
                "F1": [0.1, 0.2, 0.3, 0.4],
                "supertarget": [1, 2, 5, 7],
                "covariate": [0, 1, 0, 1],
            }
        )
        self.data.index = ["SR1", "SR2", "SR3", "SR4"]
        self.md = self.data[["host_id", "supertarget", "covariate"]]
        self.ft = self.data[["F0", "F1"]]

        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_md_path = os.path.join(self.tmpdir.name, "test_md.tsv")
        self.tmp_ft_path = os.path.join(self.tmpdir.name, "test_ft.tsv")
        self.md.to_csv(self.tmp_md_path, sep="\t")
        self.ft.to_csv(self.tmp_ft_path, sep="\t")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_load_data_ft_tsv(self):
        # Load the data from the temporary files
        ft, md = load_data(self.tmp_md_path, self.tmp_ft_path)

        pd.testing.assert_frame_equal(ft, self.ft)
        pd.testing.assert_frame_equal(md, self.md)

    def test_load_data_ft_qza(self):
        art_ft = q2.Artifact.import_data("FeatureTable[Frequency]", self.ft)
        tmp_ft_path_art = self.tmp_ft_path.replace(".tsv", ".qza")
        art_ft.save(tmp_ft_path_art)
        # Load the data from the temporary files
        ft, md = load_data(self.tmp_md_path, tmp_ft_path_art)

        pd.testing.assert_frame_equal(ft, self.ft)
        pd.testing.assert_frame_equal(md, self.md)

    def test_load_data_simulated(self):
        ft, md = load_data()
        assert ft.shape[0] == 1000
        assert md.shape[0] == 1000

    def test_load_data_no_feature_prefix(self):
        tmp_ft_path_noprefix = self.tmp_ft_path.replace(".tsv", "_noprefix.tsv")
        ft_noprefix = self.ft.rename(columns={"F0": "0", "F1": "1"})
        ft_noprefix.to_csv(tmp_ft_path_noprefix, sep="\t")

        ft, _ = load_data(self.tmp_md_path, tmp_ft_path_noprefix)
        assert set([i[0] for i in ft.columns.tolist()]) == {"F"}

    def test_filter_merge_n_sort_w_filter(self):
        obs = filter_merge_n_sort(
            self.md,
            self.ft,
            host_id="host_id",
            target="supertarget",
            filter_md=["host_id", "supertarget"],
        )

        exp = pd.DataFrame(
            {
                "host_id": ["a", "b", "c", "c"],
                "supertarget": [7, 2, 1, 5],
                "F0": [0.44, 0.23, 0.12, 0.33],
                "F1": [0.4, 0.2, 0.1, 0.3],
            },
            index=["SR4", "SR2", "SR1", "SR3"],
        )

        pd.testing.assert_frame_equal(obs, exp)

    def test_filter_merge_n_sort_no_filter(self):
        obs = filter_merge_n_sort(
            self.md,
            self.ft,
            host_id="host_id",
            target="supertarget",
        )

        exp = pd.DataFrame(
            {
                "host_id": ["a", "b", "c", "c"],
                "supertarget": [7, 2, 1, 5],
                "covariate": [1, 1, 0, 0],
                "F0": [0.44, 0.23, 0.12, 0.33],
                "F1": [0.4, 0.2, 0.1, 0.3],
            },
            index=["SR4", "SR2", "SR1", "SR3"],
        )

        pd.testing.assert_frame_equal(obs, exp)

    def test_split_data_by_host(self):
        train_obs, test_obs = split_data_by_host(self.data, "host_id", 0.5, 123)

        train_exp = self.data.iloc[[0, 2], :].copy()
        test_exp = self.data.iloc[[1, 3], :].copy()

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
        train_val, test = load_n_split_data(
            self.tmp_md_path,
            self.tmp_ft_path,
            host_id="host_id",
            target="supertarget",
            train_size=0.8,
            seed=123,
            filter_md=["host_id", "supertarget"],
        )

        # Check that the dataframes are not empty
        self.assertFalse(train_val.empty)
        self.assertFalse(test.empty)

        # Check that the dataframes do not overlap
        overlap = pd.merge(train_val, test, how="inner")
        self.assertEqual(len(overlap), 0)
