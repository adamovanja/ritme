import os
import tempfile

import pandas as pd
from pandas.testing import assert_frame_equal
from qiime2.plugin.testing import TestPluginBase

from q2_time.process_data import load_data, merge_n_sort, split_data_by_host


class TestProcessData(TestPluginBase):
    package = "q2_time.test"

    def setUp(self):
        super().setUp()
        self.data = pd.DataFrame(
            {
                "host_id": ["c", "b", "c", "a"],
                "F0": [0.12, 0.23, 0.33, 0.44],
                "F1": [0.1, 0.2, 0.3, 0.4],
                "supertarget": [1, 2, 5, 7],
            }
        )
        self.data.index = ["SR1", "SR2", "SR3", "SR4"]
        self.md = self.data[["host_id", "supertarget"]]
        self.ft = self.data[["F0", "F1"]]

    def test_load_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_md_path = os.path.join(tmpdir, "test_md.tsv")
            tmp_ft_path = os.path.join(tmpdir, "test_ft.tsv")

            self.md.to_csv(tmp_md_path, sep="\t")
            self.ft.to_csv(tmp_ft_path, sep="\t")

            # Load the data from the temporary files
            ft, md = load_data(tmp_md_path, tmp_ft_path)

        pd.testing.assert_frame_equal(ft, self.ft)
        pd.testing.assert_frame_equal(md, self.md)

    def test_merge_n_sort(self):
        obs = merge_n_sort(self.md, self.ft, host_id="host_id", target="supertarget")

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

    def test_split_data_by_host(self):
        train_obs, test_obs = split_data_by_host(self.data, "host_id", 0.5)

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
            split_data_by_host(data, "host_id", 0.5)
