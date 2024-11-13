import os
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch

import pandas as pd
import qiime2 as q2
from pandas.testing import assert_frame_equal

from ritme.split_train_test import (
    _ft_get_relative_abundance,
    _ft_remove_zero_features,
    _ft_rename_microbial_features,
    _load_data,
    _split_data_stratified,
    cli_split_train_test,
    split_train_test,
)


class TestFeatureTableHelpers(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.ft_raw = pd.DataFrame(
            {"0": [100, 10, 3, 5], "1": [20, 30, 0, 0], "2": [0, 0, 0, 0]}
        )
        self.ft_renamed = pd.DataFrame(
            {"F0": [100, 10, 3, 5], "F1": [20, 30, 0, 0], "F2": [0, 0, 0, 0]}
        )
        self.ft_zero_remov = pd.DataFrame({"F0": [100, 10, 3, 5], "F1": [20, 30, 0, 0]})
        self.ft_rel = pd.DataFrame(
            {
                "F0": [0.83333333333333333, 0.25, 1.0, 1.0],
                "F1": [0.16666666666666666, 0.75, 0.0, 0.0],
            }
        )

    def test_ft_rename_microbial_features(self):
        renamed = _ft_rename_microbial_features(self.ft_raw, "F")
        expected_columns = [f"F{x}" for x in self.ft_raw.columns]
        self.assertListEqual(renamed.columns.tolist(), expected_columns)

    def test_ft_rename_microbial_features_nothing_to_rename(self):
        renamed = _ft_rename_microbial_features(self.ft_renamed, "F")
        self.assertListEqual(renamed.columns.tolist(), self.ft_renamed.columns.tolist())

    def test_ft_remove_zero_features(self):
        with self.assertWarnsRegex(Warning, r".*all zero values: \['F2'\]"):
            removed = _ft_remove_zero_features(self.ft_renamed)
            pd.testing.assert_frame_equal(removed, self.ft_zero_remov)

    def test_ft_remove_zero_features_nothing_to_remove(self):
        removed = _ft_remove_zero_features(self.ft_zero_remov)
        pd.testing.assert_frame_equal(removed, self.ft_zero_remov)

    def test_get_relative_abundance(self):
        ft_rel_obs = _ft_get_relative_abundance(self.ft_zero_remov)
        pd.testing.assert_frame_equal(ft_rel_obs, self.ft_rel)


class TestDataHelpers(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.data_rel = pd.DataFrame(
            {
                "host_id": ["c", "b", "c", "a"],
                "F0": [0.83333333333333333, 0.25, 1.0, 1.0],
                "F1": [0.16666666666666666, 0.75, 0.0, 0.0],
                "supertarget": [1, 2, 5, 7],
                "covariate": [0, 1, 0, 1],
            }
        )
        self.data_rel.index = ["SR1", "SR2", "SR3", "SR4"]
        self.md = self.data_rel[["host_id", "supertarget", "covariate"]]
        self.ft_rel = self.data_rel[["F0", "F1"]]

        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_md_path = os.path.join(self.tmpdir.name, "test_md.tsv")
        self.md.to_csv(self.tmp_md_path, sep="\t")

        self.tmp_ft_rel_path = os.path.join(self.tmpdir.name, "test_ft_rel.tsv")
        self.ft_rel.to_csv(self.tmp_ft_rel_path, sep="\t")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_load_data_ft_tsv(self):
        md, ft_rel = _load_data(self.tmp_md_path, self.tmp_ft_rel_path)

        pd.testing.assert_frame_equal(ft_rel, self.ft_rel)
        pd.testing.assert_frame_equal(md, self.md)

    def test_load_data_ft_qza(self):
        art_ft = q2.Artifact.import_data("FeatureTable[RelativeFrequency]", self.ft_rel)
        tmp_ft_path_art = self.tmp_ft_rel_path.replace(".tsv", ".qza")
        art_ft.save(tmp_ft_path_art)

        md, ft = _load_data(self.tmp_md_path, tmp_ft_path_art)

        pd.testing.assert_frame_equal(ft, self.ft_rel)
        pd.testing.assert_frame_equal(md, self.md)

    def test_split_data_stratified_by_host(self):
        train_obs, test_obs = _split_data_stratified(self.data_rel, "host_id", 0.5, 123)

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
        stratify_by = "host_id"
        with self.assertRaisesRegex(
            ValueError,
            f"Only one unique value of '{stratify_by}' available in dataset.",
        ):
            _split_data_stratified(data, stratify_by, 0.5, 123)


class TestMainFunctions(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.data_rel = pd.DataFrame(
            {
                "host_id": ["c", "b", "c", "a"],
                "supertarget": [1, 2, 5, 7],
                "covariate": [0, 1, 0, 1],
                "F0": [0.83333333333333333, 0.25, 1.0, 1.0],
                "F1": [0.16666666666666666, 0.75, 0.0, 0.0],
            }
        )
        self.data_rel.index = ["SR1", "SR2", "SR3", "SR4"]
        self.md = self.data_rel[["host_id", "supertarget", "covariate"]]
        self.ft_rel = self.data_rel[["F0", "F1"]]
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_md_path = os.path.join(self.tmpdir.name, "test_md.tsv")
        self.md.to_csv(self.tmp_md_path, sep="\t")

        self.tmp_ft_rel_path = os.path.join(self.tmpdir.name, "test_ft_rel.tsv")
        self.ft_rel.to_csv(self.tmp_ft_rel_path, sep="\t")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_split_train_test_relative(self):
        train, test = split_train_test(self.md, self.ft_rel, "host_id", "F", 0.5, 123)
        train_exp = self.data_rel.iloc[[0, 2], :].copy()
        test_exp = self.data_rel.iloc[[1, 3], :].copy()

        assert_frame_equal(train, train_exp)
        assert_frame_equal(test, test_exp)

    def test_split_train_test_absolute(self):
        ft_abx = self.ft_rel * 100
        with self.assertWarnsRegex(
            Warning, r".*table contains absolute instead of relative abundances"
        ):
            _, _ = split_train_test(self.md, ft_abx, "host_id", "F", 0.5, 123)

    def test_cli_split_train_test_absolute(self):
        with patch("sys.stdout", new=StringIO()) as stdout:
            output_path = self.tmpdir.name
            cli_split_train_test(
                output_path,
                self.tmp_md_path,
                self.tmp_ft_rel_path,
                "host_id",
                "F",
                0.5,
                123,
            )
            self.assertIn(
                f"Train and test splits were saved in {output_path}.",
                stdout.getvalue().strip(),
            )
