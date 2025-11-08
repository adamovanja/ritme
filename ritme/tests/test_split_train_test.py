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
    _load_data_multi,
    _split_data_grouped,
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

    def test_split_data_grouped_by_host(self):
        train_obs, test_obs = _split_data_grouped(self.data_rel, "host_id", 0.5, 123)

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

    def test_split_data_grouped_seeded_not_equal(self):
        train_obs1, test_obs1 = _split_data_grouped(self.data_rel, "host_id", 0.5, 123)
        train_obs2, test_obs2 = _split_data_grouped(self.data_rel, "host_id", 0.5, 246)

        with self.assertRaises(AssertionError):
            assert_frame_equal(train_obs1, train_obs2)

        with self.assertRaises(AssertionError):
            assert_frame_equal(test_obs1, test_obs2)

    def test_split_data_grouped_seeded_equal(self):
        train_obs1, test_obs1 = _split_data_grouped(self.data_rel, "host_id", 0.5, 123)
        train_obs2, test_obs2 = _split_data_grouped(self.data_rel, "host_id", 0.5, 123)

        assert_frame_equal(train_obs1, train_obs2)
        assert_frame_equal(test_obs1, test_obs2)

    def test_split_data_by_host_error_one_host(self):
        data = pd.DataFrame(
            {"host_id": ["c", "c", "c", "c"], "supertarget": [1, 2, 1, 2]}
        )
        group_by = "host_id"
        with self.assertRaisesRegex(
            ValueError,
            f"Only one unique value of '{group_by}' available in dataset.",
        ):
            _split_data_grouped(data, group_by, 0.5, 123)

    def test_split_data_by_no_group(self):
        train_obs, test_obs = _split_data_grouped(self.data_rel, None, 0.5, 123)

        assert_frame_equal(train_obs, self.data_rel.iloc[[1, 2], :])
        assert_frame_equal(test_obs, self.data_rel.iloc[[3, 0], :])


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
        train, test = split_train_test(self.md, self.ft_rel, "host_id", 0.5, 123)
        train_exp = self.data_rel.iloc[[0, 2], :].copy()
        test_exp = self.data_rel.iloc[[1, 3], :].copy()
        # add prefix "F" to expected
        train_exp.columns = [
            f"F{col}" if col not in self.md.columns else col
            for col in train_exp.columns
        ]
        test_exp.columns = [
            f"F{col}" if col not in self.md.columns else col for col in test_exp.columns
        ]
        assert_frame_equal(train, train_exp)
        assert_frame_equal(test, test_exp)

    def test_split_train_test_absolute(self):
        ft_abx = self.ft_rel * 100
        with self.assertWarnsRegex(
            Warning, r".*table contains absolute instead of relative abundances"
        ):
            _, _ = split_train_test(self.md, ft_abx, "host_id", 0.5, 123)

    def test_cli_split_train_test_absolute(self):
        with patch("sys.stdout", new=StringIO()) as stdout:
            output_path = self.tmpdir.name
            cli_split_train_test(
                output_path,
                self.tmp_md_path,
                self.tmp_ft_rel_path,
                "host_id",
                0.5,
                123,
            )
            self.assertIn(
                f"Train and test splits were saved in {output_path}.",
                stdout.getvalue().strip(),
            )


class TestMultiSnapshotFunctions(unittest.TestCase):
    def setUp(self):
        super().setUp()
        # two snapshots metadata (same samples)
        self.md_t0 = pd.DataFrame(
            {
                "host_id": ["h1", "h2"],
                "supertarget": [1.0, 2.0],
                "covariate": [0, 1],
            },
            index=["SR1", "SR2"],
        )
        self.md_t1 = pd.DataFrame(
            {
                "host_id": ["h1", "h2"],
                "supertarget": [1.5, 2.5],
                "covariate": [1, 0],
            },
            index=["SR1", "SR2"],
        )
        # feature tables: t0 already relative, t-1 absolute (will be converted)
        self.ft_t0 = pd.DataFrame(
            {
                "0": [0.6, 0.2],
                "1": [0.4, 0.8],
            },
            index=["SR1", "SR2"],
        )
        self.ft_t1_abs = pd.DataFrame(
            {
                "0": [60, 20],
                "1": [40, 80],
            },
            index=["SR1", "SR2"],
        )

        # mismatch metadata for error test
        self.md_t1_mismatch = self.md_t1.copy()
        self.md_t1_mismatch.index = ["SR3", "SR4"]

        # temp directory for CLI test
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_md_t0_path = os.path.join(self.tmpdir.name, "md_t0.tsv")
        self.tmp_md_t1_path = os.path.join(self.tmpdir.name, "md_t1.tsv")
        self.md_t0.to_csv(self.tmp_md_t0_path, sep="\t")
        self.md_t1.to_csv(self.tmp_md_t1_path, sep="\t")

        self.tmp_ft_t0_path = os.path.join(self.tmpdir.name, "ft_t0.tsv")
        self.tmp_ft_t1_path = os.path.join(self.tmpdir.name, "ft_t1.tsv")
        self.ft_t0.to_csv(self.tmp_ft_t0_path, sep="\t")
        self.ft_t1_abs.to_csv(self.tmp_ft_t1_path, sep="\t")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_split_train_test_multi_snapshot_suffixing(self):
        train_val, test = split_train_test(
            [self.md_t0, self.md_t1], [self.ft_t0, self.ft_t1_abs], None, 0.5, 42
        )
        # Columns should be suffixed with time labels t0 and t-1
        self.assertTrue(any(col.endswith("__t0") for col in train_val.columns))
        self.assertTrue(any(col.endswith("__t-1") for col in train_val.columns))
        # Microbial feature columns start with F and have suffix
        ft_cols = [c for c in train_val.columns if c.startswith("F")]
        self.assertTrue(all("__t" in c for c in ft_cols))
        # Relative abundance conversion applied to second snapshot (absolute input)
        ft_t1_cols = [c for c in ft_cols if c.endswith("__t-1")]
        # Sum per sample across t-1 features should be 1.0
        sums_t1 = train_val[ft_t1_cols].sum(axis=1).round(3)
        self.assertTrue(sums_t1.eq(1.0).all())

    def test_split_train_test_multi_snapshot_index_mismatch_metadata(self):
        with self.assertRaisesRegex(
            ValueError, r"Indices of provided metadata dataframe at position 1"
        ):
            _ = split_train_test(
                [self.md_t0, self.md_t1_mismatch], [self.ft_t0, self.ft_t1_abs]
            )

    def test_split_train_test_multi_snapshot_index_mismatch_features(self):
        ft_t1_mismatch = self.ft_t1_abs.copy()
        ft_t1_mismatch.index = ["SR3", "SR4"]
        with self.assertRaisesRegex(
            ValueError, r"Indices of provided feature table dataframe at position 1"
        ):
            _ = split_train_test([self.md_t0, self.md_t1], [self.ft_t0, ft_t1_mismatch])

    def test_cli_split_train_test_multi_snapshot(self):
        out_dir = os.path.join(self.tmpdir.name, "out")
        md_paths = f"{self.tmp_md_t0_path},{self.tmp_md_t1_path}"
        ft_paths = f"{self.tmp_ft_t0_path},{self.tmp_ft_t1_path}"
        with patch("sys.stdout", new=StringIO()) as stdout:
            cli_split_train_test(out_dir, md_paths, ft_paths, None, 0.5, 101)
            self.assertIn(
                f"Train and test splits were saved in {out_dir}.",
                stdout.getvalue().strip(),
            )
        # Verify files written
        self.assertTrue(os.path.exists(os.path.join(out_dir, "train_val.pkl")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "test.pkl")))

    def test_split_train_test_mismatched_types_error(self):
        # md is a sequence, ft is a DataFrame -> should error
        with self.assertRaisesRegex(
            ValueError,
            r"md and ft must both be DataFrames or both be sequences of DataFrames.",
        ):
            _ = split_train_test([self.md_t0, self.md_t1], self.ft_t0)

        # md is a DataFrame, ft is a sequence -> should error
        with self.assertRaisesRegex(
            ValueError,
            r"md and ft must both be DataFrames or both be sequences of DataFrames.",
        ):
            _ = split_train_test(self.md_t0, [self.ft_t0, self.ft_t1_abs])

    def test_load_data_multi_mismatched_counts_error(self):
        # Mismatched number of metadata and feature table paths
        paths_md = ["a.tsv", "b.tsv"]
        paths_ft = ["c.tsv"]
        with self.assertRaisesRegex(
            ValueError, r"Number of metadata and feature table paths must match."
        ):
            _ = _load_data_multi(paths_md, paths_ft)
