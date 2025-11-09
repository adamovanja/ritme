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
    _generate_host_time_snapshots_from_df,
    _load_data,
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

    def test_split_data_grouped_train_size_zero(self):
        """train_size==0.0 => empty train, full test (grouped)."""
        train_obs, test_obs = _split_data_grouped(self.data_rel, "host_id", 0.0, 999)
        self.assertEqual(train_obs.shape[0], 0)
        # All columns preserved; test identical to original
        assert_frame_equal(test_obs, self.data_rel)
        # Ensure dtypes preserved (implicit via assert_frame_equal)

    def test_split_data_no_group_train_size_zero(self):
        """train_size==0.0 => empty train, full test (no grouping)."""
        train_obs, test_obs = _split_data_grouped(self.data_rel, None, 0.0, 555)
        self.assertEqual(train_obs.shape[0], 0)
        assert_frame_equal(test_obs, self.data_rel)


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
        # add F prefix
        train_exp.columns = [
            f"F{col}" if col not in self.md.columns else col
            for col in train_exp.columns
        ]
        test_exp.columns = [
            f"F{col}" if col not in self.md.columns else col for col in test_exp.columns
        ]
        # add __t0 suffix
        train_exp.columns = [f"{col}__t0" for col in train_exp.columns]
        test_exp.columns = [f"{col}__t0" for col in test_exp.columns]
        assert_frame_equal(train, train_exp)
        assert_frame_equal(test, test_exp)

    def test_split_train_test_absolute(self):
        ft_abx = self.ft_rel * 100
        with self.assertWarnsRegex(
            Warning, r".*table contains absolute instead of relative abundances"
        ):
            _, _ = split_train_test(self.md, ft_abx, "host_id", 0.5, 123)

    def test_split_train_test_group_by_missing_error(self):
        # Provide non-existent group_by_column
        with self.assertRaisesRegex(
            ValueError,
            r"Group by column 'foobar' not found in data.",
        ):
            _ = split_train_test(self.md, self.ft_rel, "foobar", 0.5, 123)

    def test_cli_split_train_test_absolute(self):
        with patch("sys.stdout", new=StringIO()) as stdout:
            output_path = self.tmpdir.name
            cli_split_train_test(
                output_path,
                self.tmp_md_path,
                self.tmp_ft_rel_path,
                group_by_column="host_id",
                train_size=0.5,
                seed=123,
            )
            self.assertIn(
                f"Train and test splits were saved in {output_path}.",
                stdout.getvalue().strip(),
            )

    def test_split_train_test_train_size_zero(self):
        """train_size==0.0 => empty train, full test with suffix/prefix handling."""
        train, test = split_train_test(self.md, self.ft_rel, "host_id", 0.0, 321)
        # Train empty
        self.assertEqual(train.shape[0], 0)
        # Build expected test DataFrame
        test_exp = self.data_rel.copy()
        # Feature columns already relative; ensure F prefix applied only to
        # microbial features
        test_exp.columns = [
            f"F{col}" if col in self.ft_rel.columns else col for col in test_exp.columns
        ]
        test_exp.columns = [f"{c}__t0" for c in test_exp.columns]
        assert_frame_equal(test, test_exp)
        # Columns of train should match columns of test
        self.assertListEqual(train.columns.tolist(), test.columns.tolist())


class TestSplitTrainTestTemporalSnapshots(unittest.TestCase):
    def setUp(self):
        super().setUp()
        # Two hosts with times 2 and 3 so only time=3 qualifies as t0 when s=1 (exclude)
        self.md = pd.DataFrame(
            {
                "host_id": ["h1", "h1", "h2", "h2"],
                "time": [2, 3, 2, 3],
                "supertarget": [1, 2, 3, 4],
                "covariate": [0, 1, 0, 1],
            },
            index=["H1_T2", "H1_T3", "H2_T2", "H2_T3"],
        )
        self.ft = pd.DataFrame(
            {
                "0": [0.6, 0.7, 0.5, 0.4],
                "1": [0.4, 0.3, 0.5, 0.6],
            },
            index=self.md.index,
        )

    def test_temporal_exclude_mode(self):
        train_val, test = split_train_test(
            self.md,
            self.ft,
            group_by_column="host_id",
            train_size=0.5,
            seed=123,
            time_col="time",
            host_col="host_id",
            n_prev=1,
            missing_mode="exclude",
        )
        # Only one t0 per host (time=3), so rows == number of hosts
        self.assertEqual(train_val.shape[0] + test.shape[0], 2)
        # Host grouping respected (no overlap)
        overlap = set(train_val["host_id__t0"]).intersection(set(test["host_id__t0"]))
        self.assertEqual(len(overlap), 0)
        # Feature columns have suffixes for t0 and t-1
        ft_cols = [c for c in train_val.columns if c.startswith("F")]
        self.assertTrue(any(c.endswith("__t0") for c in ft_cols))
        self.assertTrue(any(c.endswith("__t-1") for c in ft_cols))
        # Metadata columns suffixed
        self.assertIn("host_id__t0", train_val.columns)
        self.assertIn("supertarget__t0", train_val.columns)
        self.assertIn("covariate__t0", train_val.columns)

    def test_temporal_nan_mode(self):
        # Add a host with a single time (will yield NaNs for t-1)
        md2 = pd.DataFrame(
            {
                "host_id": ["h1", "h1", "h2"],
                "time": [2, 3, 5],
                "supertarget": [1, 2, 9],
                "covariate": [0, 1, 1],
            },
            index=["H1_T2", "H1_T3", "H2_T5"],
        )
        ft2 = pd.DataFrame(
            {
                "0": [0.6, 0.7, 0.2],
                "1": [0.4, 0.3, 0.8],
            },
            index=md2.index,
        )
        train_val, test = split_train_test(
            md2,
            ft2,
            group_by_column="host_id",
            train_size=0.67,
            seed=42,
            time_col="time",
            host_col="host_id",
            n_prev=1,
            missing_mode="nan",
        )

        # Build expected full merged dataframe (then split by indices)
        idx = ["H1_T2", "H1_T3", "H2_T5"]
        cols = [
            "host_id__t0",
            "time__t0",
            "supertarget__t0",
            "covariate__t0",
            "host_id__t-1",
            "time__t-1",
            "supertarget__t-1",
            "covariate__t-1",
            "F0__t0",
            "F1__t0",
            "F0__t-1",
            "F1__t-1",
        ]

        exp = pd.DataFrame(index=idx, columns=cols)
        # t0 metadata
        exp.loc[:, "host_id__t0"] = ["h1", "h1", "h2"]
        exp.loc[:, "time__t0"] = [2, 3, 5]
        exp.loc[:, "supertarget__t0"] = [1, 2, 9]
        exp.loc[:, "covariate__t0"] = [0, 1, 1]
        # t-1 metadata (NaNs for missing, host_id/time filled)
        exp.loc[:, "host_id__t-1"] = ["h1", "h1", "h2"]
        exp.loc[:, "time__t-1"] = [1, 2, 4]
        exp.loc[:, "supertarget__t-1"] = [pd.NA, 1, pd.NA]
        exp.loc[:, "covariate__t-1"] = [pd.NA, 0, pd.NA]
        # t0 features
        exp.loc[:, "F0__t0"] = [0.6, 0.7, 0.2]
        exp.loc[:, "F1__t0"] = [0.4, 0.3, 0.8]
        # t-1 features
        exp.loc[:, "F0__t-1"] = [pd.NA, 0.6, pd.NA]
        exp.loc[:, "F1__t-1"] = [pd.NA, 0.4, pd.NA]

        # Ensure numeric dtypes where appropriate
        for c in ["time__t0", "supertarget__t0", "covariate__t0", "time__t-1"]:
            exp[c] = exp[c].astype(int)
        for c in ["F0__t0", "F1__t0"]:
            exp[c] = exp[c].astype(float)

        # Expected split for seed=42 and grouping by host_id__t0:
        # train -> h1 rows (H1_T2, H1_T3); test -> h2 row (H2_T5)
        exp_train_val = exp.loc[["H1_T2", "H1_T3"]]
        exp_test = exp.loc[["H2_T5"]]

        assert_frame_equal(train_val, exp_train_val)
        assert_frame_equal(test, exp_test)


class TestGenerateHostTimeSnapshots(unittest.TestCase):
    def setUp(self):
        super().setUp()
        # Single host with times 2,3,5,8
        self.md = pd.DataFrame(
            {
                "host_id": ["h1", "h1", "h1", "h1"],
                "time": [2, 3, 5, 8],
                "cov": ["a", "b", "c", "d"],
            },
            index=["S2", "S3", "S5", "S8"],
        )
        # Features: 2 informative cols summing to 1.0, plus one all-zero col
        # to be removed
        self.ft = pd.DataFrame(
            {
                "0": [0.7, 0.2, 1.0, 0.4],
                "1": [0.3, 0.8, 0.0, 0.6],
                "2": [0.0, 0.0, 0.0, 0.0],
            },
            index=["S2", "S3", "S5", "S8"],
        )

    def test_nan_mode_sliding_windows_s1(self):
        md_list, ft_list = _generate_host_time_snapshots_from_df(
            self.md,
            self.ft,
            time_col="time",
            host_col="host_id",
            s=1,
            missing_mode="nan",
        )

        # Expect two snapshots: t0 and t-1
        self.assertEqual(len(md_list), 2)
        self.assertEqual(len(ft_list), 2)

        md_t0, md_t1 = md_list
        ft_t0, ft_t1 = ft_list

        # Canonical index must be t0 sample ids in ascending time order
        self.assertListEqual(md_t0.index.tolist(), ["S2", "S3", "S5", "S8"])
        self.assertListEqual(md_t1.index.tolist(), ["S2", "S3", "S5", "S8"])

        # t0 times equal original times
        self.assertListEqual(md_t0["time"].tolist(), [2, 3, 5, 8])
        # t-1 times should be t0-1 for each instance
        self.assertListEqual(md_t1["time"].tolist(), [1, 2, 4, 7])

        # Non-missing carry forward real rows: for t0=3, t-1=2 exists ->
        # cov from S2 ('a')
        self.assertEqual(md_t1.loc["S3", "cov"], "a")
        # Missing t-1 for t0=5 -> cov should be NA; features all NA
        self.assertTrue(pd.isna(md_t1.loc["S5", "cov"]))

        # Features should be prepared (renamed to F*, zero-only column removed)
        self.assertListEqual(sorted(ft_t0.columns.tolist()), ["F0", "F1"])
        self.assertListEqual(sorted(ft_t1.columns.tolist()), ["F0", "F1"])
        # Missing t-1 for t0=5/8 -> entire feature row NaN
        self.assertTrue(ft_t1.loc["S5"].isna().all())
        self.assertTrue(ft_t1.loc["S8"].isna().all())

    def test_exclude_mode_s1(self):
        md_list, ft_list = _generate_host_time_snapshots_from_df(
            self.md,
            self.ft,
            time_col="time",
            host_col="host_id",
            s=1,
            missing_mode="exclude",
        )

        md_t0, md_t1 = md_list

        # Only t0=3 has t-1 available (2). Others are excluded.
        # canonical index = t0 sample id S3
        for x in ft_list + md_list:
            self.assertEqual(x.shape[0], 1)
            self.assertListEqual(x.index.tolist(), ["S3"])
        self.assertEqual(md_t0.loc["S3", "time"], 3)
        self.assertEqual(md_t1.loc["S3", "time"], 2)

    def test_error_non_numeric_time(self):
        md_bad = self.md.copy()
        md_bad["time"] = ["x", "y", "z", "w"]
        with self.assertRaisesRegex(
            ValueError, r"Non-numeric times detected for host 'h1'"
        ):
            _ = _generate_host_time_snapshots_from_df(
                md_bad, self.ft, time_col="time", host_col="host_id", s=1
            )

    def test_exclude_mode_no_instances_error(self):
        # Construct md where no t0 has a complete window of size s=2
        md2 = pd.DataFrame(
            {
                "host_id": ["h1", "h1", "h2"],
                "time": [1, 4, 3],
            },
            index=["S1", "S4", "S3"],
        )
        ft2 = pd.DataFrame(
            {
                "0": [0.5, 0.5, 0.7],
                "1": [0.5, 0.5, 0.3],
            },
            index=["S1", "S4", "S3"],
        )
        with self.assertRaisesRegex(
            ValueError, r"No \(host, t0\) instances have complete contiguous windows"
        ):
            _ = _generate_host_time_snapshots_from_df(
                md2,
                ft2,
                time_col="time",
                host_col="host_id",
                s=2,
                missing_mode="exclude",
            )
