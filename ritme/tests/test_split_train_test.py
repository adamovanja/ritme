import os
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from ritme.split_train_test import (
    _ft_get_relative_abundance,
    _ft_remove_zero_features,
    _ft_rename_microbial_features,
    _generate_host_time_snapshots_from_df,
    _load_data,
    _make_kfold_splitter,
    _split_data_grouped,
    adaptive_k_folds,
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
        with self.assertWarnsRegex(Warning, r".*all zero values.*\['F2'\]"):
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

    def test_split_data_stratify_single_column(self):
        """Stratify by a single categorical column preserves distribution."""
        # Use 'covariate' for stratification
        train_obs, test_obs = _split_data_grouped(
            self.data_rel,
            group_by_column=None,
            train_size=0.5,
            seed=321,
            stratify_by=["covariate"],
        )
        # All unique classes must appear in both splits (each class has >=2 samples)
        cov_unique = set(self.data_rel["covariate"].unique())
        cov_train = set(train_obs["covariate"].unique())
        cov_test = set(test_obs["covariate"].unique())
        self.assertTrue(cov_unique == cov_train == cov_test)

    def test_split_data_stratify_multi_column(self):
        """Stratify by joint distribution of two columns."""
        # Create a small dataset with joint categories
        df = pd.DataFrame(
            {
                "A": ["x", "x", "y", "y"],
                "B": [1, 1, 2, 2],
                "val": [10, 11, 12, 13],
            }
        )
        train_obs, test_obs = _split_data_grouped(
            df,
            None,
            0.5,
            7,
            stratify_by=["A", "B"],
        )
        # Each joint category count is 2; 50% split should keep one instance
        # per category in train and one in test.
        joint_train = train_obs.apply(lambda r: f"{r['A']}_{r['B']}", axis=1).tolist()
        joint_test = test_obs.apply(lambda r: f"{r['A']}_{r['B']}", axis=1).tolist()
        self.assertEqual(len(joint_train), 2)
        self.assertEqual(len(joint_test), 2)
        # No overlap duplicates within a split
        self.assertEqual(len(set(joint_train)), 2)
        self.assertEqual(len(set(joint_test)), 2)
        # Combined cover all joint classes present in original data
        self.assertEqual(set(joint_train + joint_test), {"x_1", "y_2"})

    def test_split_data_stratify_missing_column_error(self):
        with self.assertRaisesRegex(ValueError, r"Stratification columns not found"):
            _split_data_grouped(
                self.data_rel,
                None,
                0.5,
                111,
                stratify_by=["does_not_exist"],
            )

    def test_split_data_group_and_stratify_error(self):
        # Grouped+stratified requires >=2 groups per class. Here class 0 has only
        # one host ('c'), so expect an error from sklearn.
        with self.assertRaises(ValueError):
            _split_data_grouped(
                self.data_rel,
                group_by_column="host_id",
                train_size=0.5,
                seed=222,
                stratify_by=["covariate"],
            )

    def test_grouped_stratify_inconsistent_group_error(self):
        # Create groups where stratify columns vary within a group -> error
        df = pd.DataFrame(
            {
                "host": ["h1", "h1", "h2", "h2"],
                "label": [0, 1, 0, 0],
                "x": [1.0, 2.0, 3.0, 4.0],
            }
        )
        with self.assertRaisesRegex(ValueError, r"must be constant within each group"):
            _split_data_grouped(
                df,
                group_by_column="host",
                train_size=0.5,
                seed=1,
                stratify_by=["label"],
            )

    def test_grouped_stratify_simple_balance(self):
        # Build dataset with 4 hosts, two per class; 50% split should yield 1 per class
        df = pd.DataFrame(
            {
                "host": ["h1", "h2", "h3", "h4"],
                "label": [0, 0, 1, 1],
                "feat": [10, 11, 12, 13],
            }
        )
        # Duplicate rows per host to simulate multiple samples; same label within group
        df = pd.concat([df, df], ignore_index=True)
        train_obs, test_obs = _split_data_grouped(
            df, group_by_column="host", train_size=0.5, seed=42, stratify_by=["label"]
        )
        # Check group separation
        self.assertEqual(
            len(set(train_obs["host"]).intersection(set(test_obs["host"]))), 0
        )
        # Check class balance per split (1 host per class)
        train_labels = train_obs.drop_duplicates("host")["label"].value_counts()
        test_labels = test_obs.drop_duplicates("host")["label"].value_counts()
        self.assertEqual(train_labels.get(0, 0), 1)
        self.assertEqual(train_labels.get(1, 0), 1)
        self.assertEqual(test_labels.get(0, 0), 1)
        self.assertEqual(test_labels.get(1, 0), 1)

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
        # add F prefix to feature columns (split_train_test renames them)
        train_exp.columns = [
            f"F{col}" if col not in self.md.columns else col
            for col in train_exp.columns
        ]
        test_exp.columns = [
            f"F{col}" if col not in self.md.columns else col for col in test_exp.columns
        ]
        # Static data has no time suffixes
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
            r"Column 'foobar' not found in data",
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
        """train_size==0.0 => empty train, full test with prefix handling."""
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
        # Static data has no time suffixes
        assert_frame_equal(test, test_exp)
        # Columns of train should match columns of test
        self.assertListEqual(train.columns.tolist(), test.columns.tolist())

    def test_split_train_test_public_stratify_non_grouped(self):
        # Stratify by covariate; ensure both classes appear in both splits
        train, test = split_train_test(
            self.md,
            self.ft_rel,
            group_by_column=None,
            train_size=0.5,
            seed=999,
            stratify_by=["covariate"],
        )
        # Static data has no time suffixes - covariate column is just "covariate"
        cov_train = set(train["covariate"])
        cov_test = set(test["covariate"])
        self.assertEqual(cov_train, {0, 1})
        self.assertEqual(cov_test, {0, 1})

    def test_split_train_test_public_stratify_grouped_error(self):
        # With grouping, class 0 has only one host in fixture -> expect error
        with self.assertRaises(ValueError):
            _ = split_train_test(
                self.md,
                self.ft_rel,
                group_by_column="host_id",
                train_size=0.5,
                seed=123,
                stratify_by=["covariate"],
            )

    def test_split_train_test_static_drops_ghost_zero_features(self):
        """Features non-zero only in ft-only samples must not survive."""
        md = pd.DataFrame(
            {"host_id": ["a", "b"], "target": [1, 2]},
            index=["s1", "s2"],
        )
        ft = pd.DataFrame(
            {"0": [0.5, 0.5, 0.0], "1": [0.5, 0.5, 0.0], "2": [0.0, 0.0, 1.0]},
            index=["s1", "s2", "s3"],
        )
        train, test = split_train_test(md, ft, train_size=0.0, seed=0)
        self.assertNotIn("F2", test.columns)
        self.assertIn("F0", test.columns)
        self.assertIn("F1", test.columns)


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
        # Host grouping respected (no overlap) - t0 columns are unsuffixed
        overlap = set(train_val["host_id"]).intersection(set(test["host_id"]))
        self.assertEqual(len(overlap), 0)
        # Feature columns: t0 unsuffixed, t-1 suffixed
        ft_cols = [c for c in train_val.columns if c.startswith("F")]
        self.assertTrue(any(not c.endswith("__t-1") for c in ft_cols))  # t0 unsuffixed
        self.assertTrue(any(c.endswith("__t-1") for c in ft_cols))
        # t0 metadata columns are unsuffixed
        self.assertIn("host_id", train_val.columns)
        self.assertIn("supertarget", train_val.columns)
        self.assertIn("covariate", train_val.columns)

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
        # t0 columns are unsuffixed; only past snapshots (t-1) get suffixes
        idx = ["H1_T2", "H1_T3", "H2_T5"]

        exp = pd.DataFrame(
            {
                "host_id": ["h1", "h1", "h2"],
                "time": [2, 3, 5],
                "supertarget": [1, 2, 9],
                "covariate": [0, 1, 1],
                "host_id__t-1": ["h1", "h1", "h2"],
                "time__t-1": [1, 2, 4],
                "supertarget__t-1": [np.nan, 1.0, np.nan],
                "covariate__t-1": [np.nan, 0.0, np.nan],
                "F0": [0.6, 0.7, 0.2],
                "F1": [0.4, 0.3, 0.8],
                "F0__t-1": [np.nan, 0.6, np.nan],
                "F1__t-1": [np.nan, 0.4, np.nan],
            },
            index=idx,
        )
        for c in ["time", "supertarget", "covariate", "time__t-1"]:
            exp[c] = exp[c].astype(int)

        # Expected split for seed=42 and grouping by host_id:
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


class TestMakeKfoldSplitter(unittest.TestCase):
    """The K-fold splitter must honor every constraint that
    :func:`_split_data_grouped` enforces for the train/test split: groups
    never span the train/val boundary, stratification preserves the joint
    label distribution across folds, the "constant within group" assertion
    is reused, and the splits are deterministic given the seed."""

    def setUp(self):
        # 30 hosts × 4 rows each. Stratify label A/B/C is constant per host.
        self.n_groups = 30
        self.host_ids = np.repeat(np.arange(self.n_groups), 4)
        self.strata = np.array([["A", "B", "C"][g % 3] for g in self.host_ids])
        self.df = pd.DataFrame(
            {
                "host_id": self.host_ids,
                "state": self.strata,
                "F1": np.arange(len(self.host_ids), dtype=float),
            }
        )

    def test_neither_group_nor_stratify(self):
        folds = _make_kfold_splitter(self.df, None, None, n_splits=5, seed=42)
        self.assertEqual(len(folds), 5)
        # All val indices together cover every row exactly once.
        all_va = np.sort(np.concatenate([va for _, va in folds]))
        np.testing.assert_array_equal(all_va, np.arange(len(self.df)))

    def test_group_only_no_overlap(self):
        folds = _make_kfold_splitter(self.df, "host_id", None, n_splits=5, seed=42)
        for tr, va in folds:
            tr_groups = set(self.df.iloc[tr]["host_id"])
            va_groups = set(self.df.iloc[va]["host_id"])
            self.assertEqual(tr_groups & va_groups, set())

    def test_stratify_only_preserves_distribution(self):
        folds = _make_kfold_splitter(self.df, None, ["state"], n_splits=5, seed=42)
        for tr, va in folds:
            va_dist = self.df.iloc[va]["state"].value_counts(normalize=True)
            for label, frac in va_dist.items():
                self.assertAlmostEqual(frac, 1 / 3, delta=0.05)

    def test_group_and_stratify_no_overlap_and_balanced(self):
        folds = _make_kfold_splitter(self.df, "host_id", ["state"], n_splits=5, seed=42)
        for tr, va in folds:
            tr_groups = set(self.df.iloc[tr]["host_id"])
            va_groups = set(self.df.iloc[va]["host_id"])
            self.assertEqual(tr_groups & va_groups, set())
            va_dist = self.df.iloc[va]["state"].value_counts(normalize=True)
            for label, frac in va_dist.items():
                self.assertAlmostEqual(frac, 1 / 3, delta=0.1)

    def test_stratify_must_be_constant_within_group(self):
        bad = self.df.copy()
        bad.loc[bad["host_id"] == 0, "state"] = ["A", "B", "A", "B"]
        with self.assertRaisesRegex(ValueError, "constant within each group"):
            _make_kfold_splitter(bad, "host_id", ["state"], n_splits=5, seed=42)

    def test_too_few_groups_for_k_raises(self):
        small = self.df.iloc[:8].copy()  # only 2 unique groups
        with self.assertRaisesRegex(ValueError, "Cannot create 5-fold split"):
            _make_kfold_splitter(small, "host_id", None, n_splits=5, seed=42)

    def test_seed_determinism(self):
        a = _make_kfold_splitter(self.df, "host_id", None, n_splits=5, seed=42)
        b = _make_kfold_splitter(self.df, "host_id", None, n_splits=5, seed=42)
        for (tr_a, va_a), (tr_b, va_b) in zip(a, b):
            np.testing.assert_array_equal(np.sort(tr_a), np.sort(tr_b))
            np.testing.assert_array_equal(np.sort(va_a), np.sort(va_b))


class TestAdaptiveKFolds(unittest.TestCase):
    def test_uses_group_count_when_grouping(self):
        df = pd.DataFrame({"host_id": [1, 1, 2, 2, 3, 3], "F1": range(6)})
        # 3 unique groups -> capped to 3 folds.
        self.assertEqual(adaptive_k_folds(df, "host_id", None), 3)

    def test_caps_at_smallest_stratum(self):
        df = pd.DataFrame(
            {
                "host_id": list(range(20)),
                "state": ["A"] * 17 + ["B"] * 3,
                "F1": range(20),
            }
        )
        # Smallest stratum has 3 members -> K capped to 3.
        self.assertEqual(adaptive_k_folds(df, "host_id", ["state"]), 3)

    def test_honors_user_request(self):
        df = pd.DataFrame({"host_id": list(range(50)), "F1": range(50)})
        self.assertEqual(adaptive_k_folds(df, "host_id", None, requested=7), 7)
        # Requested above available groups gets capped.
        self.assertEqual(adaptive_k_folds(df, "host_id", None, requested=200), 50)

    def test_missing_group_column_falls_back(self):
        # When the group column is mocked away (caller passes a name that
        # is not in `data`), adaptive_k_folds must NOT raise: the splitter
        # itself will raise with a helpful message at trial-time. With the
        # missing column the helper falls back to row count, and at row
        # count >= DEFAULT_K_FOLDS the default applies unchanged.
        df = pd.DataFrame({"F1": range(10)})
        self.assertEqual(adaptive_k_folds(df, "missing_col", None), 5)

    def test_classification_target_caps_to_smallest_class(self):
        df = pd.DataFrame(
            {
                "F1": range(30),
                "label": ["x"] * 26 + ["y"] * 4,
            }
        )
        # No grouping or stratify_by -> the classification target itself caps K.
        self.assertEqual(
            adaptive_k_folds(
                df, None, None, target="label", task_type="classification"
            ),
            4,
        )

    def test_regression_target_does_not_cap_even_when_non_numeric(self):
        # Regression targets stored as strings (e.g. month="3", "10") must
        # NOT trigger a per-class stratum cap. Caught by smoke-testing
        # run_experiment_mlflow.sh on the Moving-Pictures dataset, where
        # `month` is object-dtype and the heuristic was clipping K to 3.
        df = pd.DataFrame(
            {
                "F1": range(30),
                "month": ["3"] * 8 + ["10"] * 7 + ["4"] * 6 + ["1"] * 3 + ["2"] * 6,
            }
        )
        self.assertEqual(
            adaptive_k_folds(df, None, None, target="month", task_type="regression"),
            5,  # default applies; no stratum cap
        )

    def test_default_is_five_for_typical_n(self):
        # 100 rows ungrouped: comfortably above the default and above any
        # stratum cap -> exactly the documented default applies.
        df = pd.DataFrame({"F1": range(100)})
        self.assertEqual(adaptive_k_folds(df, None, None), 5)

    def test_user_override_takes_precedence_over_default(self):
        df = pd.DataFrame({"F1": range(100)})
        self.assertEqual(adaptive_k_folds(df, None, None, requested=10), 10)
        self.assertEqual(adaptive_k_folds(df, None, None, requested=2), 2)

    def test_requested_one_returns_one_for_single_split_fallback(self):
        # Explicit "k_folds: 1" in the experiment config must reach the
        # trainable as 1 so the original single-split path activates. Earlier
        # the floor of 2 was applied unconditionally and silently upgraded to
        # K=2; this regression guards against that.
        df = pd.DataFrame(
            {
                "host_id": list(range(20)),
                "state": ["A"] * 17 + ["B"] * 3,  # would otherwise cap to 3
                "F1": range(20),
            }
        )
        self.assertEqual(adaptive_k_folds(df, "host_id", ["state"], requested=1), 1)
        # Zero / negative are also treated as "no K-fold" rather than upgraded.
        self.assertEqual(adaptive_k_folds(df, "host_id", None, requested=0), 1)
        self.assertEqual(adaptive_k_folds(df, "host_id", None, requested=-3), 1)

    def test_group_aware_stratum_cap_with_rare_class(self):
        """Group-aware stratum cap: the smallest-stratum cap counts unique
        groups (hosts) per class, not rows. Earlier the row-count was used,
        which silently over-allocated K when the rare class had many rows
        per host. With 8 rows for the rare class but only 2 unique hosts,
        the group-aware cap is 2 (not 5, the default; not 8, the row count).
        """
        # 30 hosts × 4 rows = 120 rows total. The rare stratum class "B"
        # has only 2 hosts (8 rows). Group-aware cap is 2.
        host_ids = []
        states = []
        f1 = []
        for h in range(28):
            host_ids.extend([h, h, h, h])
            states.extend(["A", "A", "A", "A"])
            f1.extend([h * 0.1, h * 0.1 + 0.01, h * 0.1 + 0.02, h * 0.1 + 0.03])
        for h in range(28, 30):
            host_ids.extend([h, h, h, h])
            states.extend(["B", "B", "B", "B"])
            f1.extend([h * 0.1, h * 0.1 + 0.01, h * 0.1 + 0.02, h * 0.1 + 0.03])
        df = pd.DataFrame({"host_id": host_ids, "state": states, "F1": f1})
        # Sanity: 30 hosts total, 4 rows per host, rare class has 2 hosts.
        self.assertEqual(df["host_id"].nunique(), 30)
        self.assertEqual(len(df), 120)
        # WITHOUT stratify_by, adaptive_k_folds caps to 30 (groups) -> 5 default.
        self.assertEqual(adaptive_k_folds(df, "host_id", None), 5)
        # WITH stratify_by ["state"], the group-level stratum cap activates:
        # rare class "B" has only 2 unique hosts -> K capped to 2.
        self.assertEqual(adaptive_k_folds(df, "host_id", ["state"]), 2)
