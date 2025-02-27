"""Testing data simulator"""

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from ritme.simulate_data import simulate_data, simulate_feature_table, simulate_metadata


class TestFeatureTableSimulation(unittest.TestCase):
    """Test all related to simulate_feature_table"""

    def test_default_feature_table(self):
        """Test default functionality"""
        n_feat = 2
        n_samples = 3
        act_df = simulate_feature_table(n_samples, n_feat)

        self.assertEqual(act_df.shape[0], n_samples)
        self.assertEqual(act_df.shape[1], n_feat)

    def test_randomness(self):
        """Test seed functionality"""
        n_feat = 2
        n_samples = 3
        my_seed = 99

        df_1 = simulate_feature_table(n_samples, n_feat, seed=my_seed)
        df_2 = simulate_feature_table(n_samples, n_feat, seed=my_seed)
        df_other = simulate_feature_table(n_samples, n_feat, seed=22)

        # same
        assert_frame_equal(df_1, df_2)
        # not same
        with self.assertRaises(AssertionError):
            assert_frame_equal(df_1, df_other)


class TestMetadataSimulation(unittest.TestCase):
    """Test all related to simulate_metadata"""

    def setUp(self):
        super().setUp()
        self.ls_index = ["SRR0", "SRR1", "SRR2", "SRR3", "SRR4"]
        self.feat_df = pd.DataFrame(
            {"F0": [0, 0, 2, 0, 0], "F1": [10, 0, 1, 2, 0]}, index=self.ls_index
        )

    def test_default_metadata(self):
        """Test default functionality"""
        act_md = simulate_metadata(self.feat_df, 3, "target")
        # verify index
        self.assertEqual(self.ls_index, act_md.index.tolist())

        # verify hosts
        exp_hosts = ["A", "B", "C"]
        act_hosts = sorted(act_md.unique_id.unique().tolist())
        self.assertEqual(exp_hosts, act_hosts)

    def test_too_many_hosts(self):
        """Test error raising with too many hosts"""
        with self.assertRaises(ValueError):
            simulate_metadata(self.feat_df, 100, "target")

    def test_randomness(self):
        """Test seed functionality"""
        df_1 = simulate_metadata(self.feat_df, 3, "target", 12)
        df_2 = simulate_metadata(self.feat_df, 3, "target", 12)
        df_other = simulate_metadata(self.feat_df, 3, "target", 99)

        # same
        assert_frame_equal(df_1, df_2)
        # not same
        with self.assertRaises(AssertionError):
            assert_frame_equal(df_1, df_other)

    def test_target_age_days(self):
        act_md = simulate_metadata(self.feat_df, 3, "age_days")

        self.assertTrue(all(act_md["age_days"] >= 0))
        self.assertTrue(all(act_md["age_days"] <= 2 * 365))

    def test_target_age_months(self):
        act_md = simulate_metadata(self.feat_df, 3, "age_months")

        self.assertTrue(all(act_md["age_months"] >= 0))
        self.assertTrue(all(act_md["age_months"] <= 2 * 12))


class TestDataSimulation(unittest.TestCase):
    def test_simulate_data(self):
        n_samples = 10
        n_feat = 5
        n_hosts = 3
        ft, md = simulate_data(n_samples, "age_days", n_feat, n_hosts)
        self.assertEqual(ft.shape[0], n_samples)
        self.assertEqual(ft.shape[1], n_feat)
        self.assertEqual(md["unique_id"].nunique(), n_hosts)
