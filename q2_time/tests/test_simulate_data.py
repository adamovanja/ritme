"""Testing data simulator"""
import pandas as pd
from pandas.testing import assert_frame_equal
from qiime2.plugin.testing import TestPluginBase

from q2_time.simulate_data import simulate_feature_table, simulate_metadata


class TestFeatureTableSimulation(TestPluginBase):
    """Test all related to simulate_feature_table"""

    package = "q2_time.tests"

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


class TestMetadataSimulation(TestPluginBase):
    """Test all related to simulate_metadata"""

    package = "q2_time.tests"

    def setUp(self):
        super().setUp()
        self.ls_index = ["SRR0", "SRR1", "SRR2", "SRR3", "SRR4"]
        self.feat_df = pd.DataFrame(
            {"F0": [0, 0, 2, 0, 0], "F1": [10, 0, 1, 2, 0]}, index=self.ls_index
        )

    def test_default_metadata(self):
        """Test default functionality"""
        act_md = simulate_metadata(self.feat_df, 3)
        # verify index
        self.assertEqual(self.ls_index, act_md.index.tolist())

        # verify hosts
        exp_hosts = ["A", "B", "C"]
        act_hosts = sorted(act_md.host_id.unique().tolist())
        self.assertEqual(exp_hosts, act_hosts)

    def test_too_many_hosts(self):
        """Test error raising with too many hosts"""
        with self.assertRaises(ValueError):
            simulate_metadata(self.feat_df, 100)

    def test_randomness(self):
        """Test seed functionality"""
        df_1 = simulate_metadata(self.feat_df, 3, 12)
        df_2 = simulate_metadata(self.feat_df, 3, 12)
        df_other = simulate_metadata(self.feat_df, 3, 99)

        # same
        assert_frame_equal(df_1, df_2)
        # not same
        with self.assertRaises(AssertionError):
            assert_frame_equal(df_1, df_other)
