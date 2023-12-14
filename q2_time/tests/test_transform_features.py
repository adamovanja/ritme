import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from qiime2.plugin.testing import TestPluginBase
from scipy.stats.mstats import gmean
from skbio.stats.composition import ilr

from q2_time.feature_space.transform_features import (
    PSEUDOCOUNT,
    alr,
    transform_features,
)


class TestTransformFeatures(TestPluginBase):
    package = "q2_time.test"

    def setUp(self):
        super().setUp()
        self.ft = pd.DataFrame({"F0": [10.0, 20.0, 50.0], "F1": [20.0, 30.0, 5.0]})
        self.ft_zero = pd.DataFrame({"F0": [10.0, 20.0, 0.0], "F1": [20.0, 0.0, 5.0]})

    def _relative_abundances(self, ft):
        """Calculate relative frequency from absolute frequency"""
        return ft.div(ft.sum(axis=1), axis=0)

    def _clr_from_scratch(self, ft):
        """Calculate clr from scratch"""
        ft_gmean = gmean(ft, axis=1)
        ft_trans = pd.DataFrame(index=ft.index, columns=ft.columns, dtype=float)
        for i in ft.index:
            ft_trans.loc[i] = np.log(ft.loc[i, :] / ft_gmean[i])
        return ft_trans

    def test_alr(self):
        """Tests alr function"""
        # expected
        ft = self.ft.replace(0.0, PSEUDOCOUNT)
        denom = ft.iloc[:, 1]
        exp_ft = ft.div(denom, axis=0).drop(ft.columns[1], axis=1)

        # observed
        obs_ft = alr(ft, 1)

        assert_frame_equal(exp_ft, obs_ft)

    def test_transform_features_clr(self):
        """Tests default clr transformation"""
        # expected
        ft = self._relative_abundances(self.ft)
        exp_ft = self._clr_from_scratch(ft)
        exp_ft = exp_ft.add_prefix("clr_")

        # observed
        obs_ft = transform_features(self.ft, "clr")

        assert_frame_equal(exp_ft, obs_ft)

    def test_transform_features_clr_pseudocounts(self):
        """Tests clr transformation with pseudocounts introduced"""
        # expected
        ft = self.ft_zero.replace(0.0, PSEUDOCOUNT)
        ft = self._relative_abundances(ft)
        exp_ft = self._clr_from_scratch(ft)
        exp_ft = exp_ft.add_prefix("clr_")

        # observed
        obs_ft = transform_features(self.ft_zero, "clr")

        assert_frame_equal(exp_ft, obs_ft)

    def test_transform_features_alr(self):
        """Tests alr transformation"""
        # expected
        ft = self.ft.replace(0.0, PSEUDOCOUNT)
        exp_ft = alr(ft, 1)
        exp_ft = exp_ft.add_prefix("alr_")

        # observed
        obs_ft = transform_features(self.ft, "alr", 1)

        assert_frame_equal(exp_ft, obs_ft)

    def test_transform_features_ilr(self):
        """Tests ilr transformation"""
        # expected
        ft = self.ft.replace(0.0, PSEUDOCOUNT)
        ft = self._relative_abundances(ft)
        exp_ft = pd.DataFrame(
            ilr(ft),
            columns=[f"ilr_{i}" for i in range(ft.shape[1] - 1)],
            index=ft.index,
        )

        # observed
        obs_ft = transform_features(self.ft, "ilr")

        assert_frame_equal(exp_ft, obs_ft)

    def test_transform_features_none(self):
        """Tests no transformation"""
        # expected
        exp_ft = self.ft

        # observed
        obs_ft = transform_features(self.ft, None)

        assert_frame_equal(exp_ft, obs_ft)

    def test_transform_features_error(self):
        """Tests error when invalid method is provided"""
        with self.assertRaisesRegex(
            ValueError, "Method FancyTransform is not implemented yet."
        ):
            transform_features(self.ft, "FancyTransform")
