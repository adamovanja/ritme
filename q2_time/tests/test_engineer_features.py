import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from qiime2.plugin.testing import TestPluginBase
from scipy.stats.mstats import gmean

from q2_time.feature_space.transform_features import (
    PSEUDOCOUNT,
    alr,
    transform_features,
)


class TestTransformFeatures(TestPluginBase):
    package = "q2_time.test"

    # TODO: clean up these tests - lots of redundancy and
    # TODO: not all transforms are covered yet
    def _relative_abundances(self, ft):
        """Calculate relative frequency from absolute frequency"""
        return ft.div(ft.sum(axis=1), axis=0)

    def _clr_from_scratch(self, ft):
        """Calculate clr from scratch"""
        ft_gmean = gmean(ft, axis=1)
        ft_trans = pd.DataFrame(index=ft.index, columns=ft.columns, dtype=float)
        for i in ft.index:
            ft_trans.loc[i] = np.log(ft.iloc[i, :] / ft_gmean[i])
        return ft_trans

    def test_clr(self):
        """Tests default clr transformation"""
        ft = pd.DataFrame({"F0": [10.0, 20.0, 50.0], "F1": [20.0, 30.0, 5.0]})

        # expected
        ft = self._relative_abundances(ft)
        exp_ft = self._clr_from_scratch(ft)
        exp_ft = exp_ft.add_prefix("clr_")

        # observed
        obs_ft = transform_features(ft, "clr")

        assert_frame_equal(exp_ft, obs_ft)

    def test_clr_pseudocounts(self):
        """Tests clr transformation with pseudocounts introduced"""
        ft = pd.DataFrame({"F0": [10.0, 20.0, 0.0], "F1": [20.0, 0.0, 5.0]})

        # expected
        ft.replace(0.0, PSEUDOCOUNT, inplace=True)
        ft = self._relative_abundances(ft)
        exp_ft = self._clr_from_scratch(ft)
        exp_ft = exp_ft.add_prefix("clr_")

        # observed
        obs_ft = transform_features(ft, "clr")

        assert_frame_equal(exp_ft, obs_ft)

    def test_alr_transform(self):
        ft = pd.DataFrame(
            {
                "F0": [5.0, 20.0, 1.0],
                "F1": [20.0, 1.0, 5.0],
                "F2": [20.0, 1.0, 5.0],
                "F3": [10.0, 1.0, 5.0],
            }
        )
        ft_obs = transform_features(ft, "clr")
        print(ft_obs)

    def test_method_error(self):
        ft = pd.DataFrame({"F0": [10.0, 20.0, 50.0], "F1": [20.0, 30.0, 5.0]})
        method = "FancyTransform"
        with self.assertRaisesRegex(
            ValueError, f"Method {method} is not implemented yet."
        ):
            transform_features(ft, method)

    def test_alr(self):
        """Tests alr with assumed pseudo counts already implemented"""
        ft = pd.DataFrame(
            {"F0": [5.0, 20.0, 1.0], "F1": [20.0, 1.0, 5.0], "F2": [10.0, 1.0, 5.0]}
        )
        obs_ft = alr(ft, 2)
        exp_ft = pd.DataFrame({"F0": [0.5, 20.0, 0.2], "F1": [2.0, 1.0, 1.0]})
        assert_array_equal(exp_ft, obs_ft)
