from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from qiime2.plugin.testing import TestPluginBase
from scipy.stats.mstats import gmean
from skbio.stats.composition import ilr

from q2_time.feature_space._process_train import process_train
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


class TestProcessTrain(TestPluginBase):
    package = "q2_time.test"

    def setUp(self):
        super().setUp()
        self.config = {"data_transform": None, "data_alr_denom_idx": False}
        self.train_val = pd.DataFrame(
            {
                "host_id": ["c", "b", "c", "a"],
                "target": [1, 2, 1, 2],
                "F0": [0.12, 0.23, 0.33, 0.44],
                "F1": [0.1, 0.2, 0.3, 0.4],
            },
            index=["SR1", "SR2", "SR3", "SR4"],
        )
        self.target = "target"
        self.host_id = "host_id"
        self.seed_data = 0


def _assert_called_with_df(mock, expected_df, *expected_args):
    mock.assert_called_once()
    args, _ = mock.call_args
    pd.testing.assert_frame_equal(args[0], expected_df)
    for expected, actual in zip(expected_args, args[1:]):
        assert expected == actual, f"Expected {expected}, but got {actual}"


@patch("q2_time.feature_space._process_train.transform_features")
@patch("q2_time.feature_space._process_train.split_data_by_host")
def test_process_train(self, mock_split_data_by_host, mock_transform_features):
    # Arrange
    ls_ft = ["F0", "F1"]
    ft = self.train_val[ls_ft]
    mock_transform_features.return_value = ft
    mock_split_data_by_host.return_value = (
        self.train_val.iloc[:2, :],
        self.train_val.iloc[2:, :],
    )

    # Act
    X_train, y_train, X_val, y_val = process_train(
        self.config, self.train_val, self.target, self.host_id, self.seed_data
    )

    # Assert
    _assert_called_with_df(mock_transform_features, ft, None, False)
    _assert_called_with_df(
        mock_split_data_by_host,
        self.train_val[[self.target, self.host_id] + ls_ft],
        "host_id",
        0.8,
        0,
    )
