from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from parameterized import parameterized
from qiime2.plugin.testing import TestPluginBase
from sklearn.linear_model import LinearRegression

from q2_time.model import fit_model, save_predictions, split_data_by_host


class TestModel(TestPluginBase):
    package = "q2_time.test"

    def setUp(self):
        # called before every test
        super().setUp()
        self.data = pd.DataFrame(
            {
                "host_id": ["a", "b", "c", "c"],
                "F0": [0.12, 0.23, 0.33, 0.44],
                "F1": [0.1, 0.2, 0.3, 0.4],
                "supertarget": [1, 2, 5, 7],
            }
        )
        self.data.index = ["SR1", "SR2", "SR3", "SR4"]

    def test_split_data_by_host(self):
        train_obs, test_obs = split_data_by_host(self.data, "host_id", 0.5)

        train_exp = self.data.iloc[2:, :].copy()
        test_exp = self.data.iloc[:2, :].copy()

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

    def test_fit_model(self):
        # todo: add all other models as well with decorator
        model_type = "LinRegressor"  # RFRegressor
        trained_model = fit_model(
            self.data, "supertarget", ["F0", "F1"], model_type, 12
        )

        self.assertEqual(type(trained_model).__name__, "LinearRegression")
        # RandomForestRegressor

    # @patch("sklearn.model_selection.RandomizedSearchCV")
    @parameterized.expand([("RFRegressor", 1), ("LinRegressor", 0)])
    @patch("q2_time.model.RandomizedSearchCV")
    def test_fit_model_random_cv(self, model_type, exp_count, mocked_cv):
        # todo: add check not called with LinReg
        fit_model(self.data, "supertarget", ["F0", "F1"], model_type, 12)
        self.assertEqual(mocked_cv.call_count, exp_count)

    def test_save_predictions(self):
        # todo adjust for all existing models with mock decorator
        target = "supertarget"
        ls_feat = ["F0", "F1"]
        model = LinearRegression().fit(self.data[ls_feat], self.data[target])
        pred_obs = save_predictions(model, target, ls_feat, self.data)

        pred_exp = pd.DataFrame(columns=["true", "pred"], index=self.data.index)
        pred_exp["true"] = self.data[target].copy()
        # todo: adjust for all existing models
        pred_logreg = np.array([0.75, 2.25, 5.25, 6.75])
        pred_exp["pred"] = pred_logreg

        assert_frame_equal(pred_exp, pred_obs)
