"""Testing static trainables"""
import os
import tempfile
from unittest.mock import patch

import numpy as np
from qiime2.plugin.testing import TestPluginBase
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from q2_time.model_space import _static_trainables as st


class TestHelperFunctions(TestPluginBase):
    """Test all related to helper functions used by static_trainables"""

    package = "q2_time.tests"

    def setUp(self):
        super().setUp()
        self.X = np.array([[1, 2], [3, 4]])
        self.y = np.array([1, 2])
        self.model = LinearRegression().fit(self.X, self.y)

    def test_predict_rmse(self):
        expected = mean_squared_error(self.y, self.model.predict(self.X), squared=False)
        result = st._predict_rmse(self.model, self.X, self.y)
        self.assertEqual(result, expected)

    @patch("ray.tune.get_trial_dir")
    def test_save_sklearn_model(self, mock_get_trial_dir):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_get_trial_dir.return_value = tmpdir

            result = st._save_sklearn_model(self.model)
            self.assertTrue(os.path.exists(result))

    @patch("ray.air.session.report")
    @patch("ray.tune.get_trial_dir")
    def test_report_results_manually(self, mock_get_trial_dir, mock_report):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_get_trial_dir.return_value = tmpdir
            st._report_results_manually(self.model, self.X, self.y, self.X, self.y)
            mock_report.assert_called_once()
