from qiime2.plugin.testing import TestPluginBase

from q2_time.model_space import _static_searchspace as ss


class TestStatisSearchSpace(TestPluginBase):
    """Test all static search space dictionaries"""

    package = "q2_time.tests"

    def setUp(self):
        super().setUp()

    def test_linreg_space_keys(self):
        required_keys = ["data_transform", "data_alr_denom_idx", "fit_intercept"]
        for key in required_keys:
            assert key in ss.linreg_space, f"Key '{key}' not found in linreg_space"

    def test_rf_space_keys(self):
        required_keys = [
            "data_transform",
            "data_alr_denom_idx",
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "min_impurity_decrease",
            "bootstrap",
        ]
        for key in required_keys:
            assert key in ss.rf_space, f"Key '{key}' not found in rf_space"

    def test_nn_space_keys(self):
        required_keys = [
            "data_transform",
            "data_alr_denom_idx",
            "n_layers",
            "learning_rate",
            "batch_size",
        ]
        for key in required_keys:
            assert key in ss.nn_space, f"Key '{key}' not found in nn_space"
        for i in range(9):
            assert (
                f"n_units_l{i}" in ss.nn_space
            ), f"Key 'n_units_l{i}' not found in nn_space"

    def test_xgb_space_keys(self):
        required_keys = [
            "data_transform",
            "data_alr_denom_idx",
            "objective",
            "max_depth",
            "min_child_weight",
            "subsample",
            "eta",
        ]
        for key in required_keys:
            assert key in ss.xgb_space, f"Key '{key}' not found in xgb_space"
