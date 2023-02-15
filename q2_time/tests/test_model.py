import pandas as pd
from pandas.testing import assert_frame_equal
from qiime2.plugin.testing import TestPluginBase

from q2_time.model import fit_model, split_data_by_host


class TestSplit(TestPluginBase):
    package = "q2_time.test"

    def test_split_data_by_host(self):
        data = pd.DataFrame({"id": ["a", "b", "c", "c"], "supertarget": [1, 2, 1, 2]})

        train_obs, test_obs = split_data_by_host(data, "id", 0.5)

        train_exp = data.iloc[2:, :].copy()
        test_exp = data.iloc[:2, :].copy()

        assert_frame_equal(train_obs, train_exp)
        assert_frame_equal(test_obs, test_exp)

        overlap = [x for x in train_obs["id"].unique() if x in test_obs["id"].unique()]
        assert len(overlap) == 0

    def test_split_data_by_host_error_one_host(self):
        data = pd.DataFrame({"id": ["c", "c", "c", "c"], "supertarget": [1, 2, 1, 2]})

        with self.assertRaisesRegex(
            ValueError, "Only one unique host available in dataset."
        ):
            split_data_by_host(data, "id", 0.5)


class TestFitModel(TestPluginBase):
    package = "q2_time.test"

    def test_fit_model(self):
        # todo: add all other models as well with decorator
        train = pd.DataFrame(
            {
                "id": ["a", "b", "c", "c"],
                "F": [0.12, 0.23, 0.33, 0.44],
                "supertarget": [1, 2, 5, 7],
            }
        )

        model_type = "LinReg"
        trained_model = fit_model(train, "supertarget", ["F"], model_type)
        self.assertEqual(type(trained_model).__name__, "LinearRegression")


class TestSavePredictions(TestPluginBase):
    package = "q2_time.test"

    def test_save_predictions(self):
        # todo: add proper test
        self.assertEqual(1, 1)
