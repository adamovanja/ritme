"""Tests for classification task support."""

import tempfile
import unittest
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ray.tune import ResultGrid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from ritme.evaluate_models import TunedModel
from ritme.evaluate_tuned_models import (
    _calculate_classification_metrics,
    _plot_confusion_matrices,
    evaluate_tuned_models,
)
from ritme.model_space import static_trainables as st
from ritme.tune_models import CLASSIFICATION_MODELS, REGRESSION_MODELS, TASK_METRICS


class TestTaskTypeConstants(unittest.TestCase):
    def test_regression_models_defined(self):
        self.assertEqual(REGRESSION_MODELS, {"xgb", "nn_reg", "linreg", "rf", "trac"})

    def test_classification_models_defined(self):
        self.assertEqual(
            CLASSIFICATION_MODELS,
            {"xgb_class", "nn_class", "nn_corn", "logreg", "rf_class"},
        )

    def test_no_overlap_between_task_models(self):
        self.assertEqual(len(REGRESSION_MODELS & CLASSIFICATION_MODELS), 0)

    def test_nn_class_nn_corn_allowed_for_both_task_types(self):
        # nn_class and nn_corn should be in CLASSIFICATION_MODELS
        self.assertIn("nn_class", CLASSIFICATION_MODELS)
        self.assertIn("nn_corn", CLASSIFICATION_MODELS)
        # but also allowed for regression (validated in run_all_trials)

    def test_task_metrics(self):
        self.assertEqual(TASK_METRICS["regression"], ("rmse_val", "min"))
        self.assertEqual(TASK_METRICS["classification"], ("accuracy_val", "max"))


class TestClassificationHelpers(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.y = np.array([0, 1, 0, 1])
        self.model = Pipeline([("logreg", LogisticRegression(random_state=0))]).fit(
            self.X, self.y
        )

    def test_predict_accuracy_f1(self):
        acc, f1 = st._predict_accuracy_f1(self.model, self.X, self.y)
        y_pred = self.model.predict(self.X)
        self.assertAlmostEqual(acc, accuracy_score(self.y, y_pred))
        self.assertAlmostEqual(f1, f1_score(self.y, y_pred, average="weighted"))

    @patch("ray.tune.report")
    @patch("ray.tune.get_context")
    def test_report_classification_results_manually(
        self, mock_get_context, mock_report
    ):
        mock_trial_context = MagicMock()
        mock_trial_context.get_trial_id.return_value = "mock_trial_id"
        mock_get_context.return_value = mock_trial_context
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_trial_context.get_trial_dir.return_value = tmpdir
            tax = pd.DataFrame()

            st._report_classification_results_manually(
                self.model, self.X, self.y, self.X, self.y, tax
            )
            mock_report.assert_called_once()
            reported_metrics = mock_report.call_args[1]["metrics"]
            self.assertIn("accuracy_val", reported_metrics)
            self.assertIn("f1_weighted_val", reported_metrics)
            self.assertIn("accuracy_train", reported_metrics)
            self.assertIn("f1_weighted_train", reported_metrics)
            self.assertIn("model_path", reported_metrics)
            self.assertIn("nb_features", reported_metrics)


class TestClassificationTrainables(unittest.TestCase):
    def setUp(self):
        self.train_val = pd.DataFrame(
            {
                "host_id": [1, 2, 3],
                "target": [0, 1, 2],
                "F1": [0.1, 0.9, 0.0],
                "F2": [0.9, 0.1, 1.0],
            },
            index=["ERR1", "ERR2", "ERR3"],
        )
        self.target = "target"
        self.host_id = "host_id"
        self.seed_data = 0
        self.seed_model = 0
        self.tax = pd.DataFrame([])

    @patch("ritme.model_space.static_trainables.process_train")
    @patch("ritme.model_space.static_trainables.StandardScaler.fit_transform")
    @patch("ritme.model_space.static_trainables.LogisticRegression")
    @patch(
        "ritme.model_space.static_trainables._report_classification_results_manually"
    )
    def test_train_logreg(
        self,
        mock_report,
        mock_logreg,
        mock_scaler_trf,
        mock_process_train,
    ):
        config = {"C": 1.0, "penalty": "l2"}
        mock_process_train.return_value = (
            np.array([[1, 2], [3, 4]]),
            np.array([0, 1]),
            np.array([[5, 6]]),
            np.array([0]),
        )

        st.train_logreg(
            config,
            self.train_val,
            self.target,
            self.host_id,
            None,
            self.seed_data,
            self.seed_model,
            self.tax,
        )

        mock_process_train.assert_called_once()
        mock_logreg.assert_called_once_with(
            C=1.0,
            penalty="l2",
            l1_ratio=None,
            solver="saga",
            max_iter=2000,
            random_state=self.seed_model,
        )
        mock_report.assert_called_once()

    @patch("ritme.model_space.static_trainables.process_train")
    @patch("ritme.model_space.static_trainables.RandomForestClassifier")
    @patch(
        "ritme.model_space.static_trainables._report_classification_results_manually"
    )
    def test_train_rf_class(
        self,
        mock_report,
        mock_rf_class,
        mock_process_train,
    ):
        config = {
            "n_estimators": 10,
            "max_depth": 3,
            "min_samples_split": 0.1,
            "min_weight_fraction_leaf": 0.0,
            "min_samples_leaf": 0.01,
            "max_features": "sqrt",
            "min_impurity_decrease": 0.0,
            "bootstrap": True,
        }
        mock_process_train.return_value = (
            np.array([[1, 2], [3, 4]]),
            np.array([0, 1]),
            np.array([[5, 6]]),
            np.array([0]),
        )

        st.train_rf_class(
            config,
            self.train_val,
            self.target,
            self.host_id,
            None,
            self.seed_data,
            self.seed_model,
            self.tax,
        )

        mock_process_train.assert_called_once()
        mock_rf_class.assert_called_once()
        mock_report.assert_called_once()

    @patch("ritme.model_space.static_trainables._save_taxonomy")
    @patch("ritme.model_space.static_trainables.process_train")
    @patch("ritme.model_space.static_trainables.xgb.DMatrix")
    @patch("ritme.model_space.static_trainables.xgb.train")
    @patch("ritme.model_space.static_trainables._RitmeXGBCheckpointCallback")
    @patch("ray.tune.get_context")
    def test_train_xgb_class(
        self,
        mock_get_context,
        mock_checkpoint,
        mock_xgb_train,
        mock_dmatrix,
        mock_process_train,
        mock_save_taxonomy,
    ):
        config = {"n_estimators": 100}
        mock_process_train.return_value = (
            np.array([[1, 2], [3, 4]]),
            np.array([0, 1]),
            np.array([[5, 6]]),
            np.array([0]),
        )
        mock_context = mock_get_context.return_value
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_context.get_trial_dir.return_value = tmpdir

            st.train_xgb_class(
                config,
                self.train_val,
                self.target,
                self.host_id,
                None,
                self.seed_data,
                self.seed_model,
                self.tax,
            )

            mock_xgb_train.assert_called_once()
            # Verify classification-specific config was set
            self.assertEqual(config["objective"], "multi:softmax")
            self.assertIn("num_class", config)


class TestNeuralNetClassificationMetrics(unittest.TestCase):
    def test_calculate_metrics_regression(self):
        model = st.NeuralNet(
            n_units=[2, 4, 1],
            learning_rate=0.01,
            nn_type="regression",
        )
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.1, 2.1, 3.1])
        metrics = model._calculate_metrics(preds, targets)
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)
        self.assertNotIn("accuracy", metrics)

    def test_calculate_metrics_classification(self):
        model = st.NeuralNet(
            n_units=[2, 4, 3],
            learning_rate=0.01,
            nn_type="classification",
            classes=[0, 1, 2],
            task_type="classification",
        )
        # 3-class logits
        preds = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        targets = torch.tensor([0.0, 1.0, 2.0])
        metrics = model._calculate_metrics(preds, targets)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_weighted", metrics)
        self.assertNotIn("rmse", metrics)
        self.assertAlmostEqual(metrics["accuracy"].item(), 1.0)


class TestClassificationEvaluation(unittest.TestCase):
    def setUp(self):
        self.model_type = "logreg"
        self.all_preds = pd.DataFrame(
            {
                "true": [0, 1, 2, 0, 1],
                "pred": [0, 1, 2, 0, 2],
                "split": ["train", "train", "train", "test", "test"],
            }
        )

    def test_calculate_classification_metrics(self):
        metrics = _calculate_classification_metrics(self.all_preds, self.model_type)

        self.assertIn("accuracy_train", metrics.columns)
        self.assertIn("balanced_accuracy_train", metrics.columns)
        self.assertIn("f1_weighted_train", metrics.columns)
        self.assertIn("cohen_kappa_train", metrics.columns)
        self.assertIn("accuracy_test", metrics.columns)

        # Train set: all correct
        self.assertAlmostEqual(metrics.loc[self.model_type, "accuracy_train"], 1.0)
        # Test set: 1 out of 2 correct
        self.assertAlmostEqual(metrics.loc[self.model_type, "accuracy_test"], 0.5)

    def test_plot_confusion_matrices(self):
        metrics = _calculate_classification_metrics(self.all_preds, self.model_type)
        _, axs = plt.subplots(1, 2)
        _plot_confusion_matrices(
            all_preds=self.all_preds,
            metrics_split=metrics,
            axs=axs,
            row_idx=0,
            model_name="test_model",
            only_one_model=True,
        )
        # Check that axes have content
        for ax in axs:
            self.assertTrue(len(ax.images) > 0)
        plt.close("all")

    @patch("ritme.evaluate_tuned_models._predict_w_tuned_model")
    def test_evaluate_tuned_models_classification(self, mock_predict):
        mock_predict.return_value = self.all_preds
        mock_tuned_model = MagicMock(spec=TunedModel)

        exp_config = {"target": "target", "task_type": "classification"}
        train_val = pd.DataFrame({"F1": [1, 2, 3], "target": [0, 1, 2]})
        test = pd.DataFrame({"F1": [4, 5], "target": [0, 1]})

        dic_tuned_models = {"logreg": mock_tuned_model}
        metrics, fig = evaluate_tuned_models(
            dic_tuned_models, exp_config, train_val, test
        )

        self.assertIn("accuracy_train", metrics.columns)
        self.assertIn("f1_weighted_test", metrics.columns)
        self.assertNotIn("rmse_train", metrics.columns)
        plt.close(fig)


class TestRunAllTrialsTaskTypeValidation(unittest.TestCase):
    @patch("ritme.tune_models.run_trials")
    def test_invalid_task_type(self, mock_run_trials):
        from ritme.tune_models import run_all_trials

        with self.assertRaisesRegex(ValueError, "Invalid task_type"):
            run_all_trials(
                train_val=pd.DataFrame(),
                target="t",
                host_id="h",
                stratify_by=None,
                seed_data=0,
                seed_model=0,
                tax=None,
                tree_phylo=None,
                mlflow_uri="mlruns",
                path_exp="/tmp/exp",
                time_budget_s=10,
                max_concurrent_trials=1,
                experiment_tag="test_experiment",
                model_types=["xgb"],
                task_type="invalid",
            )

    @patch("ritme.tune_models.run_trials")
    def test_incompatible_model_types(self, mock_run_trials):
        from ritme.tune_models import run_all_trials

        with self.assertRaisesRegex(ValueError, "not compatible with task_type"):
            run_all_trials(
                train_val=pd.DataFrame(),
                target="t",
                host_id="h",
                stratify_by=None,
                seed_data=0,
                seed_model=0,
                tax=None,
                tree_phylo=None,
                mlflow_uri="mlruns",
                path_exp="/tmp/exp",
                time_budget_s=10,
                max_concurrent_trials=1,
                experiment_tag="test_experiment",
                model_types=["xgb"],
                task_type="classification",
            )

    @patch("ritme.tune_models.run_trials")
    def test_nn_class_nn_corn_allowed_for_regression(self, mock_run_trials):
        from ritme.tune_models import run_all_trials

        mock_run_trials.return_value = MagicMock(spec=ResultGrid)
        train_val = pd.DataFrame({"F1": [0.1, 0.2], "t": [1.0, 2.0], "h": ["a", "b"]})

        # Should NOT raise ValueError for nn_class/nn_corn with regression
        results = run_all_trials(
            train_val=train_val,
            target="t",
            host_id="h",
            stratify_by=None,
            seed_data=0,
            seed_model=0,
            tax=None,
            tree_phylo=None,
            mlflow_uri="mlruns",
            path_exp="/tmp/exp",
            time_budget_s=10,
            max_concurrent_trials=1,
            experiment_tag="test_experiment",
            model_types=["nn_class", "nn_corn"],
            task_type="regression",
        )
        self.assertIn("nn_class", results)
        self.assertIn("nn_corn", results)


class TestXgbClassMetric(unittest.TestCase):
    def test_custom_xgb_class_metric(self):
        import xgboost as xgb

        predt = np.array([0, 1, 2, 0])
        dtrain = xgb.DMatrix(np.array([[1], [2], [3], [4]]))
        dtrain.set_label(np.array([0, 1, 2, 1]))

        result = st.custom_xgb_class_metric(predt, dtrain)
        metric_names = [r[0] for r in result]
        self.assertIn("accuracy", metric_names)
        self.assertIn("f1_weighted", metric_names)


class TestTunedModelClassification(unittest.TestCase):
    def test_model_type_stored(self):
        tmodel = TunedModel(
            model=MagicMock(),
            data_config={},
            tax=pd.DataFrame(),
            path="/tmp",
            model_type="logreg",
        )
        self.assertEqual(tmodel.model_type, "logreg")
        self.assertIsNone(tmodel.label_encoder)

    def test_model_type_default_none(self):
        tmodel = TunedModel(
            model=MagicMock(),
            data_config={},
            tax=pd.DataFrame(),
            path="/tmp",
        )
        self.assertIsNone(tmodel.model_type)


class TestSearchSpaceClassification(unittest.TestCase):
    def test_logreg_space_registered(self):
        from ritme.model_space.static_searchspace import get_search_space

        trial = MagicMock()
        trial.suggest_categorical.return_value = None
        trial.suggest_float.return_value = 1.0

        result = get_search_space(
            trial,
            model_type="logreg",
            tax=None,
            train_val=pd.DataFrame({"F1": [0.5]}),
        )
        self.assertEqual(result["model"], "logreg")

    def test_rf_class_space_registered(self):
        from ritme.model_space.static_searchspace import get_search_space

        trial = MagicMock()
        trial.suggest_categorical.return_value = None
        trial.suggest_float.return_value = 0.5
        trial.suggest_int.return_value = 100

        result = get_search_space(
            trial,
            model_type="rf_class",
            tax=None,
            train_val=pd.DataFrame({"F1": [0.5]}),
        )
        self.assertEqual(result["model"], "rf")

    def test_xgb_class_space_registered(self):
        from ritme.model_space.static_searchspace import get_search_space

        trial = MagicMock()
        trial.suggest_categorical.return_value = None
        trial.suggest_float.return_value = 0.5
        trial.suggest_int.return_value = 100

        result = get_search_space(
            trial,
            model_type="xgb_class",
            tax=None,
            train_val=pd.DataFrame({"F1": [0.5]}),
        )
        self.assertEqual(result["model"], "xgb")


if __name__ == "__main__":
    unittest.main()
