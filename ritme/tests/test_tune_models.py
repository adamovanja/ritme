# test_script.py

import inspect
import os
import unittest
import warnings
from functools import partial
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import skbio
from parameterized import parameterized
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune import ResultGrid
from ray.tune.schedulers import AsyncHyperBandScheduler, HyperBandScheduler
from ray.tune.search.optuna import OptunaSearch

from ritme.model_space import static_searchspace as ss
from ritme.tune_models import (
    DEFAULT_MAX_TRIAL_FAILURE_RATE,
    DEFAULT_SCHEDULER_GRACE_PERIOD,
    DEFAULT_SCHEDULER_MAX_T,
    EARLY_ABORT_MIN_COMPLETED,
    EARLY_ABORT_MIN_ERRORS,
    KFOLD_SCHEDULER_GRACE_PERIOD,
    MODEL_TRAINABLES,
    NAN_TOLERANT_MODELS,
    OPTUNA_SAMPLER_CLASSES,
    _adaptive_n_startup_trials,
    _check_for_errors_in_trials,
    _define_callbacks,
    _define_scheduler,
    _define_search_algo,
    _get_resources,
    _get_slurm_resource,
    _load_wandb_api_key,
    _load_wandb_entity,
    _max_reportable_iterations,
    _max_usable_cpus_per_trial,
    _model_type_for,
    _RecordingTrial,
    _resolve_max_pending_trials,
    _resolve_scheduler_rungs,
    _SafeMLflowLoggerCallback,
    _SearchHealthGuard,
    _validate_budget_inputs,
    _validate_run_inputs,
    _warn_unreachable_scheduler_rungs,
    run_all_trials,
    run_trials,
)


class TestHelpersTuneModels(unittest.TestCase):
    @patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "4"})
    def test_get_slurm_resource_present(self):
        # Test when the environment variable is present
        self.assertEqual(_get_slurm_resource("SLURM_CPUS_PER_TASK"), 4)

    @patch.dict(os.environ, {}, clear=True)
    def test_get_slurm_resource_absent(self):
        # Test when the environment variable is absent
        self.assertEqual(_get_slurm_resource("SLURM_CPUS_PER_TASK"), 0)

    @patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "invalid"})
    def test_get_slurm_resource_invalid(self):
        # Test when the environment variable is invalid
        self.assertEqual(_get_slurm_resource("SLURM_CPUS_PER_TASK"), 0)

    def test_nan_tolerant_models_constant(self):
        # Guard against silent shrinking of the NaN-tolerant whitelist.
        self.assertEqual(
            NAN_TOLERANT_MODELS,
            frozenset({"xgb", "xgb_class", "rf", "rf_class"}),
        )

    def test_check_for_errors_in_trials_no_errors(self):
        mock_result = MagicMock(spec=ResultGrid)
        mock_result.__len__.return_value = 10
        mock_result.num_errors = 0
        _check_for_errors_in_trials(mock_result)

    def test_check_for_errors_in_trials_with_errors(self):
        mock_result = MagicMock(spec=ResultGrid)
        mock_result.__len__.return_value = 10
        mock_result.num_errors = 10
        with self.assertRaises(RuntimeError):
            _check_for_errors_in_trials(mock_result)

    def _mock_result_grid(self, num_trials, num_errors, error_types=()):
        mock_result = MagicMock(spec=ResultGrid)
        mock_result.__len__.return_value = num_trials
        mock_result.num_errors = num_errors
        mock_result.errors = [t() for t in error_types]
        return mock_result

    def test_check_for_errors_in_trials_below_threshold_warns(self):
        mock_result = self._mock_result_grid(1000, 1, [TimeoutError])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            breakdown = _check_for_errors_in_trials(mock_result)
        self.assertEqual(breakdown["num_trials"], 1000)
        self.assertEqual(breakdown["num_errors"], 1)
        self.assertAlmostEqual(breakdown["failure_rate"], 0.001)
        self.assertEqual(breakdown["error_classes"], ["TimeoutError"])
        msgs = [
            str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)
        ]
        # The warning must name the counts AND the error class so ops triage
        # has actionable signal even after the breakdown dict is dropped.
        self.assertTrue(any("1/1000" in m and "TimeoutError" in m for m in msgs))

    def test_check_for_errors_in_trials_above_threshold_raises(self):
        mock_result = self._mock_result_grid(100, 1, [ValueError])
        with self.assertRaises(RuntimeError) as ctx:
            _check_for_errors_in_trials(mock_result)
        msg = str(ctx.exception)
        self.assertIn("0.0100", msg)
        self.assertIn("1/100", msg)
        self.assertIn("ValueError", msg)

    def test_check_for_errors_in_trials_at_threshold_does_not_raise(self):
        # Boundary: exactly at threshold must warn, not raise. Pins the
        # spec's "> threshold raises, <= threshold warns" inequality.
        mock_result = self._mock_result_grid(200, 1, [ValueError])
        breakdown = _check_for_errors_in_trials(
            mock_result, max_trial_failure_rate=0.005
        )
        self.assertAlmostEqual(breakdown["failure_rate"], 0.005)

    def test_check_for_errors_in_trials_respects_custom_threshold(self):
        mock_result = self._mock_result_grid(100, 1, [ValueError])
        breakdown = _check_for_errors_in_trials(
            mock_result, max_trial_failure_rate=0.02
        )
        self.assertAlmostEqual(breakdown["failure_rate"], 0.01)

    def test_check_for_errors_in_trials_collects_unique_error_classes(self):
        mock_result = self._mock_result_grid(
            1000, 3, [TimeoutError, ValueError, TimeoutError]
        )
        breakdown = _check_for_errors_in_trials(mock_result)
        self.assertEqual(breakdown["error_classes"], ["TimeoutError", "ValueError"])

    def test_check_for_errors_in_trials_handles_errors_none(self):
        # ResultGrid.errors can be None on some Ray versions; the
        # ``or []`` guard must not raise on iteration.
        mock_result = self._mock_result_grid(1000, 0)
        mock_result.errors = None
        breakdown = _check_for_errors_in_trials(mock_result)
        self.assertEqual(breakdown["error_classes"], [])

    def test_check_for_errors_in_trials_zero_trials_raises(self):
        # Ray Tune yielding 0 trials = the campaign produced nothing;
        # always raise so a misconfigured time_budget / search space is
        # not silently swallowed.
        mock_result = MagicMock(spec=ResultGrid)
        mock_result.__len__.return_value = 0
        mock_result.num_errors = 0
        mock_result.errors = []
        with self.assertRaisesRegex(RuntimeError, "produced 0 trials"):
            _check_for_errors_in_trials(mock_result)

    def test_check_for_errors_in_trials_default_uses_module_constant(self):
        # Sanity check: the function's default value tracks the module
        # constant (single source of truth).
        default = _check_for_errors_in_trials.__defaults__[-1]
        self.assertEqual(default, DEFAULT_MAX_TRIAL_FAILURE_RATE)

    def _validator_train_val(self, target_values=None, with_snapshots=False):
        cols = {
            "F0": [0.1, 0.2, 0.3],
            "F1": [0.4, 0.5, 0.6],
            "target": target_values or [1.0, 2.0, 3.0],
        }
        if with_snapshots:
            cols["F0__t-1"] = [np.nan, 0.15, 0.25]
        return pd.DataFrame(cols)

    def test_validate_run_inputs_invalid_task_type(self):
        with self.assertRaisesRegex(ValueError, "Invalid task_type"):
            _validate_run_inputs(
                model_types=["linreg"],
                task_type="not_a_task",
                target="target",
                train_val=self._validator_train_val(),
            )

    def test_validate_run_inputs_nn_corn_non_numeric_target(self):
        train_val = self._validator_train_val(target_values=["a", "b", "c"])
        with self.assertRaisesRegex(ValueError, "nn_corn requires a numeric target"):
            _validate_run_inputs(
                model_types=["nn_corn"],
                task_type="regression",
                target="target",
                train_val=train_val,
            )

    def test_validate_run_inputs_nn_corn_nan_target(self):
        train_val = self._validator_train_val(target_values=[1.0, np.nan, 3.0])
        with self.assertRaisesRegex(ValueError, "contains 1 NaN"):
            _validate_run_inputs(
                model_types=["nn_corn"],
                task_type="regression",
                target="target",
                train_val=train_val,
            )

    def test_validate_run_inputs_snapshot_nan_gate(self):
        train_val = self._validator_train_val(with_snapshots=True)
        with self.assertRaisesRegex(ValueError, "NaNs in snapshot features"):
            _validate_run_inputs(
                model_types=["linreg"],
                task_type="regression",
                target="target",
                train_val=train_val,
            )

    def test_validate_run_inputs_passes_for_valid_inputs(self):
        _validate_run_inputs(
            model_types=["linreg", "rf"],
            task_type="regression",
            target="target",
            train_val=self._validator_train_val(),
        )

    @patch("ritme.tune_models._get_slurm_resource")
    def test_get_resources(self, mock_get_slurm_resource):
        # Mock get_slurm_resource to return predefined values
        mock_get_slurm_resource.side_effect = [8, 2]
        resources = _get_resources(2)

        expected_resources = {"cpu": 4, "gpu": 1}
        self.assertEqual(resources, expected_resources)

        # Check that get_slurm_resource was called with correct arguments
        mock_get_slurm_resource.assert_any_call("SLURM_CPUS_PER_TASK", 1)
        mock_get_slurm_resource.assert_any_call("SLURM_GPUS_PER_TASK", 0)

    def test_resolve_scheduler_rungs_kfold_defaults(self):
        # K-fold trials emit at most k_folds reports, so the derived rungs
        # must be reachable: first pruning decision at the fold-2 running
        # mean, max_t at the fold count.
        self.assertEqual(
            _resolve_scheduler_rungs(None, None, 5),
            (KFOLD_SCHEDULER_GRACE_PERIOD, 5),
        )

    def test_resolve_scheduler_rungs_single_split_defaults(self):
        self.assertEqual(
            _resolve_scheduler_rungs(None, None, 1),
            (DEFAULT_SCHEDULER_GRACE_PERIOD, DEFAULT_SCHEDULER_MAX_T),
        )

    def test_resolve_scheduler_rungs_reproducible_keeps_inert_defaults(self):
        # HyperBand (fully_reproducible) PAUSES at reachable milestones and
        # K-fold reports carry no checkpoint, so a resumed trial restarts
        # from fold 0 -- keep the unreachable defaults there.
        self.assertEqual(
            _resolve_scheduler_rungs(None, None, 5, fully_reproducible=True),
            (DEFAULT_SCHEDULER_GRACE_PERIOD, DEFAULT_SCHEDULER_MAX_T),
        )
        # Explicit values still win for users who accept the pause cost.
        self.assertEqual(
            _resolve_scheduler_rungs(1, 5, 5, fully_reproducible=True), (1, 5)
        )

    def test_resolve_scheduler_rungs_explicit_override(self):
        self.assertEqual(_resolve_scheduler_rungs(3, 7, 5), (3, 7))

    def test_resolve_scheduler_rungs_partial_override(self):
        self.assertEqual(
            _resolve_scheduler_rungs(None, 50, 5),
            (KFOLD_SCHEDULER_GRACE_PERIOD, 50),
        )
        self.assertEqual(_resolve_scheduler_rungs(3, None, 5), (3, 5))

    def test_max_reportable_iterations(self):
        # K-fold: every family reports once per fold.
        self.assertEqual(_max_reportable_iterations("linreg", 5), 5)
        self.assertEqual(_max_reportable_iterations("xgb", 5), 5)
        # Single-split: the manual-report families report exactly once,
        # xgb / nn report per boosting iteration / epoch.
        self.assertEqual(_max_reportable_iterations("linreg", 1), 1)
        self.assertEqual(_max_reportable_iterations("trac", 1), 1)
        self.assertEqual(_max_reportable_iterations("rf", 1), 1)
        self.assertEqual(_max_reportable_iterations("rf_class", 1), 1)
        self.assertIsNone(_max_reportable_iterations("xgb", 1))
        self.assertIsNone(_max_reportable_iterations("nn_reg", 1))

    def test_max_usable_cpus_per_trial(self):
        # Fold-parallel families: one CPU per fold.
        self.assertEqual(_max_usable_cpus_per_trial("linreg", 5), 5)
        self.assertEqual(_max_usable_cpus_per_trial("logreg", 1), 1)
        self.assertEqual(_max_usable_cpus_per_trial("trac", 5), 5)
        # Threaded families: no family cap.
        self.assertIsNone(_max_usable_cpus_per_trial("xgb", 5))
        self.assertIsNone(_max_usable_cpus_per_trial("rf", 5))
        self.assertIsNone(_max_usable_cpus_per_trial("nn_reg", 5))

    @patch("ritme.tune_models._get_slurm_resource")
    def test_get_resources_caps_fold_parallel_families(self, mock_slurm):
        mock_slurm.side_effect = [60, 0]
        # Allocation share 60 // 10 = 6, capped by k_folds = 5.
        self.assertEqual(_get_resources(10, "linreg", 5)["cpu"], 5)

    @patch("ritme.tune_models._get_slurm_resource")
    def test_get_resources_no_cap_for_threaded_families(self, mock_slurm):
        mock_slurm.side_effect = [60, 0]
        self.assertEqual(_get_resources(10, "xgb", 5)["cpu"], 6)

    def test_validate_budget_inputs(self):
        _validate_budget_inputs(None, None)
        _validate_budget_inputs(600.0, 8)
        with self.assertRaisesRegex(ValueError, "max_trial_duration_s"):
            _validate_budget_inputs(0, None)
        with self.assertRaisesRegex(ValueError, "max_trial_duration_s"):
            _validate_budget_inputs(-5, None)
        with self.assertRaisesRegex(ValueError, "max_pending_trials"):
            _validate_budget_inputs(None, 0)

    def test_model_type_for_resolves_trainable_over_exp_name(self):
        self.assertEqual(
            _model_type_for(MODEL_TRAINABLES["linreg"], "my_linreg_sweep"),
            "linreg",
        )
        # Unknown trainable (e.g. a mock): fall back to the experiment name.
        self.assertEqual(_model_type_for(Mock(), "xgb"), "xgb")

    def test_resolve_max_pending_trials_default_matches_concurrency(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TUNE_MAX_PENDING_TRIALS_PG", None)
            self.assertEqual(_resolve_max_pending_trials(10), 10)
            # Floor of 2 keeps at least one launch overlapping.
            self.assertEqual(_resolve_max_pending_trials(1), 2)

    def test_resolve_max_pending_trials_explicit_wins(self):
        with patch.dict(
            os.environ, {"TUNE_MAX_PENDING_TRIALS_PG": "7"}, clear=False
        ), patch("ritme.tune_models._SELF_SET_MAX_PENDING", None):
            self.assertEqual(_resolve_max_pending_trials(10, requested=4), 4)

    def test_resolve_max_pending_trials_external_env_respected(self):
        # A user-set env value wins even when exported after import.
        with patch.dict(
            os.environ, {"TUNE_MAX_PENDING_TRIALS_PG": "7"}, clear=False
        ), patch("ritme.tune_models._SELF_SET_MAX_PENDING", None):
            self.assertEqual(_resolve_max_pending_trials(10), 7)

    def test_resolve_max_pending_trials_ignores_own_earlier_write(self):
        # A value ritme itself wrote on a previous run_trials call must not
        # masquerade as a user override for the next model's sweep.
        with patch.dict(
            os.environ, {"TUNE_MAX_PENDING_TRIALS_PG": "12"}, clear=False
        ), patch("ritme.tune_models._SELF_SET_MAX_PENDING", "12"):
            self.assertEqual(_resolve_max_pending_trials(5), 5)

    @patch("builtins.print")
    def test_warn_unreachable_scheduler_rungs(self, mock_print):
        msg = _warn_unreachable_scheduler_rungs("linreg", 1, 10)
        self.assertIn("cannot prune", msg)
        mock_print.assert_called_once()
        self.assertIsNone(_warn_unreachable_scheduler_rungs("linreg", 5, 1))
        self.assertIsNone(_warn_unreachable_scheduler_rungs("xgb", 1, 10))

    def test_define_scheduler_not_fully_reproducible(self):
        scheduler_max_t = 100

        scheduler = _define_scheduler(False, 10, scheduler_max_t, "rmse_val", "min", 1)

        self.assertIsInstance(scheduler, AsyncHyperBandScheduler)
        self.assertEqual(scheduler._max_t, scheduler_max_t)

    def test_define_scheduler_fully_reproducible(self):
        scheduler_max_t = 100

        scheduler = _define_scheduler(True, 10, scheduler_max_t, "rmse_val", "min", 1)

        self.assertIsInstance(scheduler, HyperBandScheduler)
        self.assertEqual(scheduler._max_t_attr, scheduler_max_t)

    def test_define_scheduler_uses_mean_suffix_when_kfold(self):
        # K-fold (k_folds > 1) must point the scheduler at ``<metric>_mean``
        # since the running aggregate strips the bare ``<metric>`` key (see
        # issue_eval_class.md).
        scheduler = _define_scheduler(False, 10, 100, "rmse_val", "min", 5)
        self.assertEqual(scheduler._metric, "rmse_val_mean")
        self.assertEqual(scheduler._mode, "min")

    def test_define_scheduler_uses_bare_metric_when_single_split(self):
        # k_folds == 1 keeps the bare ``<metric>`` because the single-split
        # callbacks emit it every epoch.
        scheduler = _define_scheduler(False, 10, 100, "rmse_val", "min", 1)
        self.assertEqual(scheduler._metric, "rmse_val")

    @parameterized.expand(
        [
            "RandomSampler",
            "TPESampler",
            "CmaEsSampler",
            "GPSampler",
            "QMCSampler",
        ]
    )
    def test_define_search_algo(self, sampler):
        mock_func_to_get_search_space = Mock()

        # Use a real model type so the adaptive n_startup_trials path (which
        # introspects ss.get_search_space) does not raise for samplers that
        # need it (TPE / CmaEs / GP). The mocked func_to_get_search_space
        # still controls what OptunaSearch sees as its space.
        exp_name = "linreg"
        tax = pd.DataFrame()
        train_val = pd.DataFrame({"F0": [1.0, 2.0, 3.0], "F1": [0.5, 1.5, 2.5]})
        model_hyperparameters = {}
        seed_model = 42
        metric = "accuracy"
        mode = "max"

        search_algo = _define_search_algo(
            mock_func_to_get_search_space,
            exp_name,
            tax,
            train_val,
            model_hyperparameters,
            sampler,
            seed_model,
            metric,
            mode,
        )

        self.assertIsInstance(search_algo, OptunaSearch)

        self.assertTrue(isinstance(search_algo._space, partial))
        search_algo._space()

        mock_func_to_get_search_space.assert_called_once_with(
            model_type=exp_name,
            tax=tax,
            train_val=train_val,
            model_hyperparameters=model_hyperparameters,
        )

        self.assertEqual(search_algo._metric, metric)
        self.assertEqual(search_algo._mode, mode)
        self.assertTrue(
            isinstance(search_algo._sampler, OPTUNA_SAMPLER_CLASSES[sampler])
        )

    def test_define_search_algo_invalid_sampler(self):
        invalid_sampler = "InvalidSampler"
        with self.assertRaisesRegex(
            ValueError, f"Unrecognized sampler '{invalid_sampler}'."
        ):
            _define_search_algo(
                Mock(),
                "linreg",
                pd.DataFrame(),
                pd.DataFrame({"F0": [1.0, 2.0, 3.0], "F1": [0.5, 1.5, 2.5]}),
                {},
                invalid_sampler,
                42,
                "accuracy",
                "max",
            )

    @patch.dict(os.environ, {"WANDB_API_KEY": "test_api_key"})
    def test_load_wandb_api_key(self):
        api_key = _load_wandb_api_key()
        self.assertEqual(api_key, "test_api_key")

    @patch("ritme.tune_models.os.getenv", return_value=None)
    def test_load_wandb_api_key_missing(self, mock_getenv):
        with self.assertRaisesRegex(ValueError, "No WANDB_API_KEY found in .env file."):
            _load_wandb_api_key()

    @patch.dict(os.environ, {"WANDB_ENTITY": "test_entity"})
    def test_load_wandb_entity(self):
        entity = _load_wandb_entity()
        self.assertEqual(entity, "test_entity")

    @patch("ritme.tune_models.os.getenv", return_value=None)
    def test_load_wandb_entity_missing(self, mock_getenv):
        with self.assertRaisesRegex(ValueError, "No WANDB_ENTITY found in .env file."):
            _load_wandb_entity()

    def test_define_callbacks_mlflow(self):
        tracking_uri = "sqlite:///tmp/mlflow.db"
        exp_name = "test_exp"
        experiment_tag = "test_tag"

        callbacks = _define_callbacks(tracking_uri, exp_name, experiment_tag)

        self.assertEqual(len(callbacks), 1)
        self.assertIsInstance(callbacks[0], _SafeMLflowLoggerCallback)
        self.assertIsInstance(callbacks[0], MLflowLoggerCallback)
        self.assertEqual(callbacks[0].tracking_uri, tracking_uri)
        self.assertEqual(callbacks[0].experiment_name, exp_name)
        self.assertEqual(callbacks[0].tags, {"experiment_tag": experiment_tag})

    def test_safe_mlflow_log_trial_end_skips_unknown_trial(self):
        # Trials whose actor died before log_trial_start should not raise.
        cb = _SafeMLflowLoggerCallback(
            tracking_uri="sqlite:///tmp/mlflow.db",
            experiment_name="test_exp",
            tags={"experiment_tag": "test_tag"},
        )
        cb.mlflow_util = MagicMock()
        cb._trial_runs = {}
        unknown_trial = MagicMock()

        cb.log_trial_end(unknown_trial, failed=True)

        cb.mlflow_util.end_run.assert_not_called()

    def test_safe_mlflow_log_trial_end_delegates_known_trial(self):
        # Trials that did start must still be finalized via the parent impl.
        cb = _SafeMLflowLoggerCallback(
            tracking_uri="sqlite:///tmp/mlflow.db",
            experiment_name="test_exp",
            tags={"experiment_tag": "test_tag"},
        )
        known_trial = MagicMock()
        cb._trial_runs = {known_trial: "run-123"}

        with patch.object(MLflowLoggerCallback, "log_trial_end") as mock_super_end:
            cb.log_trial_end(known_trial, failed=True)

        mock_super_end.assert_called_once_with(known_trial, failed=True)

    @patch("ritme.tune_models._load_wandb_api_key")
    @patch("ritme.tune_models._load_wandb_entity")
    def test_define_callbacks_wandb(self, mock_load_entity, mock_load_api_key):
        mock_load_api_key.return_value = "test_api_key"
        mock_load_entity.return_value = "test_entity"
        tracking_uri = "wandb"
        exp_name = "test_exp"
        experiment_tag = "test_tag"

        callbacks = _define_callbacks(tracking_uri, exp_name, experiment_tag)

        self.assertEqual(len(callbacks), 1)
        self.assertIsInstance(callbacks[0], WandbLoggerCallback)
        self.assertEqual(callbacks[0].api_key, "test_api_key")
        self.assertEqual(callbacks[0].kwargs["entity"], "test_entity")
        self.assertEqual(callbacks[0].project, experiment_tag)
        self.assertEqual(callbacks[0].kwargs["tags"], {experiment_tag})
        mock_load_api_key.assert_called_once()
        mock_load_entity.assert_called_once()

    @patch("ritme.tune_models.print")
    def test_define_callbacks_invalid_uri(self, mock_print):
        callbacks = _define_callbacks("invalid_uri", "test_exp", "test_tag")

        self.assertEqual(len(callbacks), 0)
        mock_print.assert_called_once_with(
            "No valid tracking URI provided. Proceeding without logging callbacks."
        )


class TestSearchHealthGuard(unittest.TestCase):
    def test_healthy_search_never_stops(self):
        guard = _SearchHealthGuard("rmse_val", DEFAULT_MAX_TRIAL_FAILURE_RATE)
        for i in range(100):
            self.assertFalse(guard(f"t{i}", {"rmse_val": 1.0, "time_total_s": 10.0}))
            guard.on_trial_complete(i, [], None)
        # One sporadic flaky error among 100 completions must not abort.
        guard.on_trial_error(100, [], None)
        self.assertFalse(guard.stop_all())
        self.assertFalse(guard.stopped_for_missing_metric)

    def test_all_error_search_stops_at_min_errors(self):
        guard = _SearchHealthGuard("rmse_val", DEFAULT_MAX_TRIAL_FAILURE_RATE)
        for i in range(EARLY_ABORT_MIN_ERRORS - 1):
            guard.on_trial_error(i, [], None)
        self.assertFalse(guard.stop_all())
        guard.on_trial_error(EARLY_ABORT_MIN_ERRORS, [], None)
        self.assertTrue(guard.stop_all())

    def test_early_abort_rate_never_below_user_rate(self):
        # A user tolerating 90% failures must not see an early abort at 50%.
        guard = _SearchHealthGuard("rmse_val", max_trial_failure_rate=0.9)
        for i in range(20):
            guard.on_trial_error(i, [], None)
        for i in range(10):
            guard(f"t{i}", {"rmse_val": 1.0, "time_total_s": 5.0})
            guard.on_trial_complete(i, [], None)
        # 20 / 30 ~ 0.67 < 0.9 -> keep running.
        self.assertFalse(guard.stop_all())
        for i in range(70):
            guard.on_trial_error(i, [], None)
        # 90 / 100 = 0.9 >= 0.9 -> stop.
        self.assertTrue(guard.stop_all())

    def test_nan_metric_search_stops_after_min_completed(self):
        guard = _SearchHealthGuard(
            "roc_auc_macro_ovr_val_mean", DEFAULT_MAX_TRIAL_FAILURE_RATE
        )
        for i in range(EARLY_ABORT_MIN_COMPLETED):
            guard(
                f"t{i}",
                {
                    "roc_auc_macro_ovr_val_mean": float("nan"),
                    "time_total_s": 5.0,
                },
            )
            guard.on_trial_complete(i, [], None)
        self.assertTrue(guard.stop_all())
        self.assertTrue(guard.stopped_for_missing_metric)

    def test_single_finite_metric_prevents_missing_metric_stop(self):
        guard = _SearchHealthGuard("rmse_val", DEFAULT_MAX_TRIAL_FAILURE_RATE)
        guard("t0", {"rmse_val": 0.7, "time_total_s": 5.0})
        for i in range(5 * EARLY_ABORT_MIN_COMPLETED):
            guard(f"t{i}", {"rmse_val": float("nan"), "time_total_s": 5.0})
            guard.on_trial_complete(i, [], None)
        self.assertFalse(guard.stop_all())

    def test_systemic_failure_sets_flag(self):
        guard = _SearchHealthGuard("rmse_val", DEFAULT_MAX_TRIAL_FAILURE_RATE)
        for i in range(EARLY_ABORT_MIN_ERRORS):
            guard.on_trial_error(i, [], None)
        self.assertTrue(guard.stop_all())
        self.assertTrue(guard.stopped_for_systemic_failure)
        self.assertEqual(guard.num_errored, EARLY_ABORT_MIN_ERRORS)
        self.assertEqual(guard.num_finished, EARLY_ABORT_MIN_ERRORS)

    def test_time_cap_not_enforced_when_trainables_self_cap(self):
        # K-fold runs disable the stopper-side cap: the trainables cap
        # themselves at fold boundaries and stamp ``time_capped``; a
        # stopper-side kill on the same reports would race that check.
        guard = _SearchHealthGuard(
            "rmse_val_mean",
            DEFAULT_MAX_TRIAL_FAILURE_RATE,
            max_trial_duration_s=100,
            enforce_time_cap=False,
        )
        self.assertFalse(guard("t0", {"rmse_val_mean": 1.0, "time_total_s": 1e9}))

    def test_time_cap_stops_trial_at_report_boundary(self):
        guard = _SearchHealthGuard(
            "rmse_val", DEFAULT_MAX_TRIAL_FAILURE_RATE, max_trial_duration_s=100
        )
        self.assertFalse(guard("t0", {"rmse_val": 1.0, "time_total_s": 99.0}))
        self.assertTrue(guard("t0", {"rmse_val": 1.0, "time_total_s": 101.0}))

    def test_no_time_cap_never_stops_trials(self):
        guard = _SearchHealthGuard("rmse_val", DEFAULT_MAX_TRIAL_FAILURE_RATE)
        self.assertFalse(guard("t0", {"rmse_val": 1.0, "time_total_s": 1e9}))


class TestMainTuneModels(unittest.TestCase):
    def setUp(self):
        # Common variables for all tests. The minimal train_val needs at
        # least one F-prefixed column so the search-space introspection in
        # _adaptive_n_startup_trials does not trip on the empty .str accessor.
        self.train_val = pd.DataFrame({"F0": [1.0, 2.0, 3.0], "F1": [0.5, 1.5, 2.5]})
        self.target = "target_column"
        self.host_id = "host_id_column"
        self.seed_data = 42
        self.seed_model = 42
        self.tax = pd.DataFrame()
        self.tree_phylo = skbio.TreeNode()
        self.path2exp = "/tmp/experiment"
        self.experiment_tag = "test_experiment_tag"
        self.time_budget_s = 5
        self.max_concurrent_trials = 2
        self.model_hyperparameters = {}
        self.mlflow_uri = "sqlite:///tmp/experiment/mlflow.db"

    @patch("ritme.tune_models.init")
    @patch("ritme.tune_models.ray.cluster_resources")
    @patch("ritme.tune_models.tune.Tuner")
    def test_run_trials_uses_passed_experiment_tag_for_callback(
        self, mock_tuner_class, mock_resources, mock_init
    ):
        # Regression test: experiment_tag must come from the caller, not from
        # os.path.basename(path2exp). path2exp here is a throwaway temp dir,
        # but the MLflow tag should still be the user-supplied experiment_tag.
        mock_context = MagicMock()
        mock_context.dashboard_url = "http://localhost:8265"
        mock_init.return_value = mock_context
        mock_resources.return_value = {}
        mock_tuner = MagicMock()
        _fit_result = MagicMock(spec=ResultGrid, num_errors=0)
        _fit_result.__len__.return_value = 10
        mock_tuner.fit.return_value = _fit_result
        mock_tuner_class.return_value = mock_tuner

        temp_path2exp = "/tmp/tmp_throwaway_abc123"
        user_tag = "my_real_experiment"

        run_trials(
            tracking_uri=self.mlflow_uri,
            exp_name="linreg",
            trainable=MagicMock(),
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            path2exp=temp_path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=user_tag,
            fully_reproducible=False,
            model_hyperparameters=self.model_hyperparameters,
        )

        run_config = mock_tuner_class.call_args.kwargs["run_config"]
        mlflow_cb = run_config.callbacks[0]
        self.assertEqual(mlflow_cb.tags, {"experiment_tag": user_tag})
        self.assertNotEqual(
            mlflow_cb.tags["experiment_tag"], os.path.basename(temp_path2exp)
        )

    @patch("ritme.tune_models.init")
    @patch("ritme.tune_models.ray.cluster_resources")
    @patch("ritme.tune_models.tune.Tuner")
    def test_run_trials_not_reproducible(
        self, mock_tuner_class, mock_resources, mock_init
    ):
        mock_context = MagicMock()
        mock_context.dashboard_url = "http://localhost:8265"
        mock_init.return_value = mock_context

        mock_resources.return_value = {}

        mock_tuner = MagicMock()
        _fit_result = MagicMock(spec=ResultGrid, num_errors=0)
        _fit_result.__len__.return_value = 10
        mock_tuner.fit.return_value = _fit_result
        mock_tuner_class.return_value = mock_tuner

        trainable = MagicMock()

        result = run_trials(
            tracking_uri=self.mlflow_uri,
            exp_name="linreg",
            trainable=trainable,
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            path2exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            fully_reproducible=False,
            model_hyperparameters=self.model_hyperparameters,
        )

        # Assertions
        mock_init.assert_called_once()
        mock_tuner_class.assert_called_once()
        mock_tuner.fit.assert_called_once()
        self.assertIsInstance(result, ResultGrid)

    @patch("ritme.tune_models.init")
    @patch("ritme.tune_models.ray.cluster_resources")
    @patch("ritme.tune_models.tune.Tuner")
    @patch("ritme.tune_models._define_scheduler")
    def test_run_trials_derives_scheduler_rungs_from_k_folds(
        self, mock_scheduler, mock_tuner_class, mock_resources, mock_init
    ):
        mock_context = MagicMock()
        mock_context.dashboard_url = "http://localhost:8265"
        mock_init.return_value = mock_context
        mock_resources.return_value = {}
        mock_tuner = MagicMock()
        _fit_result = MagicMock(spec=ResultGrid, num_errors=0)
        _fit_result.__len__.return_value = 10
        mock_tuner.fit.return_value = _fit_result
        mock_tuner_class.return_value = mock_tuner

        run_trials(
            tracking_uri=self.mlflow_uri,
            exp_name="xgb",
            trainable=MagicMock(),
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            path2exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            k_folds=5,
        )

        sched_args = mock_scheduler.call_args.args
        self.assertEqual(sched_args[1], KFOLD_SCHEDULER_GRACE_PERIOD)
        self.assertEqual(sched_args[2], 5)

    @patch("ritme.tune_models.init")
    @patch("ritme.tune_models.ray.cluster_resources")
    @patch("ritme.tune_models.tune.Tuner")
    def test_run_trials_wires_health_guard_as_stopper_and_callback(
        self, mock_tuner_class, mock_resources, mock_init
    ):
        mock_context = MagicMock()
        mock_context.dashboard_url = "http://localhost:8265"
        mock_init.return_value = mock_context
        mock_resources.return_value = {}
        mock_tuner = MagicMock()
        _fit_result = MagicMock(spec=ResultGrid, num_errors=0)
        _fit_result.__len__.return_value = 10
        mock_tuner.fit.return_value = _fit_result
        mock_tuner_class.return_value = mock_tuner

        run_trials(
            tracking_uri=self.mlflow_uri,
            exp_name="linreg",
            trainable=MagicMock(),
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            path2exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
        )

        run_config = mock_tuner_class.call_args.kwargs["run_config"]
        self.assertIsInstance(run_config.stop, _SearchHealthGuard)
        self.assertIn(run_config.stop, run_config.callbacks)

    @patch("ritme.tune_models.init")
    @patch("ritme.tune_models.ray.cluster_resources")
    @patch("ritme.tune_models.tune.Tuner")
    @patch("ritme.tune_models._SearchHealthGuard")
    def test_run_trials_raises_early_no_best_trial_error(
        self, mock_guard_class, mock_tuner_class, mock_resources, mock_init
    ):
        # When the guard stopped the experiment because no finite metric was
        # ever reported, run_trials must raise the no-best-trial error right
        # after fit() -- not hours later in the retrieve step (or after the
        # remaining model types consumed their budgets too).
        mock_context = MagicMock()
        mock_context.dashboard_url = "http://localhost:8265"
        mock_init.return_value = mock_context
        mock_resources.return_value = {}
        mock_tuner = MagicMock()
        _fit_result = MagicMock(spec=ResultGrid, num_errors=0)
        _fit_result.__len__.return_value = 10
        mock_tuner.fit.return_value = _fit_result
        mock_tuner_class.return_value = mock_tuner
        mock_guard = MagicMock(spec=_SearchHealthGuard)
        mock_guard.stopped_for_missing_metric = True
        mock_guard.stopped_for_systemic_failure = False
        mock_guard_class.return_value = mock_guard

        with self.assertRaisesRegex(
            RuntimeError, "No best trial found for the given metric"
        ):
            run_trials(
                tracking_uri=self.mlflow_uri,
                exp_name="nn_class",
                trainable=MagicMock(),
                train_val=self.train_val,
                target=self.target,
                host_id=self.host_id,
                stratify_by=None,
                seed_data=self.seed_data,
                seed_model=self.seed_model,
                tax=self.tax,
                tree_phylo=self.tree_phylo,
                path2exp=self.path2exp,
                time_budget_s=self.time_budget_s,
                max_concurrent_trials=self.max_concurrent_trials,
                experiment_tag=self.experiment_tag,
                task_type="classification",
            )

    @patch("ritme.tune_models.init")
    @patch("ritme.tune_models.ray.cluster_resources")
    @patch("ritme.tune_models.tune.Tuner")
    @patch("ritme.tune_models._SearchHealthGuard")
    def test_run_trials_raises_unconditionally_on_systemic_failure(
        self, mock_guard_class, mock_tuner_class, mock_resources, mock_init
    ):
        # The post-fit failure rate is diluted by pending/running trials that
        # were terminated (not errored) at the early stop, so run_trials must
        # raise from the guard's flag rather than rely on the policy.
        mock_context = MagicMock()
        mock_context.dashboard_url = "http://localhost:8265"
        mock_init.return_value = mock_context
        mock_resources.return_value = {}
        mock_tuner = MagicMock()
        _fit_result = MagicMock(spec=ResultGrid, num_errors=0)
        _fit_result.__len__.return_value = 100
        mock_tuner.fit.return_value = _fit_result
        mock_tuner_class.return_value = mock_tuner
        mock_guard = MagicMock(spec=_SearchHealthGuard)
        mock_guard.stopped_for_missing_metric = False
        mock_guard.stopped_for_systemic_failure = True
        mock_guard.num_errored = 10
        mock_guard.num_finished = 20
        mock_guard_class.return_value = mock_guard

        with self.assertRaisesRegex(RuntimeError, "systemic failure"):
            run_trials(
                tracking_uri=self.mlflow_uri,
                exp_name="linreg",
                trainable=MagicMock(),
                train_val=self.train_val,
                target=self.target,
                host_id=self.host_id,
                stratify_by=None,
                seed_data=self.seed_data,
                seed_model=self.seed_model,
                tax=self.tax,
                tree_phylo=self.tree_phylo,
                path2exp=self.path2exp,
                time_budget_s=self.time_budget_s,
                max_concurrent_trials=self.max_concurrent_trials,
                experiment_tag=self.experiment_tag,
            )

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_forwards_scheduler_settings(self, mock_run_trials):
        mock_run_trials.return_value = MagicMock(spec=ResultGrid)

        run_all_trials(
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            model_types=["xgb"],
            scheduler_grace_period=2,
            scheduler_max_t=4,
        )

        kwargs = mock_run_trials.call_args.kwargs
        self.assertEqual(kwargs["scheduler_grace_period"], 2)
        self.assertEqual(kwargs["scheduler_max_t"], 4)

    @patch("ritme.tune_models.init")
    @patch("ritme.tune_models.ray.cluster_resources")
    @patch("ritme.tune_models.tune.Tuner")
    @patch("ritme.tune_models.tune.with_parameters")
    def test_run_trials_passes_time_cap_to_trainable_and_guard(
        self, mock_with_params, mock_tuner_class, mock_resources, mock_init
    ):
        mock_context = MagicMock()
        mock_context.dashboard_url = "http://localhost:8265"
        mock_init.return_value = mock_context
        mock_resources.return_value = {}
        mock_tuner = MagicMock()
        _fit_result = MagicMock(spec=ResultGrid, num_errors=0)
        _fit_result.__len__.return_value = 10
        mock_tuner.fit.return_value = _fit_result
        mock_tuner_class.return_value = mock_tuner

        run_trials(
            tracking_uri=self.mlflow_uri,
            exp_name="xgb",
            trainable=MagicMock(),
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            path2exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            k_folds=5,
            max_trial_duration_s=1200.0,
        )

        # Trainable receives the cap (fold-boundary stop) ...
        self.assertEqual(
            mock_with_params.call_args.kwargs["max_trial_duration_s"], 1200.0
        )
        # ... while the stopper-side cap stays disabled in K-fold mode (the
        # trainables cap themselves at fold boundaries).
        run_config = mock_tuner_class.call_args.kwargs["run_config"]
        self.assertFalse(
            run_config.stop("t0", {"rmse_val_mean": 1.0, "time_total_s": 1201.0})
        )

        run_trials(
            tracking_uri=self.mlflow_uri,
            exp_name="xgb",
            trainable=MagicMock(),
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            path2exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            k_folds=1,
            max_trial_duration_s=1200.0,
        )
        # Single-split xgb/nn have no in-trainable cap: the stopper enforces
        # it at report (iteration/epoch) boundaries.
        run_config = mock_tuner_class.call_args.kwargs["run_config"]
        self.assertTrue(
            run_config.stop("t0", {"rmse_val": 1.0, "time_total_s": 1201.0})
        )
        self.assertFalse(run_config.stop("t0", {"rmse_val": 1.0, "time_total_s": 10.0}))

    @patch("ritme.tune_models.init")
    @patch("ritme.tune_models.ray.cluster_resources")
    @patch("ritme.tune_models.tune.Tuner")
    def test_run_trials_enables_actor_reuse_and_pending_pipeline(
        self, mock_tuner_class, mock_resources, mock_init
    ):
        mock_context = MagicMock()
        mock_context.dashboard_url = "http://localhost:8265"
        mock_init.return_value = mock_context
        mock_resources.return_value = {}
        mock_tuner = MagicMock()
        _fit_result = MagicMock(spec=ResultGrid, num_errors=0)
        _fit_result.__len__.return_value = 10
        mock_tuner.fit.return_value = _fit_result
        mock_tuner_class.return_value = mock_tuner

        with patch.dict(os.environ, {}, clear=False), patch(
            "ritme.tune_models._SELF_SET_MAX_PENDING", None
        ):
            os.environ.pop("TUNE_MAX_PENDING_TRIALS_PG", None)
            run_trials(
                tracking_uri=self.mlflow_uri,
                exp_name="linreg",
                trainable=MagicMock(),
                train_val=self.train_val,
                target=self.target,
                host_id=self.host_id,
                stratify_by=None,
                seed_data=self.seed_data,
                seed_model=self.seed_model,
                tax=self.tax,
                tree_phylo=self.tree_phylo,
                path2exp=self.path2exp,
                time_budget_s=self.time_budget_s,
                max_concurrent_trials=8,
                experiment_tag=self.experiment_tag,
            )
            self.assertEqual(os.environ["TUNE_MAX_PENDING_TRIALS_PG"], "8")

        tune_config = mock_tuner_class.call_args.kwargs["tune_config"]
        self.assertTrue(tune_config.reuse_actors)

    @patch("ritme.tune_models.init")
    @patch("ritme.tune_models.ray.cluster_resources")
    @patch("ritme.tune_models.tune.Tuner")
    @patch("ritme.tune_models._get_resources")
    def test_run_trials_sizes_default_resources_by_family_and_folds(
        self, mock_get_resources, mock_tuner_class, mock_resources, mock_init
    ):
        mock_context = MagicMock()
        mock_context.dashboard_url = "http://localhost:8265"
        mock_init.return_value = mock_context
        mock_resources.return_value = {}
        mock_get_resources.return_value = {"cpu": 1, "gpu": 0}
        mock_tuner = MagicMock()
        _fit_result = MagicMock(spec=ResultGrid, num_errors=0)
        _fit_result.__len__.return_value = 10
        mock_tuner.fit.return_value = _fit_result
        mock_tuner_class.return_value = mock_tuner

        run_trials(
            tracking_uri=self.mlflow_uri,
            exp_name="linreg",
            trainable=MODEL_TRAINABLES["linreg"],
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            path2exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            k_folds=5,
        )

        mock_get_resources.assert_called_once_with(
            self.max_concurrent_trials, "linreg", 5
        )

    def test_run_trials_rejects_degenerate_time_cap(self):
        with self.assertRaisesRegex(ValueError, "max_trial_duration_s"):
            run_trials(
                tracking_uri=self.mlflow_uri,
                exp_name="xgb",
                trainable=MagicMock(),
                train_val=self.train_val,
                target=self.target,
                host_id=self.host_id,
                stratify_by=None,
                seed_data=self.seed_data,
                seed_model=self.seed_model,
                tax=self.tax,
                tree_phylo=self.tree_phylo,
                path2exp=self.path2exp,
                time_budget_s=self.time_budget_s,
                max_concurrent_trials=self.max_concurrent_trials,
                experiment_tag=self.experiment_tag,
                max_trial_duration_s=0,
            )

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_forwards_time_cap(self, mock_run_trials):
        mock_run_trials.return_value = MagicMock(spec=ResultGrid)

        run_all_trials(
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            model_types=["xgb"],
            max_trial_duration_s=900.0,
        )

        kwargs = mock_run_trials.call_args.kwargs
        self.assertEqual(kwargs["max_trial_duration_s"], 900.0)

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials(self, mock_run_trials):
        mock_result = MagicMock(spec=ResultGrid)
        mock_run_trials.return_value = mock_result

        model_types = ["xgb", "nn_reg"]
        results = run_all_trials(
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            model_types=model_types,
            model_hyperparameters=self.model_hyperparameters,
        )

        # Assertions
        self.assertEqual(len(results), len(model_types))
        for model in model_types:
            self.assertIn(model, results)
            self.assertEqual(results[model], mock_result)
        self.assertEqual(mock_run_trials.call_count, len(model_types))

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_defensive_validator_fires(self, mock_run_trials):
        # The defensive _validate_run_inputs call inside run_all_trials
        # protects standalone callers that bypass find_best_model_config.
        # Pins the call site (tune_models.py inside run_all_trials) so a
        # future refactor cannot silently delete it.
        with self.assertRaisesRegex(ValueError, "Invalid task_type"):
            run_all_trials(
                train_val=self.train_val,
                target=self.target,
                host_id=self.host_id,
                stratify_by=None,
                seed_data=self.seed_data,
                seed_model=self.seed_model,
                tax=self.tax,
                tree_phylo=self.tree_phylo,
                mlflow_uri=self.mlflow_uri,
                path_exp=self.path2exp,
                time_budget_s=self.time_budget_s,
                max_concurrent_trials=self.max_concurrent_trials,
                experiment_tag=self.experiment_tag,
                model_types=["linreg"],
                model_hyperparameters=self.model_hyperparameters,
                task_type="not_a_task",
            )
        mock_run_trials.assert_not_called()

    @patch("ritme.tune_models._get_resources")
    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_trac_resources_sized_from_original_concurrency(
        self, mock_run_trials, mock_get_resources
    ):
        # trac's memory workaround still reduces the launched concurrency to
        # a third, but the per-trial CPU reservation must be sized from the
        # ORIGINAL concurrency + family appetite -- the /3 reduction used to
        # triple the reservation of the one family that cannot thread.
        mock_run_trials.return_value = MagicMock(spec=ResultGrid)
        mock_get_resources.return_value = {"cpu": 5, "gpu": 0}

        run_all_trials(
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=9,
            experiment_tag=self.experiment_tag,
            model_types=["trac"],
            k_folds=5,
        )

        mock_get_resources.assert_called_once_with(9, "trac", 5, launched_concurrency=3)
        call = mock_run_trials.call_args
        bound = inspect.signature(run_trials).bind(*call.args, **call.kwargs)
        # Launched concurrency stays memory-reduced (9 / 3 = 3) ...
        self.assertEqual(bound.arguments["max_concurrent_trials"], 3)
        # ... while the reservation from the original concurrency is used.
        self.assertEqual(bound.arguments["resources"], {"cpu": 5, "gpu": 0})

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_remove_trac(self, mock_run_trials):
        mock_result = MagicMock(spec=ResultGrid)
        mock_run_trials.return_value = mock_result

        tax = None
        tree_phylo = None
        model_types = ["rf", "trac"]
        results = run_all_trials(
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=tax,
            tree_phylo=tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            model_types=model_types,
            model_hyperparameters=self.model_hyperparameters,
        )

        self.assertNotIn("trac", results)
        self.assertIn("rf", results)

        # Since 'trac' is removed, run_trials should be called once only for
        # 'rf' model
        mock_run_trials.assert_called_once_with(
            self.mlflow_uri,
            "rf",
            MODEL_TRAINABLES["rf"],
            self.train_val,
            self.target,
            self.host_id,
            None,
            self.seed_data,
            self.seed_model,
            tax,
            tree_phylo,
            self.path2exp,
            self.time_budget_s,
            self.max_concurrent_trials,
            self.experiment_tag,
            fully_reproducible=False,
            model_hyperparameters={"data_enrich_with": None},
            optuna_searchspace_sampler="TPESampler",
            scheduler_grace_period=None,
            scheduler_max_t=None,
            resources=None,
            task_type="regression",
            k_folds=1,
            nn_corn_max_levels=20,
            max_trial_failure_rate=0.005,
            max_trial_duration_s=None,
            max_pending_trials=None,
        )

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_remove_trac_due_to_snapshots(self, mock_run_trials):
        # create dataframe with snapshot columns
        self.train_val = pd.DataFrame(
            {
                "F1": [0.1, 0.2],
                "F2": [0.3, 0.4],
                "F1__t-1": [0.05, 0.15],
                "F2__t-1": [0.25, 0.35],
                "meta": [1, 2],
                self.target: [0.5, 0.6],
                self.host_id: ["a", "b"],
            }
        )
        mock_result = MagicMock(spec=ResultGrid)
        mock_run_trials.return_value = mock_result
        model_types = ["xgb", "trac"]
        results = run_all_trials(
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            model_types=model_types,
            model_hyperparameters=self.model_hyperparameters,
        )
        self.assertIn("xgb", results)
        self.assertNotIn("trac", results)

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_nan_snapshots_rejects_non_nan_tolerant(
        self, mock_run_trials
    ):
        self.train_val = pd.DataFrame(
            {
                "F1": [0.1, 0.2],
                "F2": [0.3, 0.4],
                "F1__t-1": [np.nan, 0.15],
                "F2__t-1": [0.25, np.nan],
                "meta": [1, 2],
                self.target: [0.5, 0.6],
                self.host_id: ["a", "b"],
            }
        )
        mock_result = MagicMock(spec=ResultGrid)
        mock_run_trials.return_value = mock_result
        model_types = ["xgb", "rf", "linreg", "trac"]
        with self.assertRaises(ValueError) as ctx:
            run_all_trials(
                train_val=self.train_val,
                target=self.target,
                host_id=self.host_id,
                stratify_by=None,
                seed_data=self.seed_data,
                seed_model=self.seed_model,
                tax=self.tax,
                tree_phylo=self.tree_phylo,
                mlflow_uri=self.mlflow_uri,
                path_exp=self.path2exp,
                time_budget_s=self.time_budget_s,
                max_concurrent_trials=self.max_concurrent_trials,
                experiment_tag=self.experiment_tag,
                model_types=model_types,
                model_hyperparameters=self.model_hyperparameters,
            )
        msg = str(ctx.exception)
        self.assertIn("NaNs in snapshot features", msg)
        # Regression task: allowed list must mention rf+xgb and NOT the
        # *_class counterparts.
        self.assertIn("'xgb'", msg)
        self.assertIn("'rf'", msg)
        self.assertNotIn("'xgb_class'", msg)
        self.assertNotIn("'rf_class'", msg)
        mock_run_trials.assert_not_called()

    @parameterized.expand(
        [
            (["xgb"],),
            (["rf"],),
            (["xgb", "rf"],),
        ]
    )
    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_nan_snapshots_xgb_or_rf_ok(
        self, model_types, mock_run_trials
    ):
        self.train_val = pd.DataFrame(
            {
                "F1": [0.1, 0.2],
                "F2": [0.3, 0.4],
                "F1__t-1": [np.nan, 0.15],
                "F2__t-1": [0.25, np.nan],
                "meta": [1, 2],
                self.target: [0.5, 0.6],
                self.host_id: ["a", "b"],
            }
        )
        mock_result = MagicMock(spec=ResultGrid)
        mock_run_trials.return_value = mock_result
        results = run_all_trials(
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            model_types=list(model_types),
            model_hyperparameters=self.model_hyperparameters,
        )
        self.assertEqual(sorted(results.keys()), sorted(model_types))

    @patch("ritme.tune_models.init")
    @patch("ritme.tune_models.ray.cluster_resources")
    @patch("ritme.tune_models.tune.Tuner")
    def test_run_trials_classification(
        self, mock_tuner_class, mock_resources, mock_init
    ):
        mock_context = MagicMock()
        mock_context.dashboard_url = "http://localhost:8265"
        mock_init.return_value = mock_context

        mock_resources.return_value = {}

        mock_tuner = MagicMock()
        _fit_result = MagicMock(spec=ResultGrid, num_errors=0)
        _fit_result.__len__.return_value = 10
        mock_tuner.fit.return_value = _fit_result
        mock_tuner_class.return_value = mock_tuner

        trainable = MagicMock()

        result = run_trials(
            tracking_uri=self.mlflow_uri,
            exp_name="logreg",
            trainable=trainable,
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            path2exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            fully_reproducible=False,
            model_hyperparameters=self.model_hyperparameters,
            task_type="classification",
        )

        # Verify the scheduler -- which now owns metric/mode (Tuner-level
        # metric/mode is intentionally None to avoid Ray Tune's "metric set
        # in both scheduler and Tuner" guard, see issue_eval_class.md) --
        # was configured with the classification metric.
        tuner_call_kwargs = mock_tuner_class.call_args
        tune_config = tuner_call_kwargs.kwargs["tune_config"]
        self.assertIsNone(tune_config.metric)
        self.assertIsNone(tune_config.mode)
        # K-fold mode reads ``<metric>_mean``; single-split reads ``<metric>``.
        # ``k_folds`` defaults to 1 in ``run_trials`` so this assertion
        # tracks the single-split branch.
        self.assertEqual(tune_config.scheduler._metric, "roc_auc_macro_ovr_val")
        self.assertEqual(tune_config.scheduler._mode, "max")

        # Verify checkpoint config also uses classification metric
        run_config = tuner_call_kwargs.kwargs["run_config"]
        self.assertEqual(
            run_config.checkpoint_config.checkpoint_score_attribute,
            "roc_auc_macro_ovr_val",
        )
        self.assertEqual(run_config.checkpoint_config.checkpoint_score_order, "max")

        self.assertIsInstance(result, ResultGrid)

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_classification(self, mock_run_trials):
        mock_result = MagicMock(spec=ResultGrid)
        mock_run_trials.return_value = mock_result

        model_types = ["logreg", "rf_class"]
        results = run_all_trials(
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            model_types=model_types,
            model_hyperparameters=self.model_hyperparameters,
            task_type="classification",
        )

        # Assertions
        self.assertEqual(len(results), len(model_types))
        for model in model_types:
            self.assertIn(model, results)
            self.assertEqual(results[model], mock_result)
        self.assertEqual(mock_run_trials.call_count, len(model_types))

        # Verify task_type="classification" was passed through to run_trials
        for call in mock_run_trials.call_args_list:
            self.assertEqual(call.kwargs.get("task_type"), "classification")

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_nan_snapshots_rejects_non_nan_tolerant_class(
        self, mock_run_trials
    ):
        self.train_val = pd.DataFrame(
            {
                "F1": [0.1, 0.2],
                "F2": [0.3, 0.4],
                "F1__t-1": [np.nan, 0.15],
                "F2__t-1": [0.25, np.nan],
                "meta": [1, 2],
                self.target: [0.5, 0.6],
                self.host_id: ["a", "b"],
            }
        )
        model_types = ["xgb_class", "rf_class", "logreg"]
        with self.assertRaises(ValueError) as ctx:
            run_all_trials(
                train_val=self.train_val,
                target=self.target,
                host_id=self.host_id,
                stratify_by=None,
                seed_data=self.seed_data,
                seed_model=self.seed_model,
                tax=self.tax,
                tree_phylo=self.tree_phylo,
                mlflow_uri=self.mlflow_uri,
                path_exp=self.path2exp,
                time_budget_s=self.time_budget_s,
                max_concurrent_trials=self.max_concurrent_trials,
                experiment_tag=self.experiment_tag,
                model_types=model_types,
                model_hyperparameters=self.model_hyperparameters,
                task_type="classification",
            )
        msg = str(ctx.exception)
        self.assertIn("NaNs in snapshot features", msg)
        # Classification task: allowed list must mention *_class
        # counterparts and NOT the bare regression names.
        self.assertIn("'xgb_class'", msg)
        self.assertIn("'rf_class'", msg)
        self.assertNotIn("'xgb'", msg)
        self.assertNotIn("'rf'", msg)
        mock_run_trials.assert_not_called()

    @parameterized.expand(
        [
            (["xgb_class"],),
            (["rf_class"],),
            (["xgb_class", "rf_class"],),
        ]
    )
    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_nan_snapshots_xgb_class_or_rf_class_ok(
        self, model_types, mock_run_trials
    ):
        self.train_val = pd.DataFrame(
            {
                "F1": [0.1, 0.2],
                "F2": [0.3, 0.4],
                "F1__t-1": [np.nan, 0.15],
                "F2__t-1": [0.25, np.nan],
                "meta": [1, 2],
                self.target: [0.5, 0.6],
                self.host_id: ["a", "b"],
            }
        )
        mock_result = MagicMock(spec=ResultGrid)
        mock_run_trials.return_value = mock_result
        results = run_all_trials(
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            model_types=list(model_types),
            model_hyperparameters=self.model_hyperparameters,
            task_type="classification",
        )
        self.assertEqual(sorted(results.keys()), sorted(model_types))

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_classification_hparams_fallback_rf_class(
        self, mock_run_trials
    ):
        mock_result = MagicMock(spec=ResultGrid)
        mock_run_trials.return_value = mock_result

        # Provide hparams under "rf" key only - rf_class should fall back to it
        model_hyperparameters = {
            "rf": {"n_estimators": {"min": 10, "max": 200}},
        }
        model_types = ["rf_class"]
        run_all_trials(
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            model_types=model_types,
            model_hyperparameters=model_hyperparameters,
            task_type="classification",
        )

        # Verify run_trials received the "rf" hparams for rf_class
        call_kwargs = mock_run_trials.call_args
        fallback = call_kwargs[0][15] if len(call_kwargs[0]) > 15 else None
        passed_hparams = call_kwargs.kwargs.get("model_hyperparameters", fallback)
        self.assertIn("n_estimators", passed_hparams)
        self.assertEqual(passed_hparams["n_estimators"], {"min": 10, "max": 200})

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_classification_hparams_fallback_xgb_class(
        self, mock_run_trials
    ):
        mock_result = MagicMock(spec=ResultGrid)
        mock_run_trials.return_value = mock_result

        # Provide hparams under "xgb" key only - xgb_class should fall back to it
        model_hyperparameters = {
            "xgb": {"n_estimators": {"min": 50, "max": 500}},
        }
        model_types = ["xgb_class"]
        run_all_trials(
            train_val=self.train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            model_types=model_types,
            model_hyperparameters=model_hyperparameters,
            task_type="classification",
        )

        # Verify run_trials received the "xgb" hparams for xgb_class
        call_kwargs = mock_run_trials.call_args
        fallback = call_kwargs[0][15] if len(call_kwargs[0]) > 15 else None
        passed_hparams = call_kwargs.kwargs.get("model_hyperparameters", fallback)
        self.assertIn("n_estimators", passed_hparams)
        self.assertEqual(passed_hparams["n_estimators"], {"min": 50, "max": 500})

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_nn_corn_level_cap_rejects_wide_target(
        self, mock_run_trials
    ):
        """``run_all_trials`` validates the ``nn_corn`` level cap up-front by
        rounding the full ``train_val[target]`` before any trial is launched.
        With 30 distinct rounded levels and a cap of 5 the cap check must
        raise ``ValueError`` and ``run_trials`` must never be invoked.
        """
        n_rows = 30
        train_val = pd.DataFrame(
            {
                "F0": np.linspace(0.0, 1.0, n_rows),
                "F1": np.linspace(1.0, 2.0, n_rows),
                self.target: np.arange(n_rows, dtype=float),
                self.host_id: [f"h{i}" for i in range(n_rows)],
            }
        )

        with self.assertRaisesRegex(ValueError, r"nn_corn_max_levels"):
            run_all_trials(
                train_val=train_val,
                target=self.target,
                host_id=self.host_id,
                stratify_by=None,
                seed_data=self.seed_data,
                seed_model=self.seed_model,
                tax=self.tax,
                tree_phylo=self.tree_phylo,
                mlflow_uri=self.mlflow_uri,
                path_exp=self.path2exp,
                time_budget_s=self.time_budget_s,
                max_concurrent_trials=self.max_concurrent_trials,
                experiment_tag=self.experiment_tag,
                model_types=["nn_corn"],
                model_hyperparameters=self.model_hyperparameters,
                nn_corn_max_levels=5,
            )
        mock_run_trials.assert_not_called()

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_nn_corn_level_cap_allows_narrow_target(
        self, mock_run_trials
    ):
        """When the rounded-target level count is at or below
        ``nn_corn_max_levels`` the up-front check passes and the trial is
        launched as usual.
        """
        mock_run_trials.return_value = MagicMock(spec=ResultGrid)
        n_rows = 30
        # 3 distinct rounded levels well below the cap of 5.
        narrow_target = np.tile([0.0, 1.0, 2.0], n_rows // 3)[:n_rows]
        train_val = pd.DataFrame(
            {
                "F0": np.linspace(0.0, 1.0, n_rows),
                "F1": np.linspace(1.0, 2.0, n_rows),
                self.target: narrow_target,
                self.host_id: [f"h{i}" for i in range(n_rows)],
            }
        )

        results = run_all_trials(
            train_val=train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            model_types=["nn_corn"],
            model_hyperparameters=self.model_hyperparameters,
            nn_corn_max_levels=5,
        )
        self.assertIn("nn_corn", results)
        # nn_corn_max_levels must be forwarded to run_trials so the trainable
        # safety net inside train_nn sees the same cap as the up-front check.
        forwarded = mock_run_trials.call_args.kwargs.get("nn_corn_max_levels")
        self.assertEqual(forwarded, 5)

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_nn_corn_level_cap_boundary_exactly_at_cap(
        self, mock_run_trials
    ):
        """Boundary: ``n_levels == nn_corn_max_levels`` must pass (`>` check,
        not `>=`). 5 distinct rounded levels with cap 5 should launch.
        """
        mock_run_trials.return_value = MagicMock(spec=ResultGrid)
        n_rows = 30
        # 5 distinct rounded levels exactly equal to the cap.
        target = np.tile([0.0, 1.0, 2.0, 3.0, 4.0], n_rows // 5)[:n_rows]
        train_val = pd.DataFrame(
            {
                "F0": np.linspace(0.0, 1.0, n_rows),
                "F1": np.linspace(1.0, 2.0, n_rows),
                self.target: target,
                self.host_id: [f"h{i}" for i in range(n_rows)],
            }
        )

        run_all_trials(
            train_val=train_val,
            target=self.target,
            host_id=self.host_id,
            stratify_by=None,
            seed_data=self.seed_data,
            seed_model=self.seed_model,
            tax=self.tax,
            tree_phylo=self.tree_phylo,
            mlflow_uri=self.mlflow_uri,
            path_exp=self.path2exp,
            time_budget_s=self.time_budget_s,
            max_concurrent_trials=self.max_concurrent_trials,
            experiment_tag=self.experiment_tag,
            model_types=["nn_corn"],
            model_hyperparameters=self.model_hyperparameters,
            nn_corn_max_levels=5,
        )
        mock_run_trials.assert_called_once()

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_nn_corn_level_cap_boundary_one_over_cap(
        self, mock_run_trials
    ):
        """Boundary: ``n_levels == nn_corn_max_levels + 1`` must raise."""
        n_rows = 30
        # 6 distinct rounded levels, one over the cap of 5.
        target = np.tile([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], n_rows // 6)[:n_rows]
        train_val = pd.DataFrame(
            {
                "F0": np.linspace(0.0, 1.0, n_rows),
                "F1": np.linspace(1.0, 2.0, n_rows),
                self.target: target,
                self.host_id: [f"h{i}" for i in range(n_rows)],
            }
        )

        with self.assertRaisesRegex(ValueError, r"nn_corn_max_levels"):
            run_all_trials(
                train_val=train_val,
                target=self.target,
                host_id=self.host_id,
                stratify_by=None,
                seed_data=self.seed_data,
                seed_model=self.seed_model,
                tax=self.tax,
                tree_phylo=self.tree_phylo,
                mlflow_uri=self.mlflow_uri,
                path_exp=self.path2exp,
                time_budget_s=self.time_budget_s,
                max_concurrent_trials=self.max_concurrent_trials,
                experiment_tag=self.experiment_tag,
                model_types=["nn_corn"],
                model_hyperparameters=self.model_hyperparameters,
                nn_corn_max_levels=5,
            )
        mock_run_trials.assert_not_called()

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_nn_corn_rejects_non_numeric_target(self, mock_run_trials):
        """Non-numeric target must raise a clear ``ValueError`` instead of a
        cryptic ``TypeError`` from ``np.round`` on an object array.
        """
        n_rows = 12
        train_val = pd.DataFrame(
            {
                "F0": np.linspace(0.0, 1.0, n_rows),
                "F1": np.linspace(1.0, 2.0, n_rows),
                self.target: ["low", "medium", "high"] * (n_rows // 3),
                self.host_id: [f"h{i}" for i in range(n_rows)],
            }
        )

        with self.assertRaisesRegex(ValueError, r"numeric target"):
            run_all_trials(
                train_val=train_val,
                target=self.target,
                host_id=self.host_id,
                stratify_by=None,
                seed_data=self.seed_data,
                seed_model=self.seed_model,
                tax=self.tax,
                tree_phylo=self.tree_phylo,
                mlflow_uri=self.mlflow_uri,
                path_exp=self.path2exp,
                time_budget_s=self.time_budget_s,
                max_concurrent_trials=self.max_concurrent_trials,
                experiment_tag=self.experiment_tag,
                model_types=["nn_corn"],
                model_hyperparameters=self.model_hyperparameters,
            )
        mock_run_trials.assert_not_called()

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_nn_corn_rejects_nan_target(self, mock_run_trials):
        """NaN-bearing target must raise rather than silently coerce NaN to 0
        via ``np.round(...).astype(int)``.
        """
        n_rows = 12
        target = np.array([0.0, 1.0, np.nan, 2.0] * (n_rows // 4), dtype=float)
        train_val = pd.DataFrame(
            {
                "F0": np.linspace(0.0, 1.0, n_rows),
                "F1": np.linspace(1.0, 2.0, n_rows),
                self.target: target,
                self.host_id: [f"h{i}" for i in range(n_rows)],
            }
        )

        with self.assertRaisesRegex(ValueError, r"NaN"):
            run_all_trials(
                train_val=train_val,
                target=self.target,
                host_id=self.host_id,
                stratify_by=None,
                seed_data=self.seed_data,
                seed_model=self.seed_model,
                tax=self.tax,
                tree_phylo=self.tree_phylo,
                mlflow_uri=self.mlflow_uri,
                path_exp=self.path2exp,
                time_budget_s=self.time_budget_s,
                max_concurrent_trials=self.max_concurrent_trials,
                experiment_tag=self.experiment_tag,
                model_types=["nn_corn"],
                model_hyperparameters=self.model_hyperparameters,
            )
        mock_run_trials.assert_not_called()

    @patch("ritme.tune_models.run_trials")
    def test_run_all_trials_rejects_invalid_cap_values(self, mock_run_trials):
        """``nn_corn_max_levels`` must be an ``int`` >= 2. Garbage values
        (zero, negative, string) raise a clear ``ValueError`` rather than
        silently disabling the guard or rejecting every run.
        """
        n_rows = 12
        train_val = pd.DataFrame(
            {
                "F0": np.linspace(0.0, 1.0, n_rows),
                "F1": np.linspace(1.0, 2.0, n_rows),
                self.target: [0.0, 1.0, 2.0] * (n_rows // 3),
                self.host_id: [f"h{i}" for i in range(n_rows)],
            }
        )

        for invalid_cap in (0, 1, -5, "20", None, 1.5):
            with self.subTest(invalid_cap=invalid_cap):
                with self.assertRaises(ValueError):
                    run_all_trials(
                        train_val=train_val,
                        target=self.target,
                        host_id=self.host_id,
                        stratify_by=None,
                        seed_data=self.seed_data,
                        seed_model=self.seed_model,
                        tax=self.tax,
                        tree_phylo=self.tree_phylo,
                        mlflow_uri=self.mlflow_uri,
                        path_exp=self.path2exp,
                        time_budget_s=self.time_budget_s,
                        max_concurrent_trials=self.max_concurrent_trials,
                        experiment_tag=self.experiment_tag,
                        model_types=["nn_corn"],
                        model_hyperparameters=self.model_hyperparameters,
                        nn_corn_max_levels=invalid_cap,
                    )
        mock_run_trials.assert_not_called()


class TestAdaptiveNStartupTrials(unittest.TestCase):
    """The pre-K-fold default of 1000 startup trials consumed nearly the
    entire time budget on most ritme runs. The adaptive helper sizes the
    random-sampling phase from each model's effective search-space dim,
    counted by introspecting the actual search space along its longest
    conditional path."""

    def setUp(self):
        # Minimal but non-empty train_val: the search space's threshold-bounds
        # path needs at least one F-prefixed feature with non-zero variance
        # in case the recording trial wanders there. Three rows is enough.
        self.train_val = pd.DataFrame(
            {
                "F0": [1.0, 2.0, 3.0],
                "F1": [0.1, 0.5, 0.9],
                "F2": [4.0, 5.0, 6.0],
                "md0": [0, 1, 0],
            }
        )
        self.tax = None

    def _startup(self, model_type: str, hparams: dict | None = None) -> int:
        return _adaptive_n_startup_trials(
            model_type, self.train_val, self.tax, hparams or {}
        )

    def test_floor_for_low_dim_models(self):
        # TRAC has only `lambda` -> 1 dim, falls back to the floor.
        self.assertEqual(self._startup("trac"), 20)

    def test_per_model_defaults_match_search_space_long_path(self):
        # 5 data-eng dims + alpha + l1_ratio -> 7 -> 5 * 7 = 35
        self.assertEqual(self._startup("linreg"), 35)
        # 5 + (C, penalty, l1_ratio) -> 8 -> 40
        self.assertEqual(self._startup("logreg"), 40)
        # 5 + 8 RF model dims -> 13 -> 65
        self.assertEqual(self._startup("rf"), 65)
        self.assertEqual(self._startup("rf_class"), 65)
        # 5 + 10 XGB model dims -> 15 -> 75
        self.assertEqual(self._startup("xgb"), 75)
        self.assertEqual(self._startup("xgb_class"), 75)

    def test_nn_uses_max_n_hidden_layers(self):
        # Default range [1, 30]: longest path has 30 hidden layers
        # -> 5 (data eng) + 8 (fixed nn params) + 30 (per-layer widths) = 43
        # -> 5 * 43 = 215
        self.assertEqual(self._startup("nn_reg"), 215)
        # User-supplied tighter range [1, 5]: longest path has 5 hidden layers
        # -> 5 + 8 + 5 = 18 -> 5 * 18 = 90
        self.assertEqual(
            self._startup("nn_class", {"n_hidden_layers": {"min": 1, "max": 5}}),
            90,
        )
        # min == max collapses to a single layer count: 5 + 8 + 4 = 17 -> 85
        self.assertEqual(
            self._startup("nn_corn", {"n_hidden_layers": {"min": 4, "max": 4}}),
            85,
        )

    def test_unknown_model_type_raises(self):
        # The recording-trial path delegates to ss.get_search_space, which
        # raises ValueError on unknown model types. This is desired: silent
        # fallbacks would mask configuration mistakes.
        from ritme.model_space import static_searchspace  # noqa: F401

        with self.assertRaises(ValueError):
            self._startup("unknown_model")


class TestRecordingTrialSteering(unittest.TestCase):
    """The adaptive ``n_startup_trials`` is sized from the longest conditional
    path of each model's search space, introspected via ``_RecordingTrial``.
    The recording trial's correctness rests on two steering invariants:
    categorical picks that trigger dependent branches (``data_selection`` ->
    ``"abundance_ith"``, ``penalty`` -> ``"elasticnet"``) and returning
    ``high`` for ``n_hidden_layers`` so every per-layer width fires. If a
    search-space parameter is renamed or a conditional branch is restructured,
    the steering can silently undercount dims. These tests lock in the
    presence of the specific parameter names that prove the steering worked,
    so drift fails loudly here rather than silently breaking ``n_startup``.
    """

    def setUp(self):
        # Minimal but non-empty train_val: a few F-prefixed feature columns
        # plus one md column. Matches the fixture used in
        # TestAdaptiveNStartupTrials so the recording-trial path is exercised
        # under realistic-shaped data.
        self.train_val = pd.DataFrame(
            {
                "F0": [1.0, 2.0, 3.0],
                "F1": [0.5, 1.5, 2.5],
                "F2": [0.1, 0.2, 0.3],
                "md0": [0, 1, 0],
            }
        )
        self.tax = None

    def test_linreg_records_data_selection_i_dependent_suggestion(self):
        trial = _RecordingTrial()
        ss.get_search_space(
            trial,
            model_type="linreg",
            tax=self.tax,
            train_val=self.train_val,
            model_hyperparameters={},
        )
        self.assertIn("data_selection_i", trial.params)

    def test_logreg_records_l1_ratio_under_elasticnet_penalty(self):
        trial = _RecordingTrial()
        ss.get_search_space(
            trial,
            model_type="logreg",
            tax=self.tax,
            train_val=self.train_val,
            model_hyperparameters={},
        )
        self.assertIn("l1_ratio", trial.params)

    def test_rf_records_data_selection_dependent_suggestion(self):
        trial = _RecordingTrial()
        ss.get_search_space(
            trial,
            model_type="rf",
            tax=self.tax,
            train_val=self.train_val,
            model_hyperparameters={},
        )
        self.assertIn("data_selection_i", trial.params)

    def test_xgb_records_data_selection_dependent_suggestion(self):
        trial = _RecordingTrial()
        ss.get_search_space(
            trial,
            model_type="xgb",
            tax=self.tax,
            train_val=self.train_val,
            model_hyperparameters={},
        )
        self.assertIn("data_selection_i", trial.params)

    def test_nn_reg_records_all_per_layer_widths_at_default_range(self):
        trial = _RecordingTrial()
        ss.get_search_space(
            trial,
            model_type="nn_reg",
            tax=self.tax,
            train_val=self.train_val,
            model_hyperparameters={},
        )
        expected = {f"n_units_hl{i}" for i in range(30)}
        recorded = {k for k in trial.params if k.startswith("n_units_hl")}
        self.assertEqual(recorded, expected)

    def test_nn_reg_records_widths_for_custom_range(self):
        trial = _RecordingTrial()
        ss.get_search_space(
            trial,
            model_type="nn_reg",
            tax=self.tax,
            train_val=self.train_val,
            model_hyperparameters={"n_hidden_layers": {"min": 1, "max": 5}},
        )
        expected = {f"n_units_hl{i}" for i in range(5)}
        recorded = {k for k in trial.params if k.startswith("n_units_hl")}
        self.assertEqual(recorded, expected)

    def test_nn_reg_records_widths_for_collapsed_range(self):
        trial = _RecordingTrial()
        ss.get_search_space(
            trial,
            model_type="nn_reg",
            tax=self.tax,
            train_val=self.train_val,
            model_hyperparameters={"n_hidden_layers": {"min": 4, "max": 4}},
        )
        expected = {f"n_units_hl{i}" for i in range(4)}
        recorded = {k for k in trial.params if k.startswith("n_units_hl")}
        self.assertEqual(recorded, expected)

    def test_trac_search_space_has_only_lambda(self):
        trial = _RecordingTrial()
        ss.get_search_space(
            trial,
            model_type="trac",
            tax=self.tax,
            train_val=self.train_val,
            model_hyperparameters={},
        )
        self.assertEqual(set(trial.params.keys()), {"lambda"})


if __name__ == "__main__":
    unittest.main()
