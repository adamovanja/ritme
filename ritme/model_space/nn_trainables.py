"""Neural-network tune trainables (torch / lightning dependent).

Kept separate from ``static_trainables`` (which must never import this
module) so Ray workers running non-nn trainables never import torch,
lightning, torchmetrics or coral_pytorch.
"""

import math
import os
import random
import shutil
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import pandas as pd
import ray
import torch
import torchmetrics
from coral_pytorch.dataset import corn_label_from_logits
from coral_pytorch.losses import corn_loss
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ritme.feature_space._process_train import process_train, process_train_kfold
from ritme.model_space.static_trainables import (
    DEFAULT_NN_CORN_MAX_LEVELS,
    _aggregate_fold_metrics,
    _check_nn_corn_levels,
    _classification_metrics_dict,
    _emit_running_fold_aggregate,
    _report_time_capped_aggregate,
    _save_label_encoder,
    _save_taxonomy,
    _time_cap_reached,
)


class NeuralNet(LightningModule):
    def __init__(
        self,
        n_units,
        learning_rate,
        nn_type="regression",
        dropout_rate=0.0,
        weight_decay=0.0,
        classes: Optional[list] = None,
        task_type: str = "regression",
    ):
        super(NeuralNet, self).__init__()
        self.save_hyperparameters()  # This saves all passed arguments to self.hparams
        self.learning_rate = learning_rate
        self.nn_type = nn_type
        self.task_type = task_type
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay

        self.input_norm = nn.BatchNorm1d(n_units[0])

        self.classes = classes
        if nn_type in ["classification", "ordinal_regression"]:
            self.class_to_index = {c: i for i, c in enumerate(classes)}
            self.index_to_class = {i: c for i, c in enumerate(classes)}
            self.num_classes = len(classes)
        self.layers = nn.ModuleList()
        n_layers = len(n_units)
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_units[i], n_units[i + 1]))
            if i != len(n_units) - 2:  # No activation after the last layer
                self.layers.append(nn.ReLU())
                if self.dropout_rate > 0:
                    self.layers.append(nn.Dropout(self.dropout_rate))

        self.train_loss = 0
        self.val_loss = 0
        self.train_predictions = []
        self.train_targets = []
        self.validation_predictions = []
        self.validation_targets = []

    def forward(self, x):
        x = self.input_norm(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def _prepare_predictions(self, predictions):
        if self.nn_type == "regression":
            return predictions
        elif self.nn_type == "classification":
            idx = torch.argmax(predictions, dim=1)
            # map back to original labels
            mapped = [self.index_to_class[int(i)] for i in idx.detach().cpu().numpy()]
            return torch.tensor(mapped, device=predictions.device, dtype=torch.float32)
        elif self.nn_type == "ordinal_regression":
            if predictions.ndim == 1:
                # [num_samples] must be [num_samples, 1] for corn
                predictions = predictions.unsqueeze(1)
            corn_label = corn_label_from_logits(predictions).float()
            # map back to original labels
            mapped = [
                self.index_to_class[int(i)] for i in corn_label.detach().cpu().numpy()
            ]
            return torch.tensor(mapped, device=predictions.device, dtype=torch.float32)

    def _predict_proba(self, predictions: torch.Tensor) -> np.ndarray:
        """Per-class probabilities for classification / CORN heads."""
        if self.nn_type == "classification":
            return torch.softmax(predictions, dim=1).detach().cpu().numpy()
        # CORN: conditional sigmoids -> cumulative P(y>k) -> per-class probs.
        if predictions.ndim == 1:
            predictions = predictions.unsqueeze(1)
        cum = torch.cumprod(torch.sigmoid(predictions), dim=1)
        proba = torch.cat(
            [1.0 - cum[:, :1], cum[:, :-1] - cum[:, 1:], cum[:, -1:]], dim=1
        )
        return proba.detach().cpu().numpy()

    def _calculate_metrics(self, predictions, targets):
        preds = self._prepare_predictions(predictions)

        if self.task_type == "regression":
            rmse = torch.sqrt(nn.functional.mse_loss(preds, targets))
            r2score = torchmetrics.regression.R2Score().to(preds.device)
            r2 = r2score(preds, targets)
            return {"rmse": rmse, "r2": r2}

        y_pred_np = preds.detach().cpu().numpy().astype(int)
        y_true_np = targets.detach().cpu().numpy().astype(int)
        y_proba_np = self._predict_proba(predictions)
        return _classification_metrics_dict(
            y_true_np, y_pred_np, y_proba_np, list(self.classes)
        )

    def _calculate_loss(self, predictions, targets):
        # loss: corn_loss, cross-entropy or mse
        # calculated on rounded classes as targets for ordinal regression and
        # classification
        targets_rounded = torch.round(targets).long()
        if self.nn_type == "ordinal_regression":
            # re-index to 0...C-1 for loss
            t = targets_rounded.detach().cpu().numpy().astype(int)
            t_idx = [self.class_to_index[v] for v in t]
            targets_rounded = torch.tensor(
                t_idx, device=targets.device, dtype=torch.long
            )
            # predictions = logits
            if predictions.ndim == 1:
                # [num_samples] must be [num_samples, 1] for corn_loss
                predictions = predictions.unsqueeze(1)
            return corn_loss(predictions, targets_rounded, self.num_classes)
        elif self.nn_type == "classification":
            # re-index to 0...C-1 for cross-entropy loss
            t = targets_rounded.detach().cpu().numpy().astype(int)
            t_idx = [self.class_to_index[v] for v in t]
            targets_rounded = torch.tensor(
                t_idx, device=targets.device, dtype=torch.long
            )

            loss_fn = nn.CrossEntropyLoss()
            # predictions = logits
            return loss_fn(predictions, targets_rounded)
        loss_fn = nn.MSELoss()
        return loss_fn(predictions, targets)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self.forward(inputs).squeeze()

        # Store predictions and targets
        self.train_predictions.append(predictions.detach())
        self.train_targets.append(targets.detach())

        self.train_loss = self._calculate_loss(predictions, targets)

        return self.train_loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self.forward(inputs).squeeze()

        self.validation_predictions.append(predictions.detach())
        self.validation_targets.append(targets.detach())

        self.val_loss = self._calculate_loss(predictions, targets)
        # nb_features log
        self.log("nb_features", inputs.shape[1])
        return {"val_loss": self.val_loss}

    def on_train_epoch_end(self):
        all_preds = torch.cat(self.train_predictions)
        all_targets = torch.cat(self.train_targets)
        loss = self._calculate_loss(all_preds, all_targets)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        metrics = self._calculate_metrics(all_preds, all_targets)
        for name, value in metrics.items():
            self.log(f"train_{name}", value, on_epoch=True, prog_bar=True, logger=True)
        self.train_predictions.clear()
        self.train_targets.clear()

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_predictions)
        all_targets = torch.cat(self.validation_targets)

        loss = self._calculate_loss(all_preds, all_targets)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        metrics = self._calculate_metrics(all_preds, all_targets)
        for name, value in metrics.items():
            self.log(f"val_{name}", value, on_epoch=True, prog_bar=True, logger=True)

        self.validation_predictions.clear()
        self.validation_targets.clear()

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_data(X_train, y_train, X_val, y_val, config, seed_model, num_workers=2):
    # fixed data loader - for reference on reproducibility:
    # https://docs.pytorch.org/docs/stable/notes/randomness.html

    # a Generator for shuffling
    g = torch.Generator()
    g.manual_seed(seed_model)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    # ``BatchNorm1d`` (used as ``NeuralNet.input_norm``) raises on a
    # 1-sample batch in train mode. A single-sample training set cannot
    # produce a usable batch under any ``batch_size``, so refuse it
    # explicitly rather than dropping silently to a zero-batch loader
    # that would record random-init metrics as a "successful" trial.
    n_train = len(train_dataset)
    if n_train < 2:
        raise ValueError(
            f"NeuralNet trainables require at least 2 training samples "
            f"(got {n_train}); BatchNorm1d cannot compute batch "
            f"statistics from a single sample."
        )
    # Drop the partial last batch ONLY when it would have size 1, so
    # larger remainders still contribute gradient updates and we lose
    # at most one sample per epoch (rotated each epoch because
    # ``shuffle=True``).
    drop_last_train = n_train % config["batch_size"] == 1
    if drop_last_train:
        # Surface the per-trial drop so coverage discrepancies in logs
        # are traceable; rationale lives in the comment above.
        print(
            f"NeuralNet load_data: drop_last=True "
            f"(n_train={n_train}, batch_size={config['batch_size']})."
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=drop_last_train,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return train_loader, val_loader


class NNTuneReportCheckpointCallback(TuneReportCheckpointCallback):
    """PyTorch Lightning callback that decouples metric reports from checkpoint writes.

    Reports metrics to Ray Tune after every validation epoch (so the ASHA
    scheduler can prune trials and intermediate progress is logged), and writes
    **at most two** Ray Tune checkpoints per trial:

    1. A safety write on the *first* validation improvement, so trials that
       are paused-then-killed by HyperBand still have at least one checkpoint
       on disk (otherwise ``result.checkpoint`` is ``None`` and downstream
       retrieval crashes).
    2. A final write in ``on_train_end`` containing the best validation state
       seen during the run -- score-based retention by ``CheckpointConfig``
       keeps the better of the two.

    Improvements *between* the first and the last are saved to a per-trial
    scratch directory only (no ``tune.report(checkpoint=...)`` call), so the
    experiment-state snapshotter is not triggered for them. Bounding writes to
    two per trial keeps it from being saturated by concurrent trials.

    Also injects ``nb_features`` into every reported metric dict (used by
    ``evaluate_models.py``).
    """

    def __init__(
        self,
        metrics: Optional[Union[str, List[str], Dict[str, str]]] = None,
        filename: str = "checkpoint",
        save_checkpoints: bool = True,
        on: Union[str, List[str]] = "validation_end",
        nb_features: int = None,
        score_attr: str = "rmse_val",
        score_mode: str = "min",
    ):
        super().__init__(
            metrics=metrics, filename=filename, save_checkpoints=save_checkpoints, on=on
        )
        self.nb_features = nb_features
        if score_mode not in ("min", "max"):
            raise ValueError(f"score_mode must be 'min' or 'max', got {score_mode!r}")
        self._score_attr = score_attr
        self._score_mode = score_mode
        self._best_score = float("inf") if score_mode == "min" else float("-inf")
        # Per-trial scratch dir holding the best Lightning checkpoint seen so
        # far. Lazily created on first improvement; cleaned up at on_train_end
        # after the contents have been reported to Ray Tune.
        self._best_scratch_dir: Optional[str] = None
        self._best_report_dict: Optional[Dict] = None
        # Tracks whether the first-improvement safety checkpoint has been
        # written. Used so we only pay the Ray Tune write cost once during
        # training (before the second write at on_train_end).
        self._wrote_safety_checkpoint = False

    def _is_improvement(self, score) -> bool:
        if score is None:
            return False
        try:
            score = float(score)
        except (TypeError, ValueError):
            return False
        if math.isnan(score):
            return False
        if self._score_mode == "min":
            return score < self._best_score
        return score > self._best_score

    def _build_report_dict(self, trainer, pl_module):
        report_dict = self._get_report_dict(trainer, pl_module)
        if not report_dict:
            return None
        report_dict["nb_features"] = self.nb_features
        return report_dict

    def _ensure_scratch_dir(self) -> str:
        if self._best_scratch_dir is None:
            self._best_scratch_dir = tempfile.mkdtemp(prefix="ritme_nn_best_")
        return self._best_scratch_dir

    def _handle(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.sanity_checking:
            return

        report_dict = self._build_report_dict(trainer, pl_module)
        if report_dict is None:
            return

        score = report_dict.get(self._score_attr)
        improved = self._is_improvement(score)
        if improved:
            self._best_score = score
            scratch_dir = self._ensure_scratch_dir()
            # Overwrite previous best on local disk (no Ray Tune visibility).
            trainer.save_checkpoint(os.path.join(scratch_dir, self._filename))
            self._best_report_dict = dict(report_dict)

        if improved and not self._wrote_safety_checkpoint:
            # Safety write: ensure paused-then-killed trials have a checkpoint.
            self._wrote_safety_checkpoint = True
            checkpoint = ray.train.Checkpoint.from_directory(self._best_scratch_dir)
            tune.report(report_dict, checkpoint=checkpoint)
        else:
            # Cheap metric-only report so ASHA can prune.
            tune.report(report_dict)

    def on_train_end(self, trainer, pl_module):
        # Final write of the best validation state seen during training.
        if self._best_scratch_dir is not None and self._best_report_dict is not None:
            checkpoint = ray.train.Checkpoint.from_directory(self._best_scratch_dir)
            tune.report(self._best_report_dict, checkpoint=checkpoint)
            # Ray Tune copies the checkpoint contents synchronously into its
            # storage during tune.report, so the scratch dir is safe to remove.
            shutil.rmtree(self._best_scratch_dir, ignore_errors=True)
            self._best_scratch_dir = None
            return

        # Fallback: no improvement was ever recorded (e.g. NaN loss every
        # epoch). Persist the current trainer state so downstream retrieval
        # still works.
        report_dict = self._build_report_dict(trainer, pl_module) or {
            "nb_features": self.nb_features
        }
        with self._get_checkpoint(trainer) as checkpoint:
            tune.report(report_dict, checkpoint=checkpoint)


# --- K-fold path for train_nn ---------------------------------------------
#
# These helpers implement the sequential K-fold + full-data refit branch of
# ``train_nn``. The fold *fits* are intentionally sequential (the per-fold
# feature engineering is fanned out separately via ``process_train_kfold``):
# in K-fold mode all ``cpus_per_trial`` are routed into each fold's
# PyTorch Lightning Trainer via ``torch.set_num_threads`` + DataLoader
# workers, so per-fold parallelism inside PyTorch itself absorbs the trial's
# CPU budget. The trainable emits ``n_splits`` ``tune.report`` calls total:
# K-1 running-aggregate mid-trial reports (no checkpoint) plus one final
# report after the full-data refit carrying the full aggregate and the
# deployable checkpoint. Mid-trial reports let ASHA prune at fold
# boundaries.


def _nn_build_n_units(
    config: Dict[str, Any], n_features: int, nn_type: str, n_classes: Optional[int]
) -> List[int]:
    """Compute the per-layer unit sizes for a fresh ``NeuralNet`` instance.

    Mirrors the layer-shape construction inside single-split ``train_nn``:
    input dim from the design matrix, hidden dims from
    ``config['n_units_hl{i}']``, output dim from ``nn_type`` (regression: 1,
    classification: n_classes, ordinal_regression: n_classes - 1).
    Returns the unit list. Class labels are derived elsewhere via
    :func:`_nn_classes_from_y`.
    """
    n_layers = int(config["n_hidden_layers"])
    if nn_type == "regression":
        output_layer = [1]
    elif nn_type == "classification":
        output_layer = [int(n_classes)]
    else:  # ordinal_regression: CORN reduces classes by 1
        output_layer = [int(n_classes) - 1]
    n_units = (
        [int(n_features)]
        + [int(config[f"n_units_hl{i}"]) for i in range(n_layers)]
        + output_layer
    )
    assert len(n_units) == n_layers + 2
    return n_units


def _nn_classes_from_y(y_full: np.ndarray) -> List[int]:
    """Sorted unique integer classes from a (possibly float) target column.

    Mirrors the rounding-to-long discipline single-split ``train_nn`` uses
    to build its ``classes`` list (the K-fold path runs the LabelEncoder
    upstream of class derivation when the target is non-numeric, so this
    just rounds the already-encoded floats).
    """
    y_tensor = torch.from_numpy(np.asarray(y_full)).float()
    return sorted(set(torch.round(y_tensor).long().cpu().numpy().tolist()))


def _build_neural_net_for_kfold(
    config: Dict[str, Any],
    n_features: int,
    nn_type: str,
    task_type: str,
    classes: Optional[List[int]],
) -> "NeuralNet":
    """Construct a fresh NeuralNet with the same arg shape as single-split.

    Single source of truth for the NeuralNet constructor in the K-fold
    path -- the per-fold model and the refit model are both built through
    this helper so they receive identical hyperparameters.
    """
    n_units = _nn_build_n_units(
        config,
        n_features,
        nn_type,
        n_classes=(len(classes) if classes is not None else None),
    )
    return NeuralNet(
        n_units=n_units,
        learning_rate=config["learning_rate"],
        nn_type=nn_type,
        dropout_rate=config["dropout_rate"],
        weight_decay=config["weight_decay"],
        classes=classes,
        task_type=task_type,
    )


# Lightning metric-name -> ritme metric-name mappings, mirroring the
# ``nn_metrics`` dict the single-split path passes to
# ``NNTuneReportCheckpointCallback``. Kept module-level so the K-fold
# extraction stays symmetric with the single-split report shape.
_NN_REG_METRIC_MAP: Dict[str, str] = {
    "rmse": "rmse",
    "r2": "r2",
    "loss": "loss",
}
_NN_CLASS_METRIC_MAP: Dict[str, str] = {
    "roc_auc_macro_ovr": "roc_auc_macro_ovr",
    "log_loss": "log_loss",
    "f1_macro": "f1_macro",
    "balanced_accuracy": "balanced_accuracy",
    "mcc": "mcc",
    "loss": "loss",
}


def _nn_extract_fold_metrics(
    trainer: Trainer,
    model: "NeuralNet",
    train_loader: DataLoader,
    val_loader: DataLoader,
    task_type: str,
) -> Dict[str, float]:
    """Per-fold metric dict mirroring the single-split nn metric shape.

    Lightning's ``Trainer.validate`` runs the validation-step pipeline on
    the supplied loader and returns the standard ``val_<name>`` callback
    metrics. We run it once on the validation loader (yielding the
    ``*_val`` ritme keys) and once on the training loader (yielding the
    ``*_train`` ritme keys); this reuses ``on_validation_epoch_end``'s
    metric computation for both splits, so the keys exactly match what
    the single-split path reports via ``NNTuneReportCheckpointCallback``.
    """
    metric_map = (
        _NN_REG_METRIC_MAP if task_type == "regression" else _NN_CLASS_METRIC_MAP
    )
    val_result = trainer.validate(model, val_loader, verbose=False)[0]
    train_result = trainer.validate(model, train_loader, verbose=False)[0]
    out: Dict[str, float] = {}
    for lt_name, ritme_name in metric_map.items():
        v_key = f"val_{lt_name}"
        if v_key in val_result:
            out[f"{ritme_name}_val"] = float(val_result[v_key])
        if v_key in train_result:
            out[f"{ritme_name}_train"] = float(train_result[v_key])
    return out


def _extract_best_epoch(early_stop: EarlyStopping, max_epochs: int) -> Optional[int]:
    """Return the epoch at which ``EarlyStopping`` thinks the best score was hit.

    Lightning's ``EarlyStopping`` in this codebase doesn't expose a
    ``best_epoch`` attribute directly. It does set ``stopped_epoch`` to the
    epoch at which training was stopped when the patience window elapses
    without improvement; under that contract the last improvement happened
    ``patience`` epochs earlier (= ``stopped_epoch - patience``). When
    ``stopped_epoch == 0`` (default) early stopping never fired during this
    run, so we return ``None`` -- the K-fold refit then falls back to the
    config's ``epochs`` cap rather than guessing.

    The returned value is clamped to ``[0, max_epochs]`` to avoid emitting a
    negative epoch from a degenerate stopped_epoch < patience case.
    """
    stopped_epoch = int(getattr(early_stop, "stopped_epoch", 0))
    if stopped_epoch <= 0:
        return None
    best_epoch = stopped_epoch - int(early_stop.patience)
    if best_epoch < 0:
        best_epoch = 0
    if best_epoch > int(max_epochs):
        best_epoch = int(max_epochs)
    return best_epoch


def _nn_refit_epochs(
    per_fold_best_epoch: List[Optional[int]], max_epochs_config: int
) -> int:
    """Resolve full-data refit ``max_epochs`` from the K-fold signal.

    Returns ``median(per_fold_best_epoch) + 1`` (with a minimum of 1) when
    every fold's ``EarlyStopping`` triggered. Falls back to
    ``max_epochs_config`` when any fold's early-stop did not fire
    (``best_epoch`` is ``None``).

    The ``+ 1`` matches Lightning's ``Trainer(max_epochs=N)`` convention
    (runs epochs ``0..N-1``), analogous to xgb's :func:`_xgb_refit_rounds`
    where ``best_iteration`` is 0-indexed. Unlike :func:`_xgb_refit_rounds`,
    when no fold's early-stop fires we fall back to ``max_epochs_config``
    rather than the raw median, because a 1-tree booster is a valid xgb
    model but a 0-epoch nn refit isn't. A legitimate ``best_epoch == 0``
    across all folds (early-overfit signal) refits for 1 epoch rather than
    silently inverting to the full epoch budget.
    """
    if any(b is None for b in per_fold_best_epoch):
        return int(max_epochs_config)
    return max(1, int(np.median(per_fold_best_epoch)) + 1)


@contextmanager
def _save_nn_checkpoint(
    refit_model: "NeuralNet", refit_trainer: Trainer
) -> Iterator["ray.train.Checkpoint"]:
    """Yield a Ray Tune :class:`Checkpoint` containing the refit NeuralNet.

    Mirrors :func:`_save_xgb_checkpoint`: write the Lightning checkpoint to
    a temp dir under the filename ``"checkpoint"`` so it lands at the exact
    path :func:`ritme.evaluate_models._get_checkpoint_path` reads
    (``result.checkpoint.to_directory() / "checkpoint"``), wrap that
    directory in a :class:`ray.train.Checkpoint`, and yield it for the
    caller to hand to ``tune.report(metrics=..., checkpoint=...)``.

    The temp dir lives for the duration of the ``with`` block. Ray Tune
    persists the checkpoint contents to durable storage during the
    ``tune.report`` call, so it is safe to let the temp dir vanish at exit.
    """
    with tempfile.TemporaryDirectory(prefix="ritme_nn_refit_") as tmpdir:
        refit_trainer.save_checkpoint(os.path.join(tmpdir, "checkpoint"))
        yield ray.train.Checkpoint.from_directory(tmpdir)


def _run_kfold_nn(
    config: Dict[str, Any],
    train_val: pd.DataFrame,
    target: str,
    host_id: str,
    tax: pd.DataFrame,
    seed_data: int,
    seed_model: int,
    stratify_by: List[str] | None,
    nn_type: str,
    cpus_per_trial: int,
    gpus_per_trial: int,
    task_type: str,
    n_splits: int,
    nn_corn_max_levels: int = DEFAULT_NN_CORN_MAX_LEVELS,
    max_trial_duration_s: Optional[float] = None,
) -> None:
    """Sequential K-fold + full-data refit for ``train_nn`` and its wrappers.

    One ``tune.report`` after each completed fold carrying the running
    aggregate over folds-so-far (so ASHA can prune obviously-bad trials
    without waiting for all K folds), plus a final ``tune.report`` after
    the full-data refit that carries the full aggregate and the deployable
    checkpoint. Refit ``max_epochs`` is resolved from per-fold
    ``EarlyStopping`` signals via :func:`_nn_refit_epochs` -- median
    best-epoch when every fold's early-stop fired, otherwise the config's
    ``epochs`` cap. The refit saves final-epoch weights (no internal
    monitor split, no per-iteration best-state tracking).

    Notes:
        ``NNTuneReportCheckpointCallback`` is intentionally NOT used in the
        K-fold path -- it lives on the single-split path so ASHA can prune
        at sub-epoch granularity there. K-fold trials emit ``n_splits``
        ``tune.report`` calls total (K-1 running-aggregate reports plus one
        final report with checkpoint); ASHA prunes at fold boundaries.

        When ``max_trial_duration_s`` is set, the elapsed wall clock is
        checked at each fold boundary and before the refit; a capped trial
        stops cleanly with the completed folds' aggregate (no checkpoint).
    """
    trial_start = time.monotonic()
    # Match single-split determinism setup (seed scope must mirror what
    # ``train_nn`` does inside the single-split body so reruns from the same
    # ``seed_model`` are reproducible across paths).
    torch.set_num_threads(max(1, int(cpus_per_trial)))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(seed_model, workers=True)
    torch.manual_seed(seed_model)
    random.seed(seed_model)
    np.random.seed(seed_model)

    engineered = process_train_kfold(
        config,
        train_val,
        target,
        host_id,
        tax,
        seed_data,
        n_splits,
        stratify_by=stratify_by,
        n_workers=max(1, min(n_splits + 1, int(cpus_per_trial))),
    )

    # Persist the label encoder once at trial level (not per fold) for
    # classification/ordinal heads so prediction-time inverse transform has
    # what it needs. ``process_train_kfold`` stashes the encoder on
    # ``config['_label_encoder']`` for non-numeric targets; numeric targets
    # leave it absent, in which case ``_save_label_encoder`` is a no-op.
    if nn_type != "regression":
        _save_label_encoder(config)

    # Class set for classification / ordinal heads -- derived once from the
    # full encoded label vector so per-fold models share the same class
    # universe even when a fold's val split is missing some classes.
    classes = (
        _nn_classes_from_y(engineered.y_refit) if nn_type != "regression" else None
    )

    if nn_type == "ordinal_regression":
        _check_nn_corn_levels(len(classes), nn_corn_max_levels)

    # ``refit_n_features`` describes the deployable model's column space
    # (the matrix that ``_save_nn_checkpoint`` ships). Per-fold matrices
    # can have different widths -- see ``process_train_kfold``'s contract:
    # under data-dependent ``data_selection`` (variance / abundance /
    # quantile / ith / topi), per-fold microbial survivors differ from
    # the full-data refit set. The per-fold model must therefore be
    # built against its own fold's matrix width, otherwise
    # ``NeuralNet.input_norm`` (BatchNorm1d) crashes with
    # ``running_mean should contain X elements not Y``.
    refit_n_features = int(engineered.X_refit.shape[1])
    max_epochs_cfg = int(config["epochs"])
    patience = int(config["early_stopping_patience"])
    min_delta = float(config["early_stopping_min_delta"])
    num_workers = max(0, int(cpus_per_trial) - 1)
    accelerator = "gpu" if int(gpus_per_trial) > 0 else "cpu"

    per_fold_metrics: List[Dict[str, float]] = []
    per_fold_best_epoch: List[Optional[int]] = []
    for fold_idx, (X_tr, y_tr, X_va, y_va) in enumerate(engineered.folds):
        if per_fold_metrics and _time_cap_reached(trial_start, max_trial_duration_s):
            _report_time_capped_aggregate(per_fold_metrics, refit_n_features)
            return
        fold_n_features = int(X_tr.shape[1])
        train_loader, val_loader = load_data(
            X_tr, y_tr, X_va, y_va, config, seed_model, num_workers=num_workers
        )
        model = _build_neural_net_for_kfold(
            config, fold_n_features, nn_type, task_type, classes
        )
        early_stop = EarlyStopping(
            monitor="val_loss",
            min_delta=min_delta,
            patience=patience,
            mode="min",
        )
        trainer = Trainer(
            max_epochs=max_epochs_cfg,
            callbacks=[early_stop],
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            deterministic=True,
            accelerator=accelerator,
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        per_fold_metrics.append(
            _nn_extract_fold_metrics(
                trainer, model, train_loader, val_loader, task_type
            )
        )
        per_fold_best_epoch.append(_extract_best_epoch(early_stop, max_epochs_cfg))

        # Running aggregate uses the deployable feature count -- it is the
        # number a downstream evaluator (and the reported best-model row)
        # will see, not the per-fold survivor count.
        _emit_running_fold_aggregate(
            per_fold_metrics, refit_n_features, fold_idx, n_splits
        )

    if _time_cap_reached(trial_start, max_trial_duration_s):
        # All folds finished but the refit no longer fits inside the cap:
        # end without a deployable checkpoint rather than overshooting.
        _report_time_capped_aggregate(per_fold_metrics, refit_n_features)
        return

    aggregated = _aggregate_fold_metrics(per_fold_metrics)
    aggregated["nb_features"] = refit_n_features

    # Full-data refit for the deployable checkpoint. Trainer needs a
    # ``val_dataloaders`` only when its callbacks ask for one; we run with
    # no callbacks (no EarlyStopping, no checkpointing), so passing only the
    # training loader is sufficient.
    refit_epochs = _nn_refit_epochs(per_fold_best_epoch, max_epochs_cfg)
    full_train_loader, _ = load_data(
        engineered.X_refit,
        engineered.y_refit,
        engineered.X_refit[:0],
        engineered.y_refit[:0],
        config,
        seed_model,
        num_workers=num_workers,
    )
    refit_model = _build_neural_net_for_kfold(
        config, refit_n_features, nn_type, task_type, classes
    )
    refit_trainer = Trainer(
        max_epochs=refit_epochs,
        callbacks=[],
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        deterministic=True,
        accelerator=accelerator,
    )
    refit_trainer.fit(refit_model, train_dataloaders=full_train_loader)

    _save_taxonomy(tax)
    with _save_nn_checkpoint(refit_model, refit_trainer) as checkpoint:
        tune.report(metrics=aggregated, checkpoint=checkpoint)


def train_nn(
    config,
    train_val,
    target,
    host_id,
    tax,
    seed_data,
    seed_model,
    stratify_by,
    nn_type="regression",
    cpus_per_trial=1,
    gpus_per_trial=0,
    task_type="regression",
    k_folds: int = 1,
    nn_corn_max_levels: int = DEFAULT_NN_CORN_MAX_LEVELS,
    max_trial_duration_s: Optional[float] = None,
):
    n_splits = int(k_folds or 1)
    if n_splits > 1:
        return _run_kfold_nn(
            config,
            train_val,
            target,
            host_id,
            tax,
            seed_data,
            seed_model,
            stratify_by,
            nn_type,
            cpus_per_trial,
            gpus_per_trial,
            task_type,
            n_splits,
            nn_corn_max_levels=nn_corn_max_levels,
            max_trial_duration_s=max_trial_duration_s,
        )
    # Limit PyTorch threads to Ray-allocated CPUs to avoid oversubscription
    torch.set_num_threads(cpus_per_trial)

    # Force deterministic algorithms and disable benchmark
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set the seed for reproducibility
    seed_everything(seed_model, workers=True)
    torch.manual_seed(seed_model)
    random.seed(seed_model)
    np.random.seed(seed_model)

    # Process dataset
    X_train, y_train, X_val, y_val = process_train(
        config, train_val, target, host_id, tax, seed_data, stratify_by=stratify_by
    )
    if nn_type != "regression":
        _save_label_encoder(config)

    # Scale DataLoader workers to allocated CPUs (reserve at least 1 for training)
    num_workers = max(0, cpus_per_trial - 1)
    train_loader, val_loader = load_data(
        X_train, y_train, X_val, y_val, config, seed_model, num_workers=num_workers
    )

    # Model
    n_layers = config["n_hidden_layers"]
    # output layer defined by target
    if nn_type == "regression":
        output_layer = [1]
        classes = None
    else:
        # nn_type == "classification" or nn_type == "ordinal_regression"

        # this rounds the targets in a the torch-way for consistency (np rounds
        # differently)
        y_tr_t = torch.from_numpy(y_train).float()
        y_val_t = torch.from_numpy(y_val).float()

        classes_train = torch.round(y_tr_t).long().unique().cpu().numpy()
        classes_val = torch.round(y_val_t).long().unique().cpu().numpy()
        classes = sorted(set(classes_train) | set(classes_val))

        if nn_type == "ordinal_regression":
            _check_nn_corn_levels(len(classes), nn_corn_max_levels)

        if nn_type == "classification":
            output_layer = [len(classes)]
        else:  # nn_type == "ordinal_regression"
            # CORN reduces number of classes by 1
            output_layer = [len(classes) - 1]

    n_units = (
        # input layer
        [X_train.shape[1]]
        # hidden layers
        + [config[f"n_units_hl{i}"] for i in range(0, n_layers)]
        # output layer defined by nn_type
        + output_layer
    )
    assert len(n_units) == n_layers + 2

    model = NeuralNet(
        n_units=n_units,
        learning_rate=config["learning_rate"],
        nn_type=nn_type,
        dropout_rate=config["dropout_rate"],
        weight_decay=config["weight_decay"],
        classes=classes,
        task_type=task_type,
    )

    _save_taxonomy(tax)

    if task_type == "regression":
        nn_metrics = {
            "rmse_val": "val_rmse",
            "rmse_train": "train_rmse",
            "r2_val": "val_r2",
            "r2_train": "train_r2",
            "loss_val": "val_loss",
            "loss_train": "train_loss",
        }
        nn_score_attr = "rmse_val"
        nn_score_mode = "min"
    else:
        nn_metrics = {
            "roc_auc_macro_ovr_val": "val_roc_auc_macro_ovr",
            "roc_auc_macro_ovr_train": "train_roc_auc_macro_ovr",
            "log_loss_val": "val_log_loss",
            "log_loss_train": "train_log_loss",
            "f1_macro_val": "val_f1_macro",
            "f1_macro_train": "train_f1_macro",
            "balanced_accuracy_val": "val_balanced_accuracy",
            "balanced_accuracy_train": "train_balanced_accuracy",
            "mcc_val": "val_mcc",
            "mcc_train": "train_mcc",
            "loss_val": "val_loss",
            "loss_train": "train_loss",
        }
        nn_score_attr = "roc_auc_macro_ovr_val"
        nn_score_mode = "max"

    callbacks = [
        NNTuneReportCheckpointCallback(
            metrics=nn_metrics,
            filename="checkpoint",
            on="validation_end",
            nb_features=X_train.shape[1],
            score_attr=nn_score_attr,
            score_mode=nn_score_mode,
        ),
        EarlyStopping(
            monitor="val_loss",
            min_delta=config["early_stopping_min_delta"],
            patience=config["early_stopping_patience"],
            mode="min",
        ),
    ]

    # Trainer
    trainer = Trainer(
        max_epochs=config["epochs"],
        callbacks=callbacks,
        deterministic=True,
        enable_progress_bar=False,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def train_nn_reg(
    config,
    train_val,
    target,
    host_id,
    stratify_by,
    seed_data,
    seed_model,
    tax,
    tree_phylo,
    cpus_per_trial=1,
    gpus_per_trial=0,
    task_type="regression",
    k_folds: int = 1,
    nn_corn_max_levels: int = DEFAULT_NN_CORN_MAX_LEVELS,
    max_trial_duration_s: Optional[float] = None,
):
    train_nn(
        config,
        train_val,
        target,
        host_id,
        tax,
        seed_data,
        seed_model,
        stratify_by,
        nn_type="regression",
        cpus_per_trial=cpus_per_trial,
        gpus_per_trial=gpus_per_trial,
        task_type=task_type,
        k_folds=k_folds,
        nn_corn_max_levels=nn_corn_max_levels,
        max_trial_duration_s=max_trial_duration_s,
    )


def train_nn_class(
    config,
    train_val,
    target,
    host_id,
    stratify_by,
    seed_data,
    seed_model,
    tax,
    tree_phylo,
    cpus_per_trial=1,
    gpus_per_trial=0,
    task_type="classification",
    k_folds: int = 1,
    nn_corn_max_levels: int = DEFAULT_NN_CORN_MAX_LEVELS,
    max_trial_duration_s: Optional[float] = None,
):
    train_nn(
        config,
        train_val,
        target,
        host_id,
        tax,
        seed_data,
        seed_model,
        stratify_by,
        nn_type="classification",
        cpus_per_trial=cpus_per_trial,
        gpus_per_trial=gpus_per_trial,
        task_type=task_type,
        k_folds=k_folds,
        nn_corn_max_levels=nn_corn_max_levels,
        max_trial_duration_s=max_trial_duration_s,
    )


def train_nn_corn(
    config,
    train_val,
    target,
    host_id,
    stratify_by,
    seed_data,
    seed_model,
    tax,
    tree_phylo,
    cpus_per_trial=1,
    gpus_per_trial=0,
    task_type="regression",
    k_folds: int = 1,
    nn_corn_max_levels: int = DEFAULT_NN_CORN_MAX_LEVELS,
    max_trial_duration_s: Optional[float] = None,
):
    # corn model from https://github.com/Raschka-research-group/coral-pytorch
    # Hard-assert the relabel: nn_corn is regression-only. A stale caller
    # passing the old default would otherwise produce an ordinal model that
    # silently reports classification metrics.
    if task_type != "regression":
        raise ValueError(
            f"train_nn_corn is regression-only after the relabel "
            f"(got task_type={task_type!r}). nn_corn was previously "
            f"dual-task; any caller still passing 'classification' must "
            f"be updated."
        )
    train_nn(
        config,
        train_val,
        target,
        host_id,
        tax,
        seed_data,
        seed_model,
        stratify_by,
        nn_type="ordinal_regression",
        cpus_per_trial=cpus_per_trial,
        gpus_per_trial=gpus_per_trial,
        task_type=task_type,
        k_folds=k_folds,
        nn_corn_max_levels=nn_corn_max_levels,
        max_trial_duration_s=max_trial_duration_s,
    )
