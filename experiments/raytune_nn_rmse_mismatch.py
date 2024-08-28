import os

import torch
from lightning import LightningModule, Trainer
from ray import init, tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleNN(LightningModule):
    def __init__(self, input_size, hidden_size, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )
        self.learning_rate = learning_rate
        self.train_loss = 0
        self.val_loss = 0
        self.train_predictions = []
        self.train_targets = []
        self.val_predictions = []
        self.val_targets = []
        self.train_log_count = 0
        self.val_log_count = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.train_loss = loss

        self.train_predictions.append(y_hat.detach())
        self.train_targets.append(y.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.val_loss = loss

        self.val_predictions.append(y_hat.detach())
        self.val_targets.append(y.detach())

        self.log("val_loss", loss)

        return {"val_loss": loss}

    def on_train_epoch_end(self):
        all_preds_train = torch.cat(self.train_predictions)
        all_targets_train = torch.cat(self.train_targets)

        rmse_train = torch.sqrt(
            nn.functional.mse_loss(all_preds_train, all_targets_train)
        )
        self.train_log_count += 1
        self.log("train_log_count", self.train_log_count)
        self.log("rmse_train", rmse_train)

        self.train_predictions.clear()
        self.train_targets.clear()

    def on_validation_epoch_end(self):
        all_preds_val = torch.cat(self.val_predictions)
        all_targets_val = torch.cat(self.val_targets)

        rmse_val = torch.sqrt(nn.functional.mse_loss(all_preds_val, all_targets_val))

        self.val_log_count += 1
        self.log("val_log_count", self.val_log_count)
        self.log("rmse_val", rmse_val)

        self.val_predictions.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def train_nn(config, train_data, val_data):
    model = SimpleNN(
        input_size=10, hidden_size=config["hidden_size"], learning_rate=config["lr"]
    )

    train_loader = DataLoader(train_data, batch_size=800)
    val_loader = DataLoader(val_data, batch_size=800)

    trainer = Trainer(
        max_epochs=10,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        val_check_interval=1,
        callbacks=[
            TuneReportCheckpointCallback(
                metrics={
                    "loss": "val_loss",
                    "rmse_val": "rmse_val",
                    "rmse_train": "rmse_train",
                    "val_log_count": "val_log_count",
                    "train_log_count": "train_log_count",
                },
                filename="checkpoint",
                on="validation_end",
                save_checkpoints=True,
            ),
        ],
        deterministic=True,
    )

    trainer.fit(model, train_loader, val_loader)


def main():
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "hidden_size": tune.choice([32, 64, 128]),
    }

    init(
        address="local",
        include_dashboard=False,
        ignore_reinit_error=True,
    )

    torch.manual_seed(42)
    X = torch.randn(1000, 10)
    y = torch.sum(X, dim=1, keepdim=True)
    X_train, y_train = X[:800], y[:800]
    X_val, y_val = X[800:], y[800:]

    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)

    tuner = tune.Tuner(
        tune.with_parameters(train_nn, train_data=train_data, val_data=val_data),
        tune_config=tune.TuneConfig(metric="rmse_val", mode="min", num_samples=2),
        param_space=config,
    )

    results = tuner.fit()

    # get best ray result
    best_result = results.get_best_result("rmse_val", "min", scope="all")
    best_rmse_train = best_result.metrics["rmse_train"]
    best_rmse_val = best_result.metrics["rmse_val"]
    print(f"Best trial final train rmse: {best_rmse_train}")
    print(f"Best trial final validation rmse: {best_rmse_val}")

    # get best model checkpoint
    checkpoint_dir = best_result.checkpoint.path
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")

    # load model
    model = SimpleNN.load_from_checkpoint(checkpoint_path)

    # recalculate rmse_train
    rmse_train_recalc = torch.sqrt(
        nn.functional.mse_loss(model(X_train), y_train)
    ).item()
    print(f"rmse_train_recalc: {rmse_train_recalc}")

    # recalculate rmse_val
    rmse_val_recalc = torch.sqrt(nn.functional.mse_loss(model(X_val), y_val)).item()
    print(f"rmse_val_recalc: {rmse_val_recalc}")

    # assertions
    if not best_rmse_val == rmse_val_recalc:
        raise ValueError("best_rmse_val != rmse_val_recalc")
    if not best_rmse_train == rmse_train_recalc:
        raise ValueError("best_rmse_train != rmse_train_recalc")


if __name__ == "__main__":
    main()
