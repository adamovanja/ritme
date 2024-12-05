import typer

from ritme.evaluate_tuned_models import cli_evaluate_tuned_models
from ritme.find_best_model_config import cli_find_best_model_config
from ritme.split_train_test import cli_split_train_test

app = typer.Typer()

app.command(name="split-train-test")(cli_split_train_test)
app.command(name="find-best-model-config")(cli_find_best_model_config)
app.command(name="evaluate-tuned-models")(cli_evaluate_tuned_models)

if __name__ == "__main__":
    app()
