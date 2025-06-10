import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer
from matplotlib.transforms import offset_copy
from sklearn.metrics import r2_score, root_mean_squared_error

from ritme._decorators import helper_function, main_function
from ritme.evaluate_models import (
    TunedModel,
    get_predictions,
    load_best_model,
    load_experiment_config,
)

plt.rcParams.update({"font.family": "DejaVu Sans"})
plt.style.use("seaborn-v0_8-pastel")


# ----------------------------------------------------------------------------
@helper_function
def _predict_w_tuned_model(
    tuned_model: TunedModel,
    exp_config: dict,
    train_val: pd.DataFrame,
    test: pd.DataFrame,
):
    # define
    target = exp_config["target"]
    features = [x for x in train_val.columns if x.startswith("F")]

    # create predictions on train_val and test set - note: ft aggregation,
    # selection and transformation are originally also performed on train_val
    # directly before splitting to train-val for hyperparameter search - so no
    # problem with doing this again here
    train_pred = get_predictions(train_val, tuned_model, target, features, "train")
    test_pred = get_predictions(test, tuned_model, target, features, "test")
    all_pred = pd.concat([train_pred, test_pred])

    return all_pred


@helper_function
def _calculate_metrics(all_preds: pd.DataFrame, model_type: str) -> pd.DataFrame:
    metrics = pd.DataFrame()
    for split in ["train", "test"]:
        pred_split = all_preds[all_preds["split"] == split].copy()

        metrics.loc[model_type, f"rmse_{split}"] = root_mean_squared_error(
            pred_split["true"], pred_split["pred"]
        )
        metrics.loc[model_type, f"r2_{split}"] = r2_score(
            pred_split["true"], pred_split["pred"]
        )
    return metrics


@helper_function
def _load_best_tuned_models(path_to_exp):
    # Get model types that yielded tuned models for this experiment
    pattern = re.compile(r"(.*)_best_model\.pkl$")
    tmodel_files = [
        pattern.match(filename).group(1)
        for filename in os.listdir(path_to_exp)
        if pattern.match(filename)
    ]
    if len(tmodel_files) == 0:
        raise ValueError(
            f"No best tuned models found in {path_to_exp}. "
            "Please run cli_find_best_model_config first or evaluate in the "
            "Python API with the function evaluate_tuned_models."
        )
    return tmodel_files


@helper_function
def _plot_scatter_plots(
    all_preds: pd.DataFrame,
    metrics_split: pd.DataFrame,
    axs: list,
    row_idx: int,
    model_name: str,
    only_one_model: bool = False,
):
    i = 0
    colors = {"train": "cornflowerblue", "test": "coral"}

    for split in ["train", "test"]:
        if only_one_model:
            axs_set = axs[i]
        else:
            axs_set = axs[row_idx, i]

        pred_split = all_preds[all_preds["split"] == split].copy()

        # scatter plot with linear regression line
        if pred_split.shape[0] > 100:
            dot_size = 20
            dot_alpha = 0.3
        else:
            dot_size = 50
            dot_alpha = 0.8
        reg = sns.regplot(
            x=pred_split["true"].astype(float),
            y=pred_split["pred"],
            ax=axs_set,
            color=colors[split],
            scatter_kws={"s": dot_size, "alpha": dot_alpha},
            line_kws={"color": "dimgrey"},
        )
        # add model name as a higher‐level y‐label on the first column
        if i == 0:
            axs_set.set_ylabel(rf"$\mathbf{{{model_name}}}$" + "\n\nPredicted")
        else:
            axs_set.set_ylabel("Predicted")

        axs_set.set_xlabel("True")
        # 1:1 ratio between true and predicted values
        x0, x1 = reg.axes.get_xlim()
        y0, y1 = reg.axes.get_ylim()
        lims = [min(x0, y0), max(x1, y1)]
        reg.axes.plot(lims, lims, ":k")

        # add rmse and r2 metrics to plot
        rmse = metrics_split[f"rmse_{split}"].values[0]
        r2 = metrics_split[f"r2_{split}"].values[0]

        trans = offset_copy(axs_set.transData, x=1, y=-1, units="dots")
        axs_set.text(
            lims[0],
            lims[1],
            f"RMSE: {rmse:.2f}\nR²: {r2:.2f}",
            transform=trans,
            color=colors[split],
            ha="left",
            va="top",
        )
        if row_idx == 0:
            axs_set.set_title(f"{split.capitalize()} set")
        i += 1


# ----------------------------------------------------------------------------
@main_function
def evaluate_tuned_models(
    dic_tuned_models: dict[str:TunedModel],
    exp_config: dict,
    train_val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, plt.Figure]:
    """
    Evaluate tuned models in dic_tuned_models on train_val and test data using
    the target and feature settings defined in exp_config.

    Args:
        dic_tuned_models (dict): Dictionary with model_type as keys and
        TunedModel as values.
        exp_config (dict): Dictionary containing the experiment configuration
        incl. target and feature settings.
        train_val (pd.DataFrame): Train and validation set used to infer best
        tuned models before.
        test (pd.DataFrame): Test set.

    Returns:
        pd.DataFrame: R2 and RMSE metrics for train and test split.
        plt.Figure: Matplotlib figure object containing scatter plots of true vs.
        predicted values for each model.
    """
    df_metrics = pd.DataFrame()
    nb_models = len(dic_tuned_models)
    r = 0
    fig, axs = plt.subplots(nb_models, 2, figsize=(12, nb_models * 5), dpi=400)

    for model_type, model in dic_tuned_models.items():
        # create predictions on train_val & test
        preds_model_type = _predict_w_tuned_model(model, exp_config, train_val, test)

        # calculate metrics for models and append
        metrics_split = _calculate_metrics(preds_model_type, model_type)
        df_metrics = pd.concat([df_metrics, metrics_split])

        # create scatterplot true vs. predicted
        _plot_scatter_plots(
            all_preds=preds_model_type,
            metrics_split=metrics_split,
            axs=axs,
            row_idx=r,
            model_name=model_type,
            only_one_model=nb_models == 1,
        )
        r += 1

    return df_metrics, fig


@main_function
def cli_evaluate_tuned_models(
    path_to_exp: str,
    path_to_train_val: str,
    path_to_test: str,
) -> None:
    """
    Evaluate best tuned models in experiment path_to_exp on train_val and test data.

    Args:
        dic_tuned_models (dict): Dictionary with model_type as keys and
        TunedModel as values.
        exp_config (dict): Dictionary containing the experiment configuration
        incl. target and feature settings.
        train_val (str): Path to train and validation set used to infer
        best tuned models before.
        test (str): Path to test set.

    Side Effects:
        Writes the best tuned model evaluation metrics (R2, RMSE) to a file
        "best_metrics.csv" in path_to_exp and the scatter plots of true vs.
        predicted values for each model to a file "best_true_vs_pred.png" in
        path_to_exp.
    """
    # ! load data
    train_val = pd.read_pickle(path_to_train_val)
    test = pd.read_pickle(path_to_test)

    # ! load best tuned models of this experiment
    tmodel_files = _load_best_tuned_models(path_to_exp)

    best_model_dict = {}
    for model_type in tmodel_files:
        best_model_dict[model_type] = load_best_model(model_type, path_to_exp)

    # load experiment config
    config = load_experiment_config(path_to_exp)

    # ! calculate metrics & plot true vs. predicted
    metrics, scatter = evaluate_tuned_models(best_model_dict, config, train_val, test)

    # ! save metrics
    path_to_save = os.path.join(path_to_exp, "best_metrics.csv")
    metrics.to_csv(path_to_save)
    print(f"Metrics were saved in {path_to_save}.")

    # ! save scatter plots
    path_to_save = os.path.join(path_to_exp, "best_true_vs_pred.png")
    scatter.savefig(path_to_save, bbox_inches="tight")
    print(f"Scatter plots were saved in {path_to_save}.")
    plt.close(scatter)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    typer.run(cli_evaluate_tuned_models)
