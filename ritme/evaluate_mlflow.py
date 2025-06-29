import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns

pio.templates.default = "seaborn"
plt.rcParams.update({"font.family": "DejaVu Sans"})
plt.style.use("tableau-colorblind10")

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)


def create_color_map(df, column, cmap_name="Set3"):
    """
    Create a color map based on unique values in a specified column.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Column name to base the color map on.
        cmap_name (str): Matplotlib colormap name.

    Returns:
        list: List of colors corresponding to each row.
        dict: Mapping of unique values to colors.
    """
    unique_vals = df[column].unique()
    cmap = plt.cm.get_cmap(cmap_name, len(unique_vals))
    color_map = {val: cmap(i) for i, val in enumerate(unique_vals)}
    colors = df[column].map(color_map).tolist()
    return colors, color_map


def barplot_metric(
    trials,
    metric_col,
    metric_name,
    group_col,
    group_name,
    display_trial_name=False,
    top_n=None,
):
    """Plot metric in increasing order with coloring based on group"""
    if top_n is not None:
        trials = trials.nsmallest(top_n, metric_col)

    # prepare color map
    _, color_map = create_color_map(trials, group_col)

    # reset index so we can use it as the x-axis category
    df = trials.reset_index().rename(columns={"index": "trial_name"})

    # define x-axis
    if display_trial_name:
        df["trial_name"] = df["tags.trial_name"].astype(str)
    else:
        df["trial_name"] = df.index.astype(str)
    # seaborn styling
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)

    fig, ax = plt.subplots(figsize=(15, 6), dpi=400)
    sns.barplot(
        x="trial_name",
        y=metric_col,
        hue=group_col,
        data=df,
        palette=color_map,
        dodge=False,
        ax=ax,
    )
    if display_trial_name:
        # rotate x-tick labels for better visibility
        ax.tick_params(axis="x", rotation=90, labelsize=10)
    else:
        # remove x-tick labels
        ax.tick_params(axis="x", which="both", length=0, labelbottom=False)
    ax.set_title(f"{metric_name} by {group_name}", pad=15)
    ax.set_xlabel("Trial", labelpad=10)
    ax.set_ylabel(metric_name, labelpad=10)

    ax.legend(title=group_name, loc="upper right", bbox_to_anchor=(1.15, 1))

    sns.despine(ax=ax)
    plt.tight_layout()
    plt.show()


def violinplot_metric(trials, metric_col, metric_name, group_col, group_name):
    """Plot distribution of metric within each group"""
    _, color_map = create_color_map(trials, group_col)

    fig, ax = plt.subplots(figsize=(15, 6), dpi=400)
    sns.violinplot(
        x=metric_col,
        y=group_col,
        data=trials,
        orient="h",
        palette=color_map,
        cut=0,
        scale="width",
        inner="quartile",
        linewidth=1.5,
        ax=ax,
    )

    ax.set_xlabel(metric_name, labelpad=10)
    ax.set_ylabel(group_name, labelpad=10)
    ax.set_title(f"Distribution of {metric_name} by {group_name}", pad=15)

    plt.tight_layout()
    plt.show()


def _static_scatter(trials, metric_col, metric_name, group_col, group_name):
    _, color_map = create_color_map(trials, "params.model")

    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    _ = sns.scatterplot(
        x="metrics.nb_features",
        y=metric_col,
        hue=group_col,
        palette=color_map,
        data=trials,
        s=80,
        ax=ax,
    )

    ax.set_xlabel("Number of Features", labelpad=10)
    ax.set_ylabel(metric_name, labelpad=10)
    ax.set_title("Model Complexity vs. Performance", pad=15)
    ax.legend(title=group_name)

    sns.despine(ax=ax)
    plt.tight_layout()

    plt.show()


def _dynamic_scatter(trials, metric_col, metric_name, group_col, group_name):
    fig = px.scatter(
        trials,
        x="metrics.nb_features",
        y=metric_col,
        color=group_col,
        hover_name="tags.trial_name",
        labels={
            "metrics.nb_features": "Number of Features",
            metric_col: metric_name,
            group_col: group_name,
        },
        template="plotly_white",
        title="Model Complexity vs. Performance",
    )
    fig.show()


def plot_complexity_vs_metric(
    trials, metric_col, metric_name, group_col, group_name, static=False
):
    """
    Plot model complexity vs. a given metric in a static or dynamic scatter plot.
    """
    if static:
        _static_scatter(trials, metric_col, metric_name, group_col, group_name)
    else:
        _dynamic_scatter(trials, metric_col, metric_name, group_col, group_name)


def parallel_coordinates_plot(
    trials, first_metric_col, first_metric_name, first_metric_cat=False
):
    """
    Create parallel coordinates plot with 3 axes: first_metric_col, model type,
    and RMSE Validation. If first_metric_cat is True, the first metric is
    treated as a categorical variable.
    """
    # pick, rename & encode
    df_pc = (
        trials[[first_metric_col, "params.model", "metrics.rmse_val"]]
        .dropna()
        .rename(
            columns={
                first_metric_col: first_metric_name,
                "params.model": "Model",
                "metrics.rmse_val": "RMSE Validation",
            }
        )
    )
    df_pc["model_numbering"] = df_pc["Model"].astype("category").cat.codes
    cats = list(df_pc["Model"].astype("category").cat.categories)

    if first_metric_cat:
        df_pc[first_metric_name + "_numbering"] = (
            df_pc[first_metric_name].astype("category").cat.codes
        )
        metric_cats = list(df_pc[first_metric_name].astype("category").cat.categories)
        dict_first_metric = dict(
            label=first_metric_name,
            tickvals=list(range(len(metric_cats))),
            ticktext=metric_cats,
            values=df_pc[first_metric_name + "_numbering"],
        )
    else:
        dict_first_metric = dict(
            label=first_metric_name,
            range=[df_pc[first_metric_name].min(), df_pc[first_metric_name].max()],
            values=df_pc[first_metric_name],
        )
    # plot
    fig = go.Figure(
        go.Parcoords(
            line=dict(
                color=df_pc["RMSE Validation"],
                colorscale="Spectral_r",
                showscale=True,
                cmin=df_pc["RMSE Validation"].min(),
                cmax=df_pc["RMSE Validation"].max(),
                colorbar=dict(
                    title="RMSE Validation",
                    thickness=15,
                    lenmode="fraction",
                    len=1.0,
                    yanchor="middle",
                ),
            ),
            dimensions=[
                dict_first_metric,
                dict(
                    label="Model",
                    tickvals=list(range(len(cats))),
                    ticktext=cats,
                    values=df_pc["model_numbering"],
                ),
                dict(
                    label="RMSE Validation",
                    range=[
                        df_pc["RMSE Validation"].min(),
                        df_pc["RMSE Validation"].max(),
                    ],
                    values=df_pc["RMSE Validation"],
                ),
            ],
        )
    )

    fig.update_layout(
        template="seaborn",
        title=f"Performance per {first_metric_name} and model type",
        font=dict(family="Arial", size=12),
        plot_bgcolor="white",
    )

    fig.show()


def _extract_run_history(client, metric, trials, model_type):
    """
    Extract the metric history for all models of the type `model_type` in
    `trials`.
    """
    model_trials = trials[trials["params.model"] == model_type]

    history = []
    for run_id in model_trials["run_id"].unique():
        for m in client.get_metric_history(run_id, metric):
            history.append(
                {
                    # "run_id": run_id,
                    "name": model_trials.loc[
                        model_trials.run_id == run_id, "tags.mlflow.runName"
                    ].iloc[0],
                    "model_type": trials.loc[
                        trials.run_id == run_id, "params.model"
                    ].iloc[0],
                    "step": m.step,
                    metric: m.value,
                    "timestamp": m.timestamp,
                }
            )
    return pd.DataFrame(history)


def plot_metric_history_per_model_type(metric, client, trials):
    """Plot the metric history for each model type."""
    all_model_types = trials["params.model"].unique()
    fig, axes = plt.subplots(
        len(all_model_types),
        1,
        sharex=True,
        figsize=(10, len(all_model_types) * 2),
        dpi=400,
    )

    for i, model_type in enumerate(all_model_types):
        hist_df = _extract_run_history(
            client=client,
            metric=metric,
            trials=trials,
            model_type=model_type,
        )
        hist_df["time"] = pd.to_datetime(hist_df["timestamp"], unit="ms")

        ax = axes[i] if len(all_model_types) > 1 else axes

        sns.lineplot(
            data=hist_df,
            x="time",
            y=metric,
            hue="name",
            palette="Set2",
            marker="o",
            markeredgewidth=0.2,
            ax=ax,
        )
        ax.set_title(model_type)
        ax.set_ylabel("")
        ax.legend(
            title="Trial name",
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
        )

    # only the bottom plot gets an x-axis label
    if len(all_model_types) > 1:
        axes[-1].set_xlabel("Timestamp")
    else:
        axes.set_xlabel("Timestamp")
    # one shared y-axis label
    fig.supylabel(metric)

    plt.tight_layout()
    plt.show()


def plot_avg_history_per_model_type(metric, client, trials):
    """
    Plot the average history of a metric per model type, with ±1 std shading.
    """
    avg_history = {}
    std_history = {}
    for model_type in trials["params.model"].unique():
        hist_df = _extract_run_history(
            client=client, metric=metric, trials=trials, model_type=model_type
        )
        grp = hist_df.groupby("step")[metric]
        avg_history[model_type] = grp.mean()
        std_history[model_type] = grp.std()

    mean_df = pd.DataFrame(avg_history)
    std_df = pd.DataFrame(std_history)
    colors = sns.color_palette("Set2", len(mean_df.columns))

    fig, ax = plt.subplots(figsize=(12, 4), dpi=400)
    for col, c in zip(mean_df.columns, colors):
        ax.plot(
            mean_df.index,
            mean_df[col],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=col,
            color=c,
        )
        ax.fill_between(
            mean_df.index,
            mean_df[col] - std_df[col],
            mean_df[col] + std_df[col],
            alpha=0.2,
            color=c,
        )

    ax.set_title(f"Mean {metric} per model type over training time", pad=15)
    ax.set_xlabel("Step", labelpad=10)
    ax.set_ylabel(f"Mean {metric}", labelpad=10)
    ax.legend(
        title="Model Type",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    plt.tight_layout()
    plt.show()


def extract_run_config(trials):
    """
    Display the run configuration for each trial - only the differing columns
    are displayed
    """
    varying_cols = trials.columns[trials.nunique() > 1]
    varying_cols = varying_cols.drop("artifact_uri")
    return trials[varying_cols]
