import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.metrics import mean_squared_error


def _highlight_differing_cols(x):
    """
    Function returning color map of differing columns in x
    Original code used as base:
    https://stackoverflow.com/questions/41654949/pandas-style-function-to-highlight-specific-columns
    """
    # copy df to new - original data is not changed
    df = x.copy()

    # extract list of columns that differ between all models
    ls_col = df.columns[df.nunique(dropna=False) > 1].tolist()

    # select default neutral background
    df.loc[:, :] = "background-color: None"

    # mark columns that differ
    df[ls_col] = "color: red"

    # return colored df
    return df


def compare_config(dic_models):
    """
    Compare configurations of models in dic_models
    """
    # todo: adjust this to become a Q2 visualization
    # todo: add unit tests
    df_config = None

    for tag, model in dic_models.items():
        # todo: reconsider what to retrieve as model configuration here
        config = model.get_params()
        df_config_model = pd.DataFrame({tag: config}).transpose()

        if df_config is None:
            df_config = df_config_model.copy(deep=True)
        else:
            df_config = pd.concat([df_config, df_config_model], axis=0)

    table = df_config.style.apply(_highlight_differing_cols, axis=None)

    return display(table)


def eval_simulations(dic_out, set="test"):
    """
    Plot train/test MSE from simulations - for demo purposes for now
    This function relies a lot on the output of model.run_models for now.
    """
    # plot settings
    # todo: set default plot settings across package
    plt.style.use("seaborn-v0_8-colorblind")  # ("tableau-colorblind10")
    titlesize = 14
    labelsize = 13
    ticklabel = 12
    plt.rcParams.update({"font.size": labelsize})

    # hardcoded index given from run_models output
    if set == "test":
        idx = 1
    else:
        idx = 0
    metrics_df = pd.DataFrame()
    for tag, preds in dic_out.items():
        mse = []
        for ind_pred in preds[idx]:  # 1 is test, 0 would be train
            mse.append(mean_squared_error(ind_pred["true"], ind_pred["pred"]))
        metrics_df[tag] = mse

    metrics_df.plot(kind="box", figsize=(12, 6))

    plt.xticks(fontsize=ticklabel)
    plt.yticks(fontsize=ticklabel)
    plt.ylabel("MSE", fontsize=labelsize)
    plt.xlabel("Simulation Tag", fontsize=labelsize)
    plt.title(f"Metrics comparison: {set.upper()}", fontsize=titlesize)
    plt.show()
