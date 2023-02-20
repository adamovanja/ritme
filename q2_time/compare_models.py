import pandas as pd
from IPython.display import display


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

    for tag, conf in dic_models.items():
        df_config_model = pd.DataFrame({tag: conf}).transpose()

        if df_config is None:
            df_config = df_config_model.copy(deep=True)
        else:
            df_config = pd.concat([df_config, df_config_model], axis=0)

    table = df_config.style.apply(_highlight_differing_cols, axis=None)

    return display(table)
