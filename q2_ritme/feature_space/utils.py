import biom
import pandas as pd


def _df_to_biom(df: pd.DataFrame) -> biom.Table:
    """
    Convert a pandas DataFrame to a biom.Table object.
    """
    return biom.Table(df.T.values, observation_ids=df.columns, sample_ids=df.index)


def _biom_to_df(biom_tab: biom.Table) -> pd.DataFrame:
    """
    Convert a biom.Table object to a pandas DataFrame.
    """
    return pd.DataFrame(
        biom_tab.matrix_data.toarray().T,
        index=biom_tab.ids(axis="sample"),
        columns=biom_tab.ids(axis="observation"),
    )


def _update_config(config):
    """Adjust data_selection config dependencies by main method selected"""
    data_selection = config.get("data_selection")

    if data_selection is None:
        config["data_selection_i"] = None
        config["data_selection_q"] = None
        config["data_selection_t"] = None
    elif data_selection.endswith("_ith") or data_selection.endswith("_topi"):
        config["data_selection_i"] = config["dsi_option"]
        config["data_selection_q"] = None
        config["data_selection_t"] = None
    elif data_selection.endswith("_quantile"):
        config["data_selection_i"] = None
        config["data_selection_q"] = config["dsq_option"]
        config["data_selection_t"] = None
    elif data_selection.endswith("_threshold"):
        config["data_selection_i"] = None
        config["data_selection_q"] = None
        config["data_selection_t"] = config["dst_option"]

    return config
