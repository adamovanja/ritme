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
    suffix_map = {
        "_ith": ["data_selection_q", "data_selection_t"],
        "_topi": ["data_selection_q", "data_selection_t"],
        "_quantile": ["data_selection_i", "data_selection_t"],
        "_threshold": ["data_selection_i", "data_selection_q"],
    }

    data_selection = config.get("data_selection")

    if data_selection is not None:
        for suffix, keys in suffix_map.items():
            if data_selection.endswith(suffix):
                keys_to_update = keys
    else:
        keys_to_update = ["data_selection_i", "data_selection_q", "data_selection_t"]
    for key in keys_to_update:
        config[key] = None

    return config
