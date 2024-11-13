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
