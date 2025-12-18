from typing import List

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


def _time_label_sort_key(lbl: str) -> tuple[int, int]:
    if lbl == "t0":
        return (0, 0)
    if isinstance(lbl, str) and lbl.startswith("t-"):
        part = lbl[2:]
        if part.isdigit():
            return (1, int(part))
    return (2, 0)


def _extract_time_labels(columns: List[str], feat_prefix: str = "F") -> List[str]:
    """
    Extract time labels from microbial feature column names and return them in a
    stable order.

    Recognizes columns like 'F123__t0', 'F123__t-1', ... for the given
    feature prefix (default 'F').

    Sorting order:
    - 't0' first
    - then 't-<n>' in ascending n
    - any other/unexpected last
    """
    micro_cols = [c for c in columns if c.startswith(feat_prefix)]
    labels = set()
    for c in micro_cols:
        if "__t" in c:
            labels.add(c.split("__", 1)[1])
    return sorted(list(labels), key=_time_label_sort_key)
