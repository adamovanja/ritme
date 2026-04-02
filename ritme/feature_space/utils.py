"""Snapshot column utilities.

Column naming convention for temporal snapshots:
- t0 (current timepoint) columns have NO suffix: ``F0``, ``host_id``, ``target``
- Past snapshot columns are suffixed: ``F0__t-1``, ``host_id__t-2``
- Internal processing always operates on unsuffixed feature names: snapshot
  slicing strips suffixes from past data and ``_add_suffix`` re-adds them
  after processing.
"""

import re
from typing import List

import biom
import pandas as pd

_PAST_SUFFIX_RE = re.compile(r"__t-\d+$")


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
    """Extract time labels from microbial feature column names.

    t0 columns are unsuffixed (e.g. ``F0``); past snapshots are suffixed
    (e.g. ``F0__t-1``).  Returns labels in stable order: ``t0`` first, then
    ``t-1``, ``t-2``, ...
    """
    micro_cols = [c for c in columns if c.startswith(feat_prefix)]
    labels = set()
    has_unsuffixed = False
    for c in micro_cols:
        if "__t" in c:
            labels.add(c.split("__", 1)[1])
        else:
            has_unsuffixed = True
    if has_unsuffixed:
        labels.add("t0")
    return sorted(list(labels), key=_time_label_sort_key)


def _slice_snapshot(ft_all: pd.DataFrame, time_label: str) -> pd.DataFrame:
    """Return a single snapshot slice with unsuffixed column names.

    For ``t0``, selects columns that carry no past-snapshot suffix.
    For past snapshots (``t-1``, ``t-2``, ...), selects columns ending with
    the matching suffix and strips it.
    """
    if time_label == "t0":
        cols = [c for c in ft_all.columns if not _PAST_SUFFIX_RE.search(c)]
        return ft_all[cols].copy()
    suffix = f"__{time_label}"
    cols = [c for c in ft_all.columns if c.endswith(suffix)]
    snap = ft_all[cols].copy()
    snap.columns = [c[: -len(suffix)] for c in cols]
    return snap


def _add_suffix(df: pd.DataFrame, time_label: str) -> pd.DataFrame:
    """Re-add a time suffix to columns after per-snapshot processing.

    t0 columns stay unsuffixed; past snapshots get ``__t-N`` appended.
    """
    if time_label == "t0":
        return df.copy()
    df_s = df.copy()
    df_s.columns = [f"{c}__{time_label}" for c in df_s.columns]
    return df_s
