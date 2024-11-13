import numpy as np
import pandas as pd
from skbio.stats.composition import clr, ilr

from ritme.feature_space.utils import _biom_to_df, _df_to_biom

PSEUDOCOUNT = 0.000001


def alr(feat: pd.DataFrame, denom_idx: int) -> pd.DataFrame:
    """
    Perform additive Log-Transformation using column with index `denom_idx`
    as common denominator. Pseudocounts must have been added to `feat`. It
    is advisable to use a low-noise variable with considerably high values
    as `denom_idx` column.
    """
    # mat = closure(mat)
    # transform
    feat_t = feat.div(feat.iloc[:, denom_idx], axis=0)
    # remove denominator column
    feat_t.drop(feat_t.columns[denom_idx], axis=1, inplace=True)
    # feat_t = np.delete(feat_t, denom_idx, 0)
    return feat_t


def _find_most_nonzero_feature_idx(data):
    """
    Find the index of the first feature with the most non-zero values.

    Args:
        data (pd.DataFrame): DataFrame containing features.

    Returns:
        int: Index of the feature with the most non-zero values.
    """
    nonzero_counts = (data != 0).sum()
    if nonzero_counts.max() > 0:
        feature_name = nonzero_counts.idxmax()
        return data.columns.get_loc(feature_name)
    else:
        raise ValueError("All features are zero in all samples.")


def presence_absence(feat: pd.DataFrame) -> np.ndarray:
    """
    Convert feature table to presence/absence array
    """
    ft_tab = _df_to_biom(feat)
    ft_tab.pa(inplace=True)
    abspres_df = _biom_to_df(ft_tab)
    return abspres_df.values


def transform_microbial_features(
    feat: pd.DataFrame,
    method: str,
    denom_idx: int = None,
    pseudocount: float = PSEUDOCOUNT,
) -> pd.DataFrame:
    """Transform features with specified `method`."""
    # note to self: clr and ilr methods include initial transformation of feature table
    # to relative abundances (closure). For alr this is not done as transformed result
    # is the same.

    method_map = {
        "clr": clr,
        "ilr": ilr,
        "alr": alr,
        "pa": presence_absence,
        None: None,
    }
    if method not in method_map.keys():
        raise ValueError(f"Method {method} is not implemented yet.")

    # add pseudocounts
    if method in ["clr", "ilr", "alr"]:
        feat = feat.replace(0, pseudocount).copy()

    # transform
    if method == "alr":
        feat_trans = method_map[method](feat, denom_idx)
        feat_trans = feat_trans.add_prefix(f"{method}_")
    elif method is None:
        feat_trans = feat.copy()
    else:  # ilr, clr, pa
        col_names = (
            [f"{method}_{x}" for x in feat.columns]
            if method in ["clr", "pa"]
            else [f"ilr_{x}" for x in range(len(feat.columns) - 1)]
        )
        feat_trans = pd.DataFrame(
            method_map[method](feat), index=feat.index, columns=col_names
        )

    return feat_trans
