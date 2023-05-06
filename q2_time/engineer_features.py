import pandas as pd
from skbio.stats.composition import clr, ilr

PSEUDOCOUNT = 0.0001


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


def transform_features(
    feat: pd.DataFrame,
    method: str,
    denom_idx: int = False,
    pseudocount: float = PSEUDOCOUNT,
) -> pd.DataFrame:
    """Transform features with specified `method`. If `method = alr` index of common
    denominator needs to be provided as `denom_idx`.
    """
    # note to self: clr and ilr methods include initial transformation of feature table
    # to relative abundances (closure). For alr this is not done as transformed result
    # is the same.

    # add pseudocounts
    feat = feat.replace(0, pseudocount).copy()

    # transform & rename columns
    method_map = {"clr": clr, "ilr": ilr, "alr": alr}
    if method not in method_map.keys():
        raise ValueError(f"Method {method} is not implemented yet.")

    # transform
    if method == "alr":
        feat_trans = method_map[method](feat, denom_idx)
        feat_trans = feat_trans.add_prefix(f"{method}_")
    else:
        if method == "clr":
            # ! do we ignore 1 dimension after clr?
            col_names = [f"{method}_{x}" for x in feat.columns]
        elif method == "ilr":
            col_names = [f"ilr_{x}" for x in range(len(feat.columns) - 1)]
        feat_trans = pd.DataFrame(
            method_map[method](feat), index=feat.index, columns=col_names
        )

    return feat_trans
